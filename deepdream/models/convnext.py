import mlx.core as mx
import mlx.nn as nn
import numpy as np


# LayerNorm2d removed (using nn.LayerNorm directly)

class GlobalResponseNorm(nn.Module):
    """Global Response Normalization layer (key to ConvNeXt V2)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # Use weight/bias to match timm parameter names
        self.weight = mx.zeros((dim,))
        self.bias = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, H, W, C]
        # Manual L2 norm to avoid [linalg::svd] error on GPU
        # gx = mx.linalg.norm(x, ord=2, axis=(1, 2), keepdims=True)
        sq = mx.square(x)
        s = mx.sum(sq, axis=(1, 2), keepdims=True)
        gx = mx.sqrt(s)
        
        nx = gx / (gx.mean(axis=-1, keepdims=True) + self.eps)
        return self.weight * (x * nx) + self.bias + x

class ConvNeXtBlock(nn.Module):
    """ConvNeXt V2 Block."""
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        # Depthwise Conv (7x7)
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # MLP: 4x expansion with GRN
        self.mlp_fc1 = nn.Linear(dim, 4 * dim)
        self.mlp_grn = GlobalResponseNorm(4 * dim)
        self.mlp_fc2 = nn.Linear(4 * dim, dim)
        self.act = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        input = x
        x = self.conv_dw(x)
        x = self.norm(x)
        x = self.mlp_fc1(x)
        x = self.act(x)
        x = self.mlp_grn(x)
        x = self.mlp_fc2(x)
        return input + x

class ConvNeXtStage(nn.Module):
    def __init__(self, in_dim, out_dim, depth, downsample=True):
        super().__init__()
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.LayerNorm(in_dim, eps=1e-6),
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
            )
        
        # Use Sequential for blocks to ensure registration
        self.blocks = nn.Sequential(*[ConvNeXtBlock(out_dim) for _ in range(depth)])

    def __call__(self, x: mx.array) -> mx.array:
        if self.downsample:
            x = self.downsample(x)
        x = self.blocks(x)
        return x

class ConvNeXtV2(nn.Module):
    """
    ConvNeXt V2 (Tiny configuration by default).
    """
    def __init__(self, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=1000):
        super().__init__()
        self.depths = depths
        self.dims = dims
        
        # Stem: 4x patch embedding
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6)
        )
        
        # Stages
        stages_list = []
        in_dim = dims[0]
        # Stage 0
        stages_list.append(ConvNeXtStage(in_dim, dims[0], depths[0], downsample=False))
        # Stages 1-3
        for i in range(1, 4):
            stages_list.append(ConvNeXtStage(dims[i-1], dims[i], depths[i], downsample=True))
            
        self.stages = nn.Sequential(*stages_list)
            
        # Head
        self.head_norm = nn.LayerNorm(dims[-1])
        self.head_fc = nn.Linear(dims[-1], num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.features(x)
        x = x.mean(axis=(1, 2)) # Global Pool
        x = self.head_norm(x)
        x = self.head_fc(x)
        return x

    def features(self, x: mx.array) -> mx.array:
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward_with_endpoints(self, x: mx.array) -> tuple[mx.array, dict[str, mx.array]]:
        endpoints = {}
        
        # Stem
        x = self.stem(x)
        endpoints["stem"] = x
        
        # Stages
        # self.stages is nn.Sequential of ConvNeXtStage
        for i, stage in enumerate(self.stages.layers):
            # Downsample if present (accessing internal downsample layer)
            if stage.downsample:
                x = stage.downsample(x)
                endpoints[f"stages.{i}.downsample"] = x
                
            # Blocks
            # stage.blocks is nn.Sequential of ConvNeXtBlock
            for j, block in enumerate(stage.blocks.layers):
                x = block(x)
                endpoints[f"stages.{i}.blocks.{j}"] = x
                
        # Head (for completeness of forward pass, though usually not dreamed on)
        global_pool = x.mean(axis=(1, 2))
        norm = self.head_norm(global_pool)
        logits = self.head_fc(norm)
        
        return logits, endpoints

    def load_npz(self, path):
        """
        Load weights from a compressed numpy file (converted from PyTorch).
        Handles shape transposition for Conv2d (NCHW -> NHWC) and Linear (OI -> IO).
        """
        if isinstance(path, str):
            ws = mx.load(path)
        else:
            ws = path # assume dict
            
        new_ws = {}
        for k, v in ws.items():
            # Handle Conv2d weights: [out, in, k, k] -> [out, k, k, in]
            if "weight" in k and len(v.shape) == 4:
                v = v.transpose(0, 2, 3, 1)
            
            # REMAP KEYS
            # 1. Flattened MLP
            k = k.replace("mlp.fc1", "mlp_fc1")
            k = k.replace("mlp.fc2", "mlp_fc2")
            k = k.replace("mlp.grn", "mlp_grn")
            
            # 2. Sequential Indexing (stages.0 -> stages.layers.0)
            # MLX nn.Sequential parameters are stored under 'layers' attribute usually.
            # But wait, does load_weights expect 'stages.layers.0' or just 'stages.0'?
            # If I use `self.load_weights`, it expects matching keys in the flattening of `self`.
            # A `nn.Sequential` named "stages" will have children "layers.0", "layers.1".
            # So `model.stages.layers.0` corresponds to `stages.0` in timm.
            k = k.replace("stages.", "stages.layers.") 
            k = k.replace("blocks.", "blocks.layers.")
            k = k.replace("stem.", "stem.layers.")
            k = k.replace("downsample.", "downsample.layers.")
            
            # 3. Head remapping
            k = k.replace("head.fc", "head_fc")
            k = k.replace("head.norm", "head_norm")
            
            new_ws[k] = mx.array(v)
            
        self.load_weights(list(new_ws.items()))
        mx.eval(self.parameters())
