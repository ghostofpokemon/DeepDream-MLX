
import mlx.core as mx
import mlx.nn as nn
import math
import numpy as np

# EfficientNet B0 Params (roughly)
# width_coefficient=1.0, depth_coefficient=1.0, resolution=224, dropout=0.2

class Swish(nn.Module):
    def __call__(self, x):
        return x * nn.Sigmoid()(x)

class SqueezeExcite(nn.Module):
    def __init__(self, in_ch, squeeze_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, squeeze_ch, 1, bias=True)
        self.act1 = Swish()
        self.conv2 = nn.Conv2d(squeeze_ch, in_ch, 1, bias=True)
        self.act2 = nn.Sigmoid()

    def __call__(self, x):
        # Global Avg Pool
        w = mx.mean(x, axis=(1, 2), keepdims=True)
        w = self.act1(self.conv1(w))
        w = self.act2(self.conv2(w))
        return x * w

class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio, kernel_size, stride, se_ratio=0.25, drop_connect_rate=0.0):
        super().__init__()
        self.stride = stride
        self.use_res_connect = (self.stride == 1 and in_ch == out_ch)
        self.drop_connect_rate = drop_connect_rate

        hidden_dim = in_ch * expand_ratio
        
        layers = []
        # Expansion
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_ch, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm(hidden_dim))
            layers.append(Swish())
        
        # Depthwise
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm(hidden_dim))
        layers.append(Swish())
        
        # SE
        if 0 < se_ratio <= 1:
            layers.append(SqueezeExcite(hidden_dim, max(1, int(in_ch * se_ratio))))
            
        # Pointwise
        layers.append(nn.Conv2d(hidden_dim, out_ch, 1, bias=False))
        layers.append(nn.BatchNorm(out_ch))
        
        self.block = nn.Sequential(*layers)

    def __call__(self, x):
        if self.use_res_connect:
            out = self.block(x)
            # Drop Connect logic would go here if training, usually skipped for inference/dreaming
            return x + out
        else:
            return self.block(x)

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # B0 Config: (expand_ratio, channels, repeats, stride, kernel)
        settings = [
            (1,  16, 1, 1, 3),
            (6,  24, 2, 2, 3),
            (6,  40, 2, 2, 5),
            (6,  80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm(32),
            Swish()
        )

        # Blocks
        self.blocks = []
        in_ch = 32
        for expand, out_ch, repeats, stride, kernel in settings:
            for i in range(repeats):
                s = stride if i == 0 else 1
                self.blocks.append(MBConvBlock(in_ch, out_ch, expand, kernel, s))
                in_ch = out_ch
        
        # Wrap blocks in Sequential for easier inspection, or keep list? 
        # MLX nn.Sequential expects *args or list.
        # But for deep dream, having them named might be nicer? 
        # For now, Sequential is standard.
        self.features = nn.Sequential(*self.blocks)

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm(1280),
            Swish(),
            # Pooling done in forward
            # Flatten
            # Linear
        )
        self.classifier = nn.Sequential(
             nn.Dropout(0.2),
             nn.Linear(1280, num_classes)
        )

    def forward_with_endpoints(self, x):
        endpoints = {}
        
        x = self.stem(x)
        endpoints['stem'] = x
        
        # Iterate blocks for endpoint capture
        for i, block in enumerate(self.features.layers):
            x = block(x)
            endpoints[f'block_{i}'] = x
            
        x = self.head(x)
        endpoints['head'] = x
        
        return x, endpoints

    def __call__(self, x):
        x, _ = self.forward_with_endpoints(x)
        x = mx.mean(x, axis=(1, 2))
        return self.classifier(x)

    def load_npz(self, path):
        if not path.endswith(".npz"):
            path = path + ".npz"
        
        try:
            data = np.load(path)
        except Exception as e:
            print(f"Error loading weights from {path}: {e}")
            return

        def load_weight(key, transpose=False):
            if key not in data:
                # Try finding it with num_batches_tracked removal or similar?
                # For now, strict.
                print(f"Warning: {key} not found in npz.")
                return None
            val = data[key]
            if transpose and val.ndim == 4:
                # PyTorch (Out, In, H, W) -> MLX (Out, H, W, In)
                val = val.transpose(0, 2, 3, 1)
            return mx.array(val)

        def load_bn(prefix, bn):
            w = load_weight(f"{prefix}.weight")
            b = load_weight(f"{prefix}.bias")
            m = load_weight(f"{prefix}.running_mean")
            v = load_weight(f"{prefix}.running_var")
            if w is not None: bn.weight = w
            if b is not None: bn.bias = b
            if m is not None: bn.running_mean = m
            if v is not None: bn.running_var = v

        def load_conv(prefix, conv):
            w = load_weight(f"{prefix}.weight", transpose=True)
            if w is not None: conv.weight = w
            if hasattr(conv, "bias") and conv.bias is not None:
                b = load_weight(f"{prefix}.bias")
                if b is not None: conv.bias = b

        # 1. Stem
        # PyTorch: features.0.0 (Conv), features.0.1 (BN)
        load_conv("features.0.0", self.stem.layers[0])
        load_bn("features.0.1", self.stem.layers[1])

        # 2. Blocks
        # PyTorch stages 1..7 map to our flattened self.features list.
        # We need to know how many blocks per stage to iterate correctly.
        # Settings from __init__: (expand, out_ch, repeats, stride, kernel)
        settings = [
            (1,  16, 1, 1, 3), # Stage 1 (features.1)
            (6,  24, 2, 2, 3), # Stage 2 (features.2)
            (6,  40, 2, 2, 5), # Stage 3 (features.3)
            (6,  80, 3, 2, 3), # Stage 4 (features.4)
            (6, 112, 3, 1, 5), # Stage 5 (features.5)
            (6, 192, 4, 2, 5), # Stage 6 (features.6)
            (6, 320, 1, 1, 3), # Stage 7 (features.7)
        ]

        flat_idx = 0
        for stage_idx, (expand, c, repeats, s, k) in enumerate(settings):
            pt_stage = stage_idx + 1 # PyTorch features.1 is first stage
            
            for block_idx in range(repeats):
                # PyTorch: features.{stage}.{block}
                # MLX: self.features[flat_idx] (which is self.blocks[flat_idx].block)
                
                prefix = f"features.{pt_stage}.{block_idx}.block"
                target_block = self.features.layers[flat_idx].block
                
                # Iterate layers in our MLX Sequential block
                # Structure:
                # [Expand Conv, BN, Swish] (Optional)
                # [Depthwise Conv, BN, Swish]
                # [SE] (Optional)
                # [Pointwise Conv, BN]
                
                layer_ptr = 0
                pt_ptr = 0 # PyTorch block.0, block.1 etc
                
                # Expansion
                # Note: PyTorch Block structure:
                # 0: ConvBNActivation (Expand) -> 0: Conv, 1: BN, 2: Act
                # 1: ConvBNActivation (Depthwise)
                # 2: SqueezeExcite
                # 3: ConvBNActivation (Pointwise)
                
                # We need to map carefully.
                # Actually, based on inspect_keys: features.5.0.block.0.0.weight
                # This means block.0 is the expand module, and inside it 0 is conv, 1 is bn.
                
                if expand != 1:
                    load_conv(f"{prefix}.{pt_ptr}.0", target_block.layers[layer_ptr])
                    load_bn(f"{prefix}.{pt_ptr}.1", target_block.layers[layer_ptr+1])
                    layer_ptr += 3
                    pt_ptr += 1 
                
                # This mapping is tricky without exact introspection. 
                # Let's use the assumption that `block` sequence matches.
                # If expand!=1:
                #   MLX: layers[0]=Conv, layers[1]=BN
                #   PT:  block.0.0=Conv, block.0.1=BN
                #   Index increment: MLX +3, PT +1 (since PT groups them into sub-sequential 0)
                

                
                # Depthwise
                # MLX: layers[layer_ptr], [layer_ptr+1]
                # PT: block.{pt_ptr}.0, block.{pt_ptr}.1
                load_conv(f"{prefix}.{pt_ptr}.0", target_block.layers[layer_ptr])
                load_bn(f"{prefix}.{pt_ptr}.1", target_block.layers[layer_ptr+1])
                layer_ptr += 3
                pt_ptr += 1
                
                # SE
                # MLX: layers[layer_ptr] is SqueezeExcite
                # PT: block.{pt_ptr} is SqueezeExcitation
                # Inside SE:
                # MLX: conv1, conv2
                # PT: fc1, fc2
                if 0 < 0.25 <= 1: # Default ratio logic used in init
                    se_layer = target_block.layers[layer_ptr]
                    load_conv(f"{prefix}.{pt_ptr}.fc1", se_layer.conv1)
                    load_conv(f"{prefix}.{pt_ptr}.fc2", se_layer.conv2)
                    layer_ptr += 1
                    pt_ptr += 1
                
                # Pointwise
                # MLX: layers[layer_ptr], [layer_ptr+1]
                # PT: block.{pt_ptr}.0, block.{pt_ptr}.1
                load_conv(f"{prefix}.{pt_ptr}.0", target_block.layers[layer_ptr])
                load_bn(f"{prefix}.{pt_ptr}.1", target_block.layers[layer_ptr+1])
                
                flat_idx += 1
        
        # 3. Head & Classifier
        # features.8 -> self.head
        # PT: features.8.0 (Conv), 8.1 (BN)
        load_conv("features.8.0", self.head.layers[0])
        load_bn("features.8.1", self.head.layers[1])
        
        # Classifier
        # PT: classifier.1 (Linear) - index 0 is dropout
        # MLX: self.classifier.layers[1]
        
        # Linear weight in PT: (Out, In)
        # MLX Linear: (In, Out) -> Transpose!
        l_w = load_weight("classifier.1.weight")
        l_b = load_weight("classifier.1.bias")
        if l_w is not None:
             self.classifier.layers[1].weight = l_w.T
        if l_b is not None:
             self.classifier.layers[1].bias = l_b
             
        print(f"Loaded EfficientNetB0 weights from {path}")
