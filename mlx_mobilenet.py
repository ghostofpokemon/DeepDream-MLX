import mlx.core as mx
import mlx.nn as nn
import numpy as np

class HSwish(nn.Module):
    def __call__(self, x):
        return x * nn.ReLU()(x + 3.0) / 6.0

class HSigmoid(nn.Module):
    def __call__(self, x):
        return nn.ReLU()(x + 3.0) / 6.0

class SqueezeExcite(nn.Module):
    def __init__(self, in_ch, squeeze_ch, scale_activation=HSigmoid):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, squeeze_ch, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(squeeze_ch, in_ch, 1, bias=False),
            scale_activation()
        )

    def __call__(self, x):
        # Global Average Pooling
        # x: N, H, W, C
        w = mx.mean(x, axis=(1, 2), keepdims=True)
        w = self.fc(w)
        return x * w

class ConvBNActivation(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride, activation=nn.ReLU):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm(out_ch),
            activation()
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, expand_ratio, stride, use_se, use_hs):
        super().__init__()
        hidden_dim = int(in_ch * expand_ratio)
        self.use_res_connect = (stride == 1 and in_ch == out_ch)
        
        activation = HSwish if use_hs else nn.ReLU
        
        layers = []
        if expand_ratio != 1:
            # Pointwise Expand
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
                nn.BatchNorm(hidden_dim),
                activation()
            )
        else:
            self.conv1 = None

        # Depthwise + SE + Pointwise Linear
        # Note: We structure this to match the npz keys:
        # If exp=1: conv1=[DW, BN, SE], conv2=[Proj, BN]
        # If exp>1: conv1=[Exp, BN, Act], conv2=[DW, BN, SE, Proj, BN]
        
        # Wait, the npz structure analysis showed:
        # Exp=1 (Block 1): conv1=[DW, BN, SE], conv2=[Proj, BN]
        # Exp>1 (Block 4): conv1=[Exp, BN, Act], conv2=[DW, BN, SE, Proj, BN]
        
        self.expand_ratio = expand_ratio
        
        if expand_ratio == 1:
            # Block 1 style
            c1_layers = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
                nn.BatchNorm(hidden_dim),
            ]
            if use_se:
                # Squeeze channels? standard is hidden_dim / 4
                squeeze_ch = int(hidden_dim / 4) # Fixed usually? Or in_ch/4? 
                # In pytorch mobilenet_v3_small: 
                # Block 1 (16->16): squeeze=8. (16/2?)
                # The generic rule is often _make_divisible(expanded_channels // 4)
                c1_layers.append(SqueezeExcite(hidden_dim, 8)) # Hardcoded for block 1 usually 8
            
            c1_layers.append(activation()) # Is activation here?
            # Check Block 1 keys again: conv1.0, conv1.1, conv1.2.fc. No explicit activation key. 
            # But Activation is stateless.
            # Standard: DW -> BN -> Act -> SE -> Proj
            
            self.conv1 = nn.Sequential(*c1_layers)
            
            self.conv2 = nn.Sequential(
                nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
                nn.BatchNorm(out_ch)
            )
            
        else:
            # Block > 1 style
            # self.conv1 is the Expand part (defined above)
            
            # self.conv2 is the rest
            c2_layers = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
                nn.BatchNorm(hidden_dim),
            ]
            if use_se:
                # Calculate squeeze ch
                # For MV3 Small, it varies.
                # We might need to pass it in or genericize.
                # Using hidden_dim // 4 for now.
                c2_layers.append(SqueezeExcite(hidden_dim, int(hidden_dim // 4)))
            
            c2_layers.append(activation())
            
            c2_layers.append(nn.Conv2d(hidden_dim, out_ch, 1, bias=False))
            c2_layers.append(nn.BatchNorm(out_ch))
            
            self.conv2 = nn.Sequential(*c2_layers)

    def __call__(self, x):
        identity = x
        
        if self.expand_ratio == 1:
            out = self.conv1(x)
            out = self.conv2(out)
        else:
            out = self.conv1(x)
            out = self.conv2(out)
            
        if self.use_res_connect:
            return x + out
        return out

class MobileNetV3Small(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Config for Small
        # k, exp, out, se, hs, s
        config = [
            [3, 16, 16, True, False, 2],  # 1 (Exp=1, SE=True)
            [3, 72, 24, False, False, 2], # 2
            [3, 88, 24, False, False, 1], # 3
            [5, 96, 40, True, True, 2],   # 4
            [5, 240, 40, True, True, 1],  # 5
            [5, 240, 40, True, True, 1],  # 6
            [5, 120, 48, True, True, 1],  # 7
            [5, 144, 48, True, True, 1],  # 8
            [5, 288, 96, True, True, 2],  # 9
            [5, 576, 96, True, True, 1],  # 10
            [5, 576, 96, True, True, 1],  # 11
        ]
        
        # Block 0: Conv
        self.featureList = []
        # Note: npz has featureList.0 as the first conv.
        # We will use a Python list and register modules manually or use Sequential if possible, 
        # but we want named endpoints.
        
        self.feature0 = ConvBNActivation(3, 16, 3, 2, HSwish)
        
        self.blocks = []
        for i, c in enumerate(config):
            k, exp, out, se, hs, s = c
            ratio = exp / (16 if i==0 else config[i-1][2]) # approx ratio
            # Actually we should pass exact params
            # The class expects ratio? No, I can rewrite init to take hidden_dim
            pass

        # Refactoring InvertedResidual to take hidden_dim instead of ratio for precision
        
    # ... Redefining class below ...

class MobileNetV3Small_Defined(nn.Module):
    def __init__(self):
        super().__init__()
        
        # featureList.0
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm(16),
            HSwish()
        )
        
        self.layers = []
        
        # Helper to make block matching the keys
        def make_block(in_ch, exp_ch, out_ch, k, s, se, hs):
            return InvertedResidual_Exact(in_ch, exp_ch, out_ch, k, s, se, hs)

        # 1: 16 -> 16 (exp 16) SE, RE, s=2
        self.layer1 = make_block(16, 16, 16, 3, 2, True, False)
        # 2: 16 -> 24 (exp 72) -, RE, s=2
        self.layer2 = make_block(16, 72, 24, 3, 2, False, False)
        # 3: 24 -> 24 (exp 88) -, RE, s=1
        self.layer3 = make_block(24, 88, 24, 3, 1, False, False)
        # 4: 24 -> 40 (exp 96) SE, HS, s=2
        self.layer4 = make_block(24, 96, 40, 5, 2, True, True)
        # 5: 40 -> 40 (exp 240) SE, HS, s=1
        self.layer5 = make_block(40, 240, 40, 5, 1, True, True)
        # 6: 40 -> 40 (exp 240) SE, HS, s=1
        self.layer6 = make_block(40, 240, 40, 5, 1, True, True)
        # 7: 40 -> 48 (exp 120) SE, HS, s=1
        self.layer7 = make_block(40, 120, 48, 5, 1, True, True)
        # 8: 48 -> 48 (exp 144) SE, HS, s=1
        self.layer8 = make_block(48, 144, 48, 5, 1, True, True)
        # 9: 48 -> 96 (exp 288) SE, HS, s=2
        self.layer9 = make_block(48, 288, 96, 5, 2, True, True)
        # 10: 96 -> 96 (exp 576) SE, HS, s=1
        self.layer10 = make_block(96, 576, 96, 5, 1, True, True)
        # 11: 96 -> 96 (exp 576) SE, HS, s=1
        self.layer11 = make_block(96, 576, 96, 5, 1, True, True)
        
        # 12: Conv 1x1 96->576 (HS)
        self.layer12 = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm(576),
            HSwish()
        )
        
        # Last Stage
        # last_stage_layers.1 (Conv 1x1 576->1024)
        # Where is 0? maybe 0 is Pool? 
        # keys had last_stage_layers.1.weight.
        # Standard is Pool -> Conv.
        # We'll just define the conv.
        self.last_conv = nn.Conv2d(576, 1024, 1, bias=False)

    def forward_with_endpoints(self, x):
        endpoints = {}
        x = self.layer0(x)
        endpoints['layer0'] = x
        
        x = self.layer1(x)
        endpoints['layer1'] = x
        x = self.layer2(x)
        endpoints['layer2'] = x
        x = self.layer3(x)
        endpoints['layer3'] = x
        x = self.layer4(x)
        endpoints['layer4'] = x
        x = self.layer5(x)
        endpoints['layer5'] = x
        x = self.layer6(x)
        endpoints['layer6'] = x
        x = self.layer7(x)
        endpoints['layer7'] = x
        x = self.layer8(x)
        endpoints['layer8'] = x
        x = self.layer9(x)
        endpoints['layer9'] = x
        x = self.layer10(x)
        endpoints['layer10'] = x
        x = self.layer11(x)
        endpoints['layer11'] = x
        
        x = self.layer12(x)
        endpoints['layer12'] = x
        
        x = self.last_conv(x) # Should be after pool? 
        # DeepDream usually stops before pool/FC.
        # We can include it.
        endpoints['last_conv'] = x
        
        return x, endpoints

    def load_npz(self, path):
        print(f"Loading {path}...")
        data = np.load(path, allow_pickle=True)
        keys = set(data.files)
        
        # Determine style
        is_standard = any(k.startswith("features.") for k in keys)
        
        def to_mlx(w):
            return np.transpose(w, (0, 2, 3, 1)) if w.ndim == 4 else w

        def get_tensor(key):
            if key not in keys: return None
            return mx.array(data[key])

        def load_bn_std(mod, prefix):
            # Standard keys: weight, bias, running_mean, running_var
            if f"{prefix}.weight" not in keys: return
            mod.weight = get_tensor(f"{prefix}.weight")
            mod.bias = get_tensor(f"{prefix}.bias")
            mod.running_mean = get_tensor(f"{prefix}.running_mean")
            mod.running_var = get_tensor(f"{prefix}.running_var")

        def load_se_std(mod, prefix):
            # Standard keys: fc1.weight, fc1.bias, fc2.weight, fc2.bias
            # PyTorch SE: fc1 (Conv), ReLU, fc2 (Conv), Scale
            # Prefix example: features.1.block.1 (if SE is item 1)
            # Actually PyTorch SqueezeExcitation has .fc1 and .fc2
            
            # We need to find the SE module keys. 
            # Usually `prefix.fc1.weight`
            
            w1 = get_tensor(f"{prefix}.fc1.weight")
            if w1 is None: return
            if w1.ndim == 2: w1 = w1[..., None, None] # (out, in, 1, 1)
            mod.fc.layers[0].weight = to_mlx(w1)
            mod.fc.layers[0].bias = get_tensor(f"{prefix}.fc1.bias")
            
            w2 = get_tensor(f"{prefix}.fc2.weight")
            if w2.ndim == 2: w2 = w2[..., None, None]
            mod.fc.layers[2].weight = to_mlx(w2)
            mod.fc.layers[2].bias = get_tensor(f"{prefix}.fc2.bias")

        # --- Standard Loading (Torchvision) ---
        if is_standard:
            print("Detected Standard Torchvision weights.")
            # Layer 0
            # features.0.0 (Conv), features.0.1 (BN)
            self.layer0.layers[0].weight = to_mlx(get_tensor("features.0.0.weight"))
            load_bn_std(self.layer0.layers[1], "features.0.1")
            
            # Layers 1-11
            # PyTorch blocks: features.1 ... features.11
            # Structure:
            #  If exp=1 (Layer 1): block[0]=DW, block[1]=SE, block[2]=Proj
            #  If exp!=1: block[0]=Exp, block[1]=DW, block[2]=SE, block[3]=Proj
            
            for i in range(1, 12):
                src = f"features.{i}.block"
                dst = getattr(self, f"layer{i}")
                
                # Check if expand exists (Block 0 is expand if exp!=1)
                has_expand = f"{src}.0.0.weight" in keys and dst.conv1 is not None
                
                # Map indices based on presence of Expand and SE
                # This is tricky because PyTorch uses a sequential list.
                # We need to identify which index corresponds to what.
                # Count convs/modules in the block?
                
                # Simple heuristic based on layer index known config
                # Layer 1 (i=1): exp=1. block.0=DW, block.1=SE, block.2=Proj
                # Layer 2 (i=2): exp!=1. block.0=Exp, block.1=DW, block.2=Proj (No SE)
                # Layer 4 (i=4): exp!=1. block.0=Exp, block.1=DW, block.2=SE, block.3=Proj
                
                idx = 0
                
                # 1. Expand (if applicable)
                if has_expand:
                    # self.conv1 is the expand layer
                    dst.conv1.layers[0].weight = to_mlx(get_tensor(f"{src}.{idx}.0.weight"))
                    load_bn_std(dst.conv1.layers[1], f"{src}.{idx}.1")
                    idx += 1
                
                # 2. Depthwise
                # dst.conv2[0] or dst.conv1[0] depending on structure
                # My InvertedResidual_Exact:
                # if exp=1: conv1=DW+BN+SE, conv2=Proj+BN
                # if exp!=1: conv1=Exp+BN, conv2=DW+BN+SE+Proj+BN
                
                if i == 1: # Exp=1
                    # DW is at block.0
                    dst.conv1.layers[0].weight = to_mlx(get_tensor(f"{src}.0.0.weight"))
                    load_bn_std(dst.conv1.layers[1], f"{src}.0.1")
                    
                    # SE is at block.1
                    # dst.conv1.layers[2] is SE
                    load_se_std(dst.conv1.layers[2], f"{src}.1")
                    
                    # Proj is at block.2
                    # dst.conv2.layers[0] is Proj
                    dst.conv2.layers[0].weight = to_mlx(get_tensor(f"{src}.2.0.weight"))
                    load_bn_std(dst.conv2.layers[1], f"{src}.2.1")
                    
                else: # Exp != 1
                    # Exp loaded above at idx 0. idx is now 1.
                    # DW is at block.1
                    # dst.conv2 structure: DW, BN, [SE], Act, Proj, BN
                    
                    dst.conv2.layers[0].weight = to_mlx(get_tensor(f"{src}.{idx}.0.weight"))
                    load_bn_std(dst.conv2.layers[1], f"{src}.{idx}.1")
                    idx += 1
                    
                    # Check for SE
                    # SE usually next if it exists.
                    # My class knows if it has SE.
                    # Check dst.conv2.layers[2] type
                    is_se_layer = isinstance(dst.conv2.layers[2], SqueezeExcite)
                    
                    proj_idx = 3 # DW, BN, Act, Proj
                    if is_se_layer:
                        load_se_std(dst.conv2.layers[2], f"{src}.{idx}")
                        idx += 1 # Move past SE
                        proj_idx = 4 # DW, BN, SE, Act, Proj
                    
                    # Project
                    dst.conv2.layers[proj_idx].weight = to_mlx(get_tensor(f"{src}.{idx}.0.weight"))
                    load_bn_std(dst.conv2.layers[proj_idx+1], f"{src}.{idx}.1")

            # Layer 12: features.12.0 (Conv), features.12.1 (BN)
            self.layer12.layers[0].weight = to_mlx(get_tensor("features.12.0.weight"))
            load_bn_std(self.layer12.layers[1], "features.12.1")
            
            # Last Conv: classifier.1? Or last_stage?
            # Standard MV3 Small:
            # features(0-12) -> pool -> classifier(0:Lin, 1:Dropout, 2:Lin, 3:Drop)
            # Wait, where is the 576->1024 conv? 
            # In PyTorch MV3 Small, features.12 outputs 576.
            # There is NO 1024 conv in features?
            # The paper says: 576 -> Pool -> 1024 (Linear/Conv1x1) -> 1000
            # Torchvision implementation: 
            # `classifier` block: 
            # 0: Linear(576, 1024)
            # 1: HardSwish
            # 2: Dropout
            # 3: Linear(1024, num_classes)
            
            # So `last_conv` in my definition (Conv 576->1024) corresponds to `classifier.0` Linear.
            # Convert Linear to Conv1x1
            w_last = get_tensor("classifier.0.weight") # (1024, 576)
            if w_last is not None:
                w_last = w_last[..., None, None] # (1024, 576, 1, 1)
                w_last = to_mlx(w_last) # (1024, 1, 1, 576)
                self.last_conv.weight = w_last
                if "classifier.0.bias" in keys:
                    self.last_conv.bias = get_tensor("classifier.0.bias")
            
            print("Loaded Standard weights successfully.")
            return

        # --- Custom Loading (Existing logic) ---
        
        # Helper for BN
        def load_bn(mod, key_prefix):
            if f"{key_prefix}.weight" not in keys:
                print(f"Warning: Missing BN weights for {key_prefix}")
                return
            mod.weight = mx.array(data[f"{key_prefix}.weight"])
            mod.bias = mx.array(data[f"{key_prefix}.bias"])
            mod.running_mean = mx.array(data[f"{key_prefix}.running_mean"])
            mod.running_var = mx.array(data[f"{key_prefix}.running_var"])

        # Helper for SE
        def load_se(mod, key_prefix):
            w0_key = f"{key_prefix}.fc.0.weight"
            if w0_key not in keys: return
            
            w0 = data[w0_key]
            if w0.ndim == 2: w0 = w0[..., None, None] # (out, in, 1, 1)
            
            # mod.fc is Sequential. Use layers directly to be safe
            mod.fc.layers[0].weight = mx.array(to_mlx(w0))
            # mod.fc.layers[0].bias = mx.array(data[f"{key_prefix}.fc.0.bias"]) # No bias in this model
            
            w2 = data[f"{key_prefix}.fc.2.weight"]
            if w2.ndim == 2: w2 = w2[..., None, None]
            mod.fc.layers[2].weight = mx.array(to_mlx(w2))
            # mod.fc.layers[2].bias = mx.array(data[f"{key_prefix}.fc.2.bias"]) # No bias

        # Layer 0
        if "featureList.0.0.weight" in keys:
            self.layer0.layers[0].weight = mx.array(to_mlx(data["featureList.0.0.weight"]))
            load_bn(self.layer0.layers[1], "featureList.0.1")
        else:
            print("Error: featureList.0.0.weight not found")

        # Layers 1-11
        for i in range(1, 12):
            src = f"featureList.{i}"
            dst = getattr(self, f"layer{i}")
            
            # If exp == 1 (Layer 1)
            if i == 1: 
                # conv1: DW (0) + BN (1) + SE (2)
                if hasattr(dst, 'conv1') and dst.conv1:
                     # Access layers via .layers for Sequential in MLX? 
                     # nn.Sequential has .layers list.
                     dst.conv1.layers[0].weight = mx.array(to_mlx(data[f"{src}.conv1.0.weight"]))
                     load_bn(dst.conv1.layers[1], f"{src}.conv1.1")
                     load_se(dst.conv1.layers[2], f"{src}.conv1.2")
                
                # conv2: Proj (0) + BN (1)
                if hasattr(dst, 'conv2') and dst.conv2:
                    dst.conv2.layers[0].weight = mx.array(to_mlx(data[f"{src}.conv2.0.weight"]))
                    load_bn(dst.conv2.layers[1], f"{src}.conv2.1")
            else:
                # conv1: Exp (0) + BN (1)
                dst.conv1.layers[0].weight = mx.array(to_mlx(data[f"{src}.conv1.0.weight"]))
                load_bn(dst.conv1.layers[1], f"{src}.conv1.1")
                
                # conv2: DW (0) + BN (1) + SE? (2) + Proj (4) + BN (5)
                has_se = any(k.startswith(f"{src}.conv2.2") for k in keys)
                
                dst.conv2.layers[0].weight = mx.array(to_mlx(data[f"{src}.conv2.0.weight"]))
                load_bn(dst.conv2.layers[1], f"{src}.conv2.1")
                
                idx_proj = 4
                if has_se:
                    load_se(dst.conv2.layers[2], f"{src}.conv2.2")
                    idx_proj = 4 # If SE is at 2, Act is 3 (virtual), Proj is 4.
                else:
                     # Without SE: DW, BN, Act, Proj, BN
                     # Indices in dst.conv2.layers: 0, 1, 2(Act), 3(Proj), 4(BN)
                     # Keys in data: conv2.0, conv2.1, conv2.4, conv2.5
                     idx_proj = 3
                
                # Load Proj
                proj_weight = data[f"{src}.conv2.4.weight"]
                dst.conv2.layers[idx_proj].weight = mx.array(to_mlx(proj_weight))
                
                # Last BN
                bn_key = f"{src}.conv2.5"
                if f"{bn_key}.weight" not in keys:
                    bn_key = f"{src}.conv2.5.lastBN"
                
                load_bn(dst.conv2.layers[idx_proj+1], bn_key)

        # Layer 12
        self.layer12.layers[0].weight = mx.array(to_mlx(data["featureList.12.0.weight"]))
        load_bn(self.layer12.layers[1], "featureList.12.1")
        
        # Last Conv
        self.last_conv.weight = mx.array(to_mlx(data["last_stage_layers.1.weight"]))
        # No BN for last conv? usually there is.
        # Dump keys: only `last_stage_layers.1.weight`. No bias? No BN?
        # It might be just Conv.

class InvertedResidual_Exact(nn.Module):
    def __init__(self, in_ch, exp_ch, out_ch, k, s, se, hs):
        super().__init__()
        self.use_res_connect = (s == 1 and in_ch == out_ch)
        activation = HSwish if hs else nn.ReLU
        
        if exp_ch == in_ch: # exp=1 case
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, k, s, padding=(k-1)//2, groups=in_ch, bias=False),
                nn.BatchNorm(in_ch),
                SqueezeExcite(in_ch, 8) if se else nn.Identity(), # 8 is specific to layer 1
                activation() # Is activation here in exp=1 case?
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm(out_ch)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, exp_ch, 1, bias=False),
                nn.BatchNorm(exp_ch),
                activation()
            )
            
            c2 = [
                nn.Conv2d(exp_ch, exp_ch, k, s, padding=(k-1)//2, groups=exp_ch, bias=False),
                nn.BatchNorm(exp_ch),
            ]
            if se:
                c2.append(SqueezeExcite(exp_ch, exp_ch//4))
            
            c2.append(activation())
            c2.append(nn.Conv2d(exp_ch, out_ch, 1, bias=False))
            c2.append(nn.BatchNorm(out_ch))
            
            self.conv2 = nn.Sequential(*c2)
    
    def __call__(self, x):
        out = self.conv2(self.conv1(x)) if self.conv1 else self.conv2(x) # Wait, if exp=1 conv1 is not None in my Exact class
        # But wait, my Exact class puts DW in conv1 if exp=1.
        if self.use_res_connect:
            return x + out
        return out

