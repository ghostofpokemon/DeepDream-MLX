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

class InvertedResidual_Exact(nn.Module):
    def __init__(self, in_ch, exp_ch, out_ch, k, s, se, hs):
        super().__init__()
        self.use_res_connect = (s == 1 and in_ch == out_ch)
        activation = HSwish if hs else nn.ReLU
        
        # Logic for expansion vs non-expansion block structure
        if exp_ch == in_ch: # exp=1 case
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, k, s, padding=(k-1)//2, groups=in_ch, bias=False),
                nn.BatchNorm(in_ch),
                SqueezeExcite(in_ch, 8) if se else nn.Identity(), # 8 is specific to layer 1
                activation()
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
        if self.conv1 and len(self.conv1.layers) > 0:
             out = self.conv1(x)
             out = self.conv2(out)
        else: # Should not happen in this design but for safety
             out = self.conv2(x)

        if self.use_res_connect:
            return x + out
        return out


class MobileNetV3Small_Defined(nn.Module):
    def __init__(self):
        super().__init__()
        
        # featureList.0
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm(16),
            HSwish()
        )
        
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
        
        # Last Conv 576 -> 1024
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
        
        x = self.last_conv(x)
        endpoints['last_conv'] = x
        
        return x, endpoints
    
    def __call__(self, x):
        x, _ = self.forward_with_endpoints(x)
        return x

    def load_npz(self, path):
        print(f"Loading {path}...")
        data = np.load(path, allow_pickle=True)
        keys = set(data.files)
        
        def to_mlx(w):
            return np.transpose(w, (0, 2, 3, 1)) if w.ndim == 4 else w

        def get_tensor(key):
            if key not in keys: return None
            return mx.array(data[key])
            
        # Helper for BN
        def load_bn(mod, key_prefix):
            if f"{key_prefix}.weight" not in keys: return
            mod.weight = get_tensor(f"{key_prefix}.weight")
            mod.bias = get_tensor(f"{key_prefix}.bias")
            mod.running_mean = get_tensor(f"{key_prefix}.running_mean")
            mod.running_var = get_tensor(f"{key_prefix}.running_var")

        # Helper for SE
        def load_se(mod, key_prefix):
            w0 = get_tensor(f"{key_prefix}.fc.0.weight")
            if w0 is None: return
            if w0.ndim == 2: w0 = w0[..., None, None]
            mod.fc.layers[0].weight = to_mlx(w0)
            
            w2 = get_tensor(f"{key_prefix}.fc.2.weight")
            if w2.ndim == 2: w2 = w2[..., None, None]
            mod.fc.layers[2].weight = to_mlx(w2)

        # Layer 0
        if "featureList.0.0.weight" in keys:
            self.layer0.layers[0].weight = to_mlx(get_tensor("featureList.0.0.weight"))
            load_bn(self.layer0.layers[1], "featureList.0.1")

        # Layers 1-11
        for i in range(1, 12):
            src = f"featureList.{i}"
            dst = getattr(self, f"layer{i}")
            
            if i == 1: # exp=1
                if hasattr(dst, 'conv1') and dst.conv1:
                     dst.conv1.layers[0].weight = to_mlx(get_tensor(f"{src}.conv1.0.weight"))
                     load_bn(dst.conv1.layers[1], f"{src}.conv1.1")
                     load_se(dst.conv1.layers[2], f"{src}.conv1.2")
                if hasattr(dst, 'conv2') and dst.conv2:
                    dst.conv2.layers[0].weight = to_mlx(get_tensor(f"{src}.conv2.0.weight"))
                    load_bn(dst.conv2.layers[1], f"{src}.conv2.1")
            else:
                # conv1
                dst.conv1.layers[0].weight = to_mlx(get_tensor(f"{src}.conv1.0.weight"))
                load_bn(dst.conv1.layers[1], f"{src}.conv1.1")
                
                # conv2
                has_se = any(k.startswith(f"{src}.conv2.2") for k in keys)
                
                dst.conv2.layers[0].weight = to_mlx(get_tensor(f"{src}.conv2.0.weight"))
                load_bn(dst.conv2.layers[1], f"{src}.conv2.1")
                
                idx_proj = 4
                if has_se:
                    load_se(dst.conv2.layers[2], f"{src}.conv2.2")
                    idx_proj = 4
                else:
                     idx_proj = 3
                
                dst.conv2.layers[idx_proj].weight = to_mlx(get_tensor(f"{src}.conv2.4.weight"))
                
                bn_key = f"{src}.conv2.5"
                if f"{bn_key}.weight" not in keys: bn_key = f"{src}.conv2.5.lastBN"
                load_bn(dst.conv2.layers[idx_proj+1], bn_key)

        # Layer 12
        self.layer12.layers[0].weight = to_mlx(get_tensor("featureList.12.0.weight"))
        load_bn(self.layer12.layers[1], "featureList.12.1")
        
        # Last Conv
        self.last_conv.weight = to_mlx(get_tensor("last_stage_layers.1.weight"))


class MobileNetV3SmallTrain(MobileNetV3Small_Defined):
    def __init__(self, num_classes, aux_weight=0.3, use_aux=False):
        super().__init__()
        # Standard Head: Pool -> Linear(1024, 1000)
        # Our last_conv outputs 1024 channels.
        self.classifier = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.use_aux = use_aux

    def forward_logits(self, x, train=False):
        x, _ = self.forward_with_endpoints(x)
        # x is (B, 1024, H/32, W/32)
        
        # Global Average Pooling
        x = mx.mean(x, axis=(1, 2)) # (B, 1024)
        
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits, None, None