
import mlx.core as mx
import mlx.nn as nn
import numpy as np

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super().__init__()
        # In PyTorch: norm1, relu1, conv1, norm2, relu2, conv2
        self.norm1 = nn.BatchNorm(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = float(drop_rate)

    def __call__(self, x):
        # Concatenation happens in the block
        # x is a list of tensors in PT impl? No, usually cat-ed before.
        # But here input x is the concatenated features from previous layers.
        
        out = self.norm1(x)
        out = nn.relu(out)
        out = self.conv1(out)
        
        out = self.norm2(out)
        out = nn.relu(out)
        out = self.conv2(out)
        
        # Dropout?
        # if self.drop_rate > 0:
        #    out = nn.Dropout(self.drop_rate)(out)
        
        return out

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate
            )
            self.layers.append(layer)
        
        # We store as list to iterate carefully or Sequential?
        # Sequential doesn't handle the concatenation logic inherent to DenseNet easily
        # unless each layer outputs cat(in, out).
        # Standard PT impl: Loop, cat outputs.
        
        # For visualization simplicity, we can keep them in a list but "register" them?
        # In MLX, we must assign to self attributes or a list that is traversed.
        # self.block = nn.Sequential(*self.layers) would just chain them, which is WRONG for DenseNet.
        # DenseNet: y = layer(x); x = cat([x, y])
        
        # So we explicitly name them
        for i, layer in enumerate(self.layers):
            setattr(self, f"denselayer{i+1}", layer)

    def __call__(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(mx.concatenate(features, axis=-1)) # MLX is NHWC, so cat on -1 (C)
            features.append(new_feat)
        return mx.concatenate(features, axis=-1)

class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def __call__(self, x):
        out = self.norm(x)
        out = nn.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out

class DenseNet121(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super().__init__()

        # First convolution
        self.features_start = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm(num_init_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense Blocks
        num_features = num_init_features
        self.blocks = []
        
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            # Register explicitly for parameters() scan?
            # self.blocks.append(block)
            setattr(self, f"denseblock{i+1}", block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )
                setattr(self, f"transition{i+1}", trans)
                num_features = num_features // 2

        # Final Batch Norm
        self.norm5 = nn.BatchNorm(num_features)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def __call__(self, x):
        # Initial
        out = self.features_start(x)
        
        # Blocks (Manually sequenced based on config knowledge)
        # Block 1
        out = self.denseblock1(out)
        out = self.transition1(out)
        
        # Block 2
        out = self.denseblock2(out)
        out = self.transition2(out)
        
        # Block 3
        out = self.denseblock3(out)
        out = self.transition3(out)
        
        # Block 4
        out = self.denseblock4(out)
        
        out = self.norm5(out)
        out = nn.relu(out)
        
        out = mx.mean(out, axis=(1, 2)) # Global Avg Pool
        out = self.classifier(out)
        return out
    
    def forward_with_endpoints(self, x):
        endpoints = {}
        
        # Note: self.features_start contains [Conv, BN, ReLU, Pool]
        # We can run them? Or just run sequential.
        x = self.features_start(x)
        endpoints['initial'] = x
        
        x = self.denseblock1(x)
        endpoints['denseblock1'] = x
        x = self.transition1(x)
        endpoints['transition1'] = x
        
        x = self.denseblock2(x)
        endpoints['denseblock2'] = x
        x = self.transition2(x)
        endpoints['transition2'] = x
        
        x = self.denseblock3(x)
        endpoints['denseblock3'] = x
        x = self.transition3(x)
        endpoints['transition3'] = x
        
        x = self.denseblock4(x)
        endpoints['denseblock4'] = x
        
        return x, endpoints

    def load_npz(self, path):
        if not path.endswith(".npz"): path += ".npz"
        try:
            data = np.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return

        def load_w(key, transpose=False):
            if key not in data:
                # DenseNet in torchvision sometimes has 'features.' prefix
                # or just 'denseblock1...'? 
                # Usually 'features.denseblock1...' if it has a features container.
                # Try adding/removing 'features.'
                if f"features.{key}" in data:
                    key = f"features.{key}"
                else: 
                     # Should we warn?
                     # print(f"Missing {key}")
                     return None
            val = data[key]
            if transpose and val.ndim == 4:
                val = val.transpose(0, 2, 3, 1)
            return mx.array(val)
        
        def load_bn(mod, prefix):
            w = load_w(f"{prefix}.weight")
            b = load_w(f"{prefix}.bias")
            m = load_w(f"{prefix}.running_mean")
            v = load_w(f"{prefix}.running_var")
            if w is not None: mod.weight = w
            if b is not None: mod.bias = b
            if m is not None: mod.running_mean = m
            if v is not None: mod.running_var = v

        def load_conv(mod, prefix):
            w = load_w(f"{prefix}.weight", transpose=True)
            if w is not None: mod.weight = w
        
        # 1. Initial Features
        # PT: features.conv0, features.norm0
        load_conv(self.features_start.layers[0], "conv0")
        load_bn(self.features_start.layers[1], "norm0")
        
        # 2. Dense Blocks
        # Structure: denseblock{i}.denselayer{j}
        # denselayer{j}: norm1, conv1, norm2, conv2
        
        blocks = [self.denseblock1, self.denseblock2, self.denseblock3, self.denseblock4]
        for b_idx, block in enumerate(blocks):
            b_name = f"denseblock{b_idx+1}"
            
            for l_idx, layer in enumerate(block.layers):
                l_name = f"denselayer{l_idx+1}"
                prefix = f"{b_name}.{l_name}"
                
                load_bn(layer.norm1, f"{prefix}.norm1")
                load_conv(layer.conv1, f"{prefix}.conv1")
                load_bn(layer.norm2, f"{prefix}.norm2")
                load_conv(layer.conv2, f"{prefix}.conv2")
            
            # Transition
            if b_idx < len(blocks) - 1:
                t_name = f"transition{b_idx+1}"
                trans = getattr(self, t_name)
                load_bn(trans.norm, f"{t_name}.norm")
                load_conv(trans.conv, f"{t_name}.conv")
                
        # 3. Final Norm
        load_bn(self.norm5, "norm5")
        
        # 4. Classifier
        # PT: classifier.weight (Out, In), bias
        cw = load_w("classifier.weight")
        cb = load_w("classifier.bias")
        if cw is not None: self.classifier.weight = cw.T
        if cb is not None: self.classifier.bias = cb
