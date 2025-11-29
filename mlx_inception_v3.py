import mlx.core as mx
import mlx.nn as nn
import numpy as np

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        # PyTorch Inception uses standard Conv2d with BN and ReLU
        # kwargs might handle kernel_size, stride, padding
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm(out_channels, eps=0.001)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nn.ReLU()(x)

class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def __call__(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)
        branch_pool = self.branch_pool(branch_pool)

        return mx.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1)

class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def __call__(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)(x)

        return mx.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1)

class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def __call__(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)
        branch_pool = self.branch_pool(branch_pool)

        return mx.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1)

class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def __call__(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)(x)

        return mx.concatenate([branch3x3, branch7x7x3, branch_pool], axis=-1)

class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def __call__(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = mx.concatenate(branch3x3, axis=-1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = mx.concatenate(branch3x3dbl, axis=-1)

        branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)
        branch_pool = self.branch_pool(branch_pool)

        return mx.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1)

class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

    def forward_with_endpoints(self, x):
        endpoints = {}
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        endpoints['Conv2d_2b_3x3'] = x
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        endpoints['Conv2d_3b_1x1'] = x
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        endpoints['Conv2d_4a_3x3'] = x
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        
        x = self.Mixed_5b(x)
        endpoints['Mixed_5b'] = x
        x = self.Mixed_5c(x)
        endpoints['Mixed_5c'] = x
        x = self.Mixed_5d(x)
        endpoints['Mixed_5d'] = x
        
        x = self.Mixed_6a(x)
        endpoints['Mixed_6a'] = x
        x = self.Mixed_6b(x)
        endpoints['Mixed_6b'] = x
        x = self.Mixed_6c(x)
        endpoints['Mixed_6c'] = x
        x = self.Mixed_6d(x)
        endpoints['Mixed_6d'] = x
        x = self.Mixed_6e(x)
        endpoints['Mixed_6e'] = x
        
        x = self.Mixed_7a(x)
        endpoints['Mixed_7a'] = x
        x = self.Mixed_7b(x)
        endpoints['Mixed_7b'] = x
        x = self.Mixed_7c(x)
        endpoints['Mixed_7c'] = x
        
        return x, endpoints

    def load_npz(self, path):
        data = np.load(path)
        
        def to_mlx(w):
            # PyTorch: (out, in, h, w)
            # MLX: (out, h, w, in)
            return np.transpose(w, (0, 2, 3, 1)) if w.ndim == 4 else w

        def load_conv(mod, prefix):
            mod.conv.weight = mx.array(to_mlx(data[f"{prefix}.conv.weight"]))
            # BN
            mod.bn.weight = mx.array(data[f"{prefix}.bn.weight"])
            mod.bn.bias = mx.array(data[f"{prefix}.bn.bias"])
            mod.bn.running_mean = mx.array(data[f"{prefix}.bn.running_mean"])
            mod.bn.running_var = mx.array(data[f"{prefix}.bn.running_var"])

        # Initial Layers
        load_conv(self.Conv2d_1a_3x3, "Conv2d_1a_3x3")
        load_conv(self.Conv2d_2a_3x3, "Conv2d_2a_3x3")
        load_conv(self.Conv2d_2b_3x3, "Conv2d_2b_3x3")
        load_conv(self.Conv2d_3b_1x1, "Conv2d_3b_1x1")
        load_conv(self.Conv2d_4a_3x3, "Conv2d_4a_3x3")

        # Mixed Blocks
        # We iterate module names and recursively load
        # Note: The structure of InceptionA/B/C etc matches the keys in torchvision
        
        def load_block(block, prefix):
            # block is e.g. self.Mixed_5b
            # prefix is "Mixed_5b"
            # Keys inside: Mixed_5b.branch1x1.conv.weight etc.
            
            # Inspect block children
            for name, child in block.__dict__.items():
                # This relies on internal structure, but cleaner to use vars(block)
                pass
            
            # Manually mapping based on class defs is safer
            if isinstance(block, InceptionA):
                load_conv(block.branch1x1, f"{prefix}.branch1x1")
                load_conv(block.branch5x5_1, f"{prefix}.branch5x5_1")
                load_conv(block.branch5x5_2, f"{prefix}.branch5x5_2")
                load_conv(block.branch3x3dbl_1, f"{prefix}.branch3x3dbl_1")
                load_conv(block.branch3x3dbl_2, f"{prefix}.branch3x3dbl_2")
                load_conv(block.branch3x3dbl_3, f"{prefix}.branch3x3dbl_3")
                load_conv(block.branch_pool, f"{prefix}.branch_pool")
            
            elif isinstance(block, InceptionB):
                load_conv(block.branch3x3, f"{prefix}.branch3x3")
                load_conv(block.branch3x3dbl_1, f"{prefix}.branch3x3dbl_1")
                load_conv(block.branch3x3dbl_2, f"{prefix}.branch3x3dbl_2")
                load_conv(block.branch3x3dbl_3, f"{prefix}.branch3x3dbl_3")
            
            elif isinstance(block, InceptionC):
                load_conv(block.branch1x1, f"{prefix}.branch1x1")
                load_conv(block.branch7x7_1, f"{prefix}.branch7x7_1")
                load_conv(block.branch7x7_2, f"{prefix}.branch7x7_2")
                load_conv(block.branch7x7_3, f"{prefix}.branch7x7_3")
                load_conv(block.branch7x7dbl_1, f"{prefix}.branch7x7dbl_1")
                load_conv(block.branch7x7dbl_2, f"{prefix}.branch7x7dbl_2")
                load_conv(block.branch7x7dbl_3, f"{prefix}.branch7x7dbl_3")
                load_conv(block.branch7x7dbl_4, f"{prefix}.branch7x7dbl_4")
                load_conv(block.branch7x7dbl_5, f"{prefix}.branch7x7dbl_5")
                load_conv(block.branch_pool, f"{prefix}.branch_pool")

            elif isinstance(block, InceptionD):
                load_conv(block.branch3x3_1, f"{prefix}.branch3x3_1")
                load_conv(block.branch3x3_2, f"{prefix}.branch3x3_2")
                load_conv(block.branch7x7x3_1, f"{prefix}.branch7x7x3_1")
                load_conv(block.branch7x7x3_2, f"{prefix}.branch7x7x3_2")
                load_conv(block.branch7x7x3_3, f"{prefix}.branch7x7x3_3")
                load_conv(block.branch7x7x3_4, f"{prefix}.branch7x7x3_4")

            elif isinstance(block, InceptionE):
                load_conv(block.branch1x1, f"{prefix}.branch1x1")
                load_conv(block.branch3x3_1, f"{prefix}.branch3x3_1")
                load_conv(block.branch3x3_2a, f"{prefix}.branch3x3_2a")
                load_conv(block.branch3x3_2b, f"{prefix}.branch3x3_2b")
                load_conv(block.branch3x3dbl_1, f"{prefix}.branch3x3dbl_1")
                load_conv(block.branch3x3dbl_2, f"{prefix}.branch3x3dbl_2")
                load_conv(block.branch3x3dbl_3a, f"{prefix}.branch3x3dbl_3a")
                load_conv(block.branch3x3dbl_3b, f"{prefix}.branch3x3dbl_3b")
                load_conv(block.branch_pool, f"{prefix}.branch_pool")

        load_block(self.Mixed_5b, "Mixed_5b")
        load_block(self.Mixed_5c, "Mixed_5c")
        load_block(self.Mixed_5d, "Mixed_5d")
        
        load_block(self.Mixed_6a, "Mixed_6a")
        load_block(self.Mixed_6b, "Mixed_6b")
        load_block(self.Mixed_6c, "Mixed_6c")
        load_block(self.Mixed_6d, "Mixed_6d")
        load_block(self.Mixed_6e, "Mixed_6e")
        
        load_block(self.Mixed_7a, "Mixed_7a")
        load_block(self.Mixed_7b, "Mixed_7b")
        load_block(self.Mixed_7c, "Mixed_7c")

    def __call__(self, x):
        x, _ = self.forward_with_endpoints(x)
        return x


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(768, num_classes)

    def __call__(self, x):
        x = self.avgpool(x)
        x = self.conv0(x)
        x = self.conv1(x)
        x = mx.mean(x, axis=(1, 2))
        x = self.dropout(x)
        return self.fc(x)


class InceptionV3Train(InceptionV3):
    def __init__(self, num_classes, aux_weight=0.3, use_aux=True):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)
        self.use_aux = use_aux
        self.aux_weight = aux_weight
        self.aux_classifier = InceptionAux(768, num_classes)

    def forward_logits(self, x, train=False):
        x, endpoints = self.forward_with_endpoints(x)
        pooled = mx.mean(x, axis=(1, 2))
        logits = self.fc(self.dropout(pooled))

        aux1 = aux2 = None
        if train and self.use_aux and "Mixed_6e" in endpoints:
            aux1 = self.aux_classifier(endpoints["Mixed_6e"])
        return logits, aux1, aux2

    def load_npz(self, path):
        super().load_npz(path)
        data = np.load(path)

        if "fc.weight" in data:
            self.fc.weight = mx.array(data["fc.weight"])
        if "fc.bias" in data:
            self.fc.bias = mx.array(data["fc.bias"])

        if not self.use_aux or "AuxLogits.fc.weight" not in data:
            return

        def to_mlx(w):
            return np.transpose(w, (0, 2, 3, 1)) if w.ndim == 4 else w

        # Aux conv0
        if "AuxLogits.conv0.conv.weight" in data:
            self.aux_classifier.conv0.conv.weight = mx.array(
                to_mlx(data["AuxLogits.conv0.conv.weight"])
            )
            bn = self.aux_classifier.conv0.bn
            bn.weight = mx.array(data["AuxLogits.conv0.bn.weight"])
            bn.bias = mx.array(data["AuxLogits.conv0.bn.bias"])
            bn.running_mean = mx.array(data["AuxLogits.conv0.bn.running_mean"])
            bn.running_var = mx.array(data["AuxLogits.conv0.bn.running_var"])

        # Aux conv1
        if "AuxLogits.conv1.conv.weight" in data:
            self.aux_classifier.conv1.conv.weight = mx.array(
                to_mlx(data["AuxLogits.conv1.conv.weight"])
            )
            bn = self.aux_classifier.conv1.bn
            bn.weight = mx.array(data["AuxLogits.conv1.bn.weight"])
            bn.bias = mx.array(data["AuxLogits.conv1.bn.bias"])
            bn.running_mean = mx.array(data["AuxLogits.conv1.bn.running_mean"])
            bn.running_var = mx.array(data["AuxLogits.conv1.bn.running_var"])

        self.aux_classifier.fc.weight = mx.array(data["AuxLogits.fc.weight"])
        self.aux_classifier.fc.bias = mx.array(data["AuxLogits.fc.bias"])
