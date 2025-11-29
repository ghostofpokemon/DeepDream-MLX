"""
Minimal GoogLeNet (Inception V1) in MLX, up to inception4e.
Loads weights from a torchvision-exported npz (see export_googlenet_npz.py).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _conv_bn(in_ch, out_ch, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm(out_ch, eps=1e-3, momentum=0.1),
        nn.ReLU(),
    )


class Inception(nn.Module):
    def __init__(self, in_ch, ch1, ch3r, ch3, ch5r, ch5, pool_proj):
        super().__init__()
        self.branch1 = _conv_bn(in_ch, ch1, 1)

        self.branch2_1 = _conv_bn(in_ch, ch3r, 1)
        self.branch2_2 = _conv_bn(ch3r, ch3, 3, padding=1)

        self.branch3_1 = _conv_bn(in_ch, ch5r, 1)
        # The reference torchvision GoogLeNet uses a 3x3 conv here (not 5x5)
        self.branch3_2 = _conv_bn(ch5r, ch5, 3, padding=1)

        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_2 = _conv_bn(in_ch, pool_proj, 1)

    def __call__(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2_2(self.branch2_1(x))
        b3 = self.branch3_2(self.branch3_1(x))
        b4 = self.branch4_2(self.branch4_pool(x))
        return mx.concatenate([b1, b2, b3, b4], axis=-1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _conv_bn(3, 64, 7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv2 = _conv_bn(64, 64, 1)
        self.conv3 = _conv_bn(64, 192, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

    def forward_with_endpoints(self, x):
        endpoints = {}
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        endpoints["inception3a"] = x
        x = self.inception3b(x)
        endpoints["inception3b"] = x
        x = self.maxpool3(x)

        x = self.inception4a(x)
        endpoints["inception4a"] = x
        x = self.inception4b(x)
        endpoints["inception4b"] = x
        x = self.inception4c(x)
        endpoints["inception4c"] = x
        x = self.inception4d(x)
        endpoints["inception4d"] = x
        x = self.inception4e(x)
        endpoints["inception4e"] = x
        x = self.maxpool4(x)

        x = self.inception5a(x)
        endpoints["inception5a"] = x
        x = self.inception5b(x)
        endpoints["inception5b"] = x
        return x, endpoints

    def __call__(self, x):
        _, endpoints = self.forward_with_endpoints(x)
        return endpoints

    def load_npz(self, path: str):
        data = np.load(path)

        def load_weight(key, target_module, param_name="weight", transpose=False):
            # Check for standard float16/32 key
            if key in data:
                w = data[key]
            # Check for int8 quantized key
            elif f"{key}_int8" in data:
                w_int8 = data[f"{key}_int8"]
                scale = data[f"{key}_scale"]
                # Dequantize
                w = w_int8.astype(scale.dtype) * scale
            else:
                raise ValueError(f"Missing key {key} (or {key}_int8) in npz")

            # Transpose for Conv2d weights if needed (PyTorch [O,I,H,W] -> MLX [O,H,W,I])
            if transpose and w.ndim == 4:
                w = np.transpose(w, (0, 2, 3, 1))
            
            # Assign to module
            target_module[param_name] = mx.array(w)

        def load_conv_bn(prefix, seq_mod: nn.Sequential):
            conv = seq_mod.layers[0]
            bn = seq_mod.layers[1]
            
            load_weight(f"{prefix}.conv.weight", conv, transpose=True)
            
            load_weight(f"{prefix}.bn.weight", bn)
            load_weight(f"{prefix}.bn.bias", bn, param_name="bias")
            load_weight(f"{prefix}.bn.running_mean", bn, param_name="running_mean")
            load_weight(f"{prefix}.bn.running_var", bn, param_name="running_var")

        load_conv_bn("conv1", self.conv1)
        load_conv_bn("conv2", self.conv2)
        load_conv_bn("conv3", self.conv3)

        def load_inception(prefix, module: Inception):
            load_conv_bn(f"{prefix}.branch1", module.branch1)
            load_conv_bn(f"{prefix}.branch2.0", module.branch2_1)
            load_conv_bn(f"{prefix}.branch2.1", module.branch2_2)
            load_conv_bn(f"{prefix}.branch3.0", module.branch3_1)
            load_conv_bn(f"{prefix}.branch3.1", module.branch3_2)
            load_conv_bn(f"{prefix}.branch4.1", module.branch4_2)

        load_inception("inception3a", self.inception3a)
        load_inception("inception3b", self.inception3b)
        load_inception("inception4a", self.inception4a)
        load_inception("inception4b", self.inception4b)
        load_inception("inception4c", self.inception4c)
        load_inception("inception4d", self.inception4d)
        load_inception("inception4e", self.inception4e)
        load_inception("inception5a", self.inception5a)
        load_inception("inception5b", self.inception5b)


class AuxHead(nn.Module):
    def __init__(self, in_ch, num_classes, dropout=0.7):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.proj = _conv_bn(in_ch, 128, 1)
        self.fc1 = nn.Linear(128, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, num_classes)

    def __call__(self, x):
        x = self.avgpool(x)
        x = self.proj(x)
        x = mx.mean(x, axis=(1, 2))  # global average to stabilize feature size
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class GoogLeNetTrain(GoogLeNet):
    def __init__(self, num_classes, aux_weight=0.3, use_aux=True):
        super().__init__()
        self.global_dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        self.use_aux = use_aux
        self.aux_weight = aux_weight
        self.aux1 = AuxHead(512, num_classes)
        self.aux2 = AuxHead(528, num_classes)

    def forward_logits(self, x, train=False):
        x, endpoints = self.forward_with_endpoints(x)
        pooled = mx.mean(x, axis=(1, 2))
        logits = self.fc(self.global_dropout(pooled))

        aux1_logits = aux2_logits = None
        if train and self.use_aux:
            aux1_logits = self.aux1(endpoints["inception4a"])
            aux2_logits = self.aux2(endpoints["inception4d"])
        return logits, aux1_logits, aux2_logits
