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

        def to_mlx_weight(w):
            # PyTorch Conv2d weights are (out_channels, in_channels, kH, kW)
            # MLX expects channel-last filters: (out_channels, kH, kW, in_channels)
            return np.transpose(w, (0, 2, 3, 1)) if w.ndim == 4 else w

        def load_conv_bn(prefix, seq_mod: nn.Sequential):
            conv = seq_mod.layers[0]
            bn = seq_mod.layers[1]
            conv.weight = mx.array(to_mlx_weight(data[f"{prefix}.conv.weight"]))
            bn.weight = mx.array(data[f"{prefix}.bn.weight"])
            bn.bias = mx.array(data[f"{prefix}.bn.bias"])
            bn.running_mean = mx.array(data[f"{prefix}.bn.running_mean"])
            bn.running_var = mx.array(data[f"{prefix}.bn.running_var"])

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
