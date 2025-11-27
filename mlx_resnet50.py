"""
ResNet50 in MLX for DeepDream.
Loads weights from a torchvision-exported npz (see export_resnet50_npz.py).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(planes, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes, eps=1e-5, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(planes * self.expansion, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(self.inplanes, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(planes * block.expansion, eps=1e-5, momentum=0.1),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_with_endpoints(self, x):
        endpoints = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        endpoints['conv1'] = x
        
        x = self.maxpool(x)
        
        # Layer 1
        for i, layer in enumerate(self.layer1.layers):
            x = layer(x)
            endpoints[f'layer1_{i}'] = x
        endpoints['layer1'] = x

        # Layer 2
        for i, layer in enumerate(self.layer2.layers):
            x = layer(x)
            endpoints[f'layer2_{i}'] = x
        endpoints['layer2'] = x

        # Layer 3
        for i, layer in enumerate(self.layer3.layers):
            x = layer(x)
            endpoints[f'layer3_{i}'] = x
        endpoints['layer3'] = x

        # Layer 4
        for i, layer in enumerate(self.layer4.layers):
            x = layer(x)
            endpoints[f'layer4_{i}'] = x
        endpoints['layer4'] = x

        return x, endpoints

    def load_npz(self, path: str):
        data = np.load(path)

        def to_mlx_weight(w):
            return np.transpose(w, (0, 2, 3, 1)) if w.ndim == 4 else w

        def load_bn(prefix, bn):
            bn.weight = mx.array(data[f"{prefix}.weight"])
            bn.bias = mx.array(data[f"{prefix}.bias"])
            bn.running_mean = mx.array(data[f"{prefix}.running_mean"])
            bn.running_var = mx.array(data[f"{prefix}.running_var"])

        def load_conv(prefix, conv):
            conv.weight = mx.array(to_mlx_weight(data[f"{prefix}.weight"]))
        
        # Initial layers
        load_conv("conv1", self.conv1)
        load_bn("bn1", self.bn1)

        def load_layer(prefix, layer_mod):
            for i, block in enumerate(layer_mod.layers):
                block_prefix = f"{prefix}.{i}"
                load_conv(f"{block_prefix}.conv1", block.conv1)
                load_bn(f"{block_prefix}.bn1", block.bn1)
                load_conv(f"{block_prefix}.conv2", block.conv2)
                load_bn(f"{block_prefix}.bn2", block.bn2)
                load_conv(f"{block_prefix}.conv3", block.conv3)
                load_bn(f"{block_prefix}.bn3", block.bn3)
                
                if block.downsample is not None:
                    load_conv(f"{block_prefix}.downsample.0", block.downsample.layers[0])
                    load_bn(f"{block_prefix}.downsample.1", block.downsample.layers[1])

        load_layer("layer1", self.layer1)
        load_layer("layer2", self.layer2)
        load_layer("layer3", self.layer3)
        load_layer("layer4", self.layer4)

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])
