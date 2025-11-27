"""
VGG16 in MLX with endpoints for relu1_2, relu2_2, relu3_3, relu4_2, relu4_3,
relu5_2, relu5_3. Loads weights from a torchvision-exported npz
(see export_vgg16_npz.py).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _conv(in_ch, out_ch, kernel_size=3, padding=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True)


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            _conv(3, 64),  # 0 conv1_1
            nn.ReLU(),
            _conv(64, 64),  # 2 conv1_2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _conv(64, 128),  # 5 conv2_1
            nn.ReLU(),
            _conv(128, 128),  # 7 conv2_2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _conv(128, 256),  # 10 conv3_1
            nn.ReLU(),
            _conv(256, 256),  # 12 conv3_2
            nn.ReLU(),
            _conv(256, 256),  # 14 conv3_3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _conv(256, 512),  # 17 conv4_1
            nn.ReLU(),
            _conv(512, 512),  # 19 conv4_2
            nn.ReLU(),
            _conv(512, 512),  # 21 conv4_3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _conv(512, 512),  # 24 conv5_1
            nn.ReLU(),
            _conv(512, 512),  # 26 conv5_2
            nn.ReLU(),
            _conv(512, 512),  # 28 conv5_3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        # Layer indices in self.layers corresponding to named endpoints
        self.endpoint_indices = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_3": 15,
            "relu4_1": 18,
            "relu4_2": 20,
            "relu4_3": 22,
            "relu5_1": 25,
            "relu5_2": 27,
            "relu5_3": 29,
        }

    def forward_with_endpoints(self, x):
        endpoints = {}
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            for name, i in self.endpoint_indices.items():
                if idx == i:
                    endpoints[name] = x
        return x, endpoints

    def __call__(self, x):
        _, endpoints = self.forward_with_endpoints(x)
        return endpoints

    def load_npz(self, path: str):
        data = np.load(path)

        def to_mlx_weight(w):
            return np.transpose(w, (0, 2, 3, 1)) if w.ndim == 4 else w

        conv_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        for idx in conv_indices:
            conv = self.layers[idx]
            weight_key = f"features.{idx}.weight"
            bias_key = f"features.{idx}.bias"
            conv.weight = mx.array(to_mlx_weight(data[weight_key]))
            conv.bias = mx.array(data[bias_key])
