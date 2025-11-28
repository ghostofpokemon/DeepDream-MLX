"""
VGG19 in MLX with endpoints for common DeepDream layers.
Loads weights from a torchvision-exported npz (see export_vgg19_npz.py).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _conv(in_ch, out_ch, kernel_size=3, padding=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True)


class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        # Mirrors torchvision.models.vgg19(features) layout
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
            _conv(256, 256),  # 16 conv3_4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            _conv(256, 512),  # 19 conv4_1
            nn.ReLU(),
            _conv(512, 512),  # 21 conv4_2
            nn.ReLU(),
            _conv(512, 512),  # 23 conv4_3
            nn.ReLU(),
            _conv(512, 512),  # 25 conv4_4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            _conv(512, 512),  # 28 conv5_1
            nn.ReLU(),
            _conv(512, 512),  # 30 conv5_2
            nn.ReLU(),
            _conv(512, 512),  # 32 conv5_3
            nn.ReLU(),
            _conv(512, 512),  # 34 conv5_4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        self.endpoint_indices = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_2": 13,
            "relu3_3": 15,
            "relu3_4": 17,
            "relu4_1": 20,
            "relu4_2": 22,
            "relu4_3": 24,
            "relu4_4": 26,
            "relu5_1": 29,
            "relu5_2": 31,
            "relu5_3": 33,
            "relu5_4": 35,
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

        def load_weight(key, transpose=False):
            if key in data:
                w = data[key]
            elif f"{key}_int8" in data:
                w_int8 = data[f"{key}_int8"]
                scale = data[f"{key}_scale"]
                w = w_int8.astype(scale.dtype) * scale
            else:
                raise ValueError(f"Missing key {key} in npz")
            
            if transpose and w.ndim == 4:
                w = np.transpose(w, (0, 2, 3, 1))
            return mx.array(w)

        conv_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        for idx in conv_indices:
            conv = self.layers[idx]
            weight_key = f"features.{idx}.weight"
            bias_key = f"features.{idx}.bias"
            
            conv.weight = load_weight(weight_key, transpose=True)
            conv.bias = load_weight(bias_key)
