"""
AlexNet in MLX with endpoints for relu1, relu2, relu3, relu4, relu5.
Loads weights from a torchvision-exported npz.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _conv(in_ch, out_ch, kernel_size, stride=1, padding=0):
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    )


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            _conv(3, 64, kernel_size=11, stride=4, padding=2),  # 0
            nn.ReLU(),  # 1 (relu1)
            nn.MaxPool2d(kernel_size=3, stride=2),  # 2
            _conv(64, 192, kernel_size=5, padding=2),  # 3
            nn.ReLU(),  # 4 (relu2)
            nn.MaxPool2d(kernel_size=3, stride=2),  # 5
            _conv(192, 384, kernel_size=3, padding=1),  # 6
            nn.ReLU(),  # 7 (relu3)
            _conv(384, 256, kernel_size=3, padding=1),  # 8
            nn.ReLU(),  # 9 (relu4)
            _conv(256, 256, kernel_size=3, padding=1),  # 10
            nn.ReLU(),  # 11 (relu5)
            nn.MaxPool2d(kernel_size=3, stride=2),  # 12
        ]

        self.endpoint_indices = {
            "relu1": 1,
            "relu2": 4,
            "relu3": 7,
            "relu4": 9,
            "relu5": 11,
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

        # Map layer indices to 'features.X' in standard torchvision keys
        conv_indices = [0, 3, 6, 8, 10]
        
        for idx in conv_indices:
            conv = self.layers[idx]
            weight_key = f"features.{idx}.weight"
            bias_key = f"features.{idx}.bias"
            
            conv.weight = load_weight(weight_key, transpose=True)
            conv.bias = load_weight(bias_key)
