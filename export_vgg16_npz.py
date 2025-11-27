"""
Export torchvision VGG16 weights to an .npz for MLX.
Run this in a PyTorch+torchvision env:
    python export_vgg16_npz.py
It writes models/vgg16_mlx.npz
"""
import os
import numpy as np
import torch
from torchvision.models import vgg16, VGG16_Weights


def main():
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    state = model.state_dict()
    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", "vgg16_mlx.npz")
    np.savez(out_path, **{k: v.cpu().numpy() for k, v in state.items()})
    print(f"Saved {out_path} with {len(state)} tensors.")


if __name__ == "__main__":
    main()
