"""
Export torchvision ResNet50 weights to an .npz for MLX.
Run this in a PyTorch+torchvision env:
    python export_resnet50_npz.py
It writes models/resnet50_mlx.npz
"""
import os
import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights


def main():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    state = model.state_dict()
    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", "resnet50_mlx.npz")
    np.savez(out_path, **{k: v.cpu().numpy() for k, v in state.items()})
    print(f"Saved {out_path} with {len(state)} tensors.")


if __name__ == "__main__":
    main()
