import numpy as np
import sys

def inspect_npz(path):
    try:
        data = np.load(path)
        print(f"Keys in {path}:")
        for key in data.files:
            print(f"  {key}: {data[key].shape}")
    except Exception as e:
        print(f"Error reading {path}: {e}")

if __name__ == "__main__":
    inspect_npz("models/vgg16_mlx.npz")
    inspect_npz("models/vgg19_mlx.npz")
