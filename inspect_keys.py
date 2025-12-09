
import numpy as np
import sys

def inspect(path):
    try:
        data = np.load(path)
        print(f"Keys in {path}:")
        for k in sorted(data.files):
            print(f"  {k}: {data[k].shape}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_keys.py <path_to_npz>")
        sys.exit(1)
    inspect(sys.argv[1])
