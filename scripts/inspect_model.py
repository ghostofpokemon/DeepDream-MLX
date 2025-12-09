#!/usr/bin/env python3
"""
Inspect MLX Model Architecture & Weights.
Visualizes layer shapes and statistics with 24-bit color ASCII art.
Style: Futurist-Beaux-Arts / Futurist-Neoclassical
"""

import sys
import os
import argparse
import numpy as np
import mlx.core as mx
import mlx.utils

# Ensure we can import deepdream
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from deepdream import MODEL_REGISTRY
    from deepdream.visualization import print_header, print_footer, traverse_model
except ImportError as e:
    print(f"Error: Could not import deepdream package: {e}")
    sys.exit(1)


def visualize_model(model_name, weights_path=None):
    if model_name not in MODEL_REGISTRY:
        print(f"Unknown model: {model_name}")
        return

    info = MODEL_REGISTRY[model_name]
    print_header(f"{model_name.upper()}")
    
    # Instantiate
    try:
        model = info["cls"]()
    except Exception as e:
        print(f"Failed to instantiate {model_name}: {e}")
        return
    
    # Load Weights if available
    if weights_path:
        print(f"Loading weights from: {weights_path}")
        try:
            model.load_npz(weights_path)
            print(f"\033[92mWeights loaded.\033[0m")
        except Exception as e:
            print(f"\033[91mWeights load warning: {e}\033[0m")
            
    # Calculate GLOBAL Total Parameters (Accurate)
    try:
        # Flatten parameters, but ensure uniqueness by object ID to avoid double counting
        # (e.g. if blocks list and features Sequential both ref the same modules)
        flat_params = mlx.utils.tree_flatten(model.parameters())
        unique_params = {id(v): v for _, v in flat_params}
        total_params = sum(v.size for v in unique_params.values())
    except Exception as e:
        print(f"Error counting params: {e}")
        total_params = 0

    # Traverse Visual
    print("-" * 72)
    traverse_model(model, model_name)
    print_footer(total_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect MLX Model")
    parser.add_argument("--model", type=str, default="googlenet")
    parser.add_argument("--weights", type=str, help="Path to .npz weights")
    args = parser.parse_args()
    
    visualize_model(args.model, args.weights)
