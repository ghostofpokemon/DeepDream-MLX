"""
Unified export script for converting PyTorch models to MLX .npz format.
Supports VGG16, VGG19, GoogLeNet, and ResNet50.
Handles both float32 (default) and float16/bfloat16 (efficient) exports.

Usage:
    python export_models.py --model all --dtype float16
    python export_models.py --model vgg16
"""

import argparse
import os
import numpy as np
import torch
import torchvision.models as models

def get_model_info(model_name):
    if model_name == "vgg16":
        return models.vgg16, models.VGG16_Weights.IMAGENET1K_V1
    elif model_name == "vgg19":
        return models.vgg19, models.VGG19_Weights.IMAGENET1K_V1
    elif model_name == "googlenet":
        return models.googlenet, models.GoogLeNet_Weights.IMAGENET1K_V1
    elif model_name == "resnet50":
        return models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f"Unknown model: {model_name}")

def export_model(model_name, dtype="float32"):
    print(f"Exporting {model_name} ({dtype})...")
    model_fn, weights = get_model_info(model_name)
    model = model_fn(weights=weights)
    model.eval()
    
    state = model.state_dict()
    converted_state = {}
    
    target_type = np.float32
    suffix = ""
    
    if dtype in ["float16", "bf16", "half"]:
        target_type = np.float16
        suffix = "_bf16" # Keep legacy suffix for compatibility with dream.py logic
    
    for k, v in state.items():
        v_np = v.cpu().detach().numpy()
        
        # ResNet specific: Transpose conv weights if needed?
        # MLX ResNet implementation usually handles transposition in load_weights or expects PyTorch layout?
        # Our mlx_resnet50.py has `to_mlx_weight` which does transpose (0, 2, 3, 1).
        # VGG MLX implementation also does transpose.
        # However, our previous export scripts (e.g. export_vgg16_npz.py) just saved state_dict AS IS.
        # And `mlx_*.py` `load_npz` handled the transposition.
        # So we just save the numpy array as is.
        
        converted_state[k] = v_np.astype(target_type)

    out_name = f"{model_name}_mlx{suffix}.npz"
    np.savez(out_name, **converted_state)
    
    original_size = sum(v.numel() * 4 for v in state.values()) / (1024*1024)
    new_size = os.path.getsize(out_name) / (1024*1024)
    
    print(f"✅ Saved {out_name}")
    print(f"   Size: {new_size:.1f} MB (Original: ~{original_size:.1f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch models to MLX")
    parser.add_argument("--model", choices=["vgg16", "vgg19", "googlenet", "resnet50", "all"], default="all")
    parser.add_argument("--dtype", choices=["float32", "float16", "bf16"], default="float16", help="Output data type")
    args = parser.parse_args()
    
    models_to_export = ["vgg16", "vgg19", "googlenet", "resnet50"] if args.model == "all" else [args.model]
    
    for m in models_to_export:
        export_model(m, args.dtype)

if __name__ == "__main__":
    main()
