#!/usr/bin/env python3
"""
Universal Model Converter for DeepDream-MLX.
Converts PyTorch (.pth) and Torch7 (.t7) models to MLX (.npz).
Also supports auto-downloading standard Places365 models.
Defaults to float16 for optimal performance on Apple Silicon.
"""

import os
import argparse
import glob
import numpy as np
import torch
import torchvision.models as models
from torch.hub import download_url_to_file

# Optional Torchfile for .t7 support
try:
    import torchfile
except ImportError:
    torchfile = None

# --- Configuration ---
PLACES365_URLS = {
    "alexnet": "http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar",
    "resnet50": "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar",
    "vgg16": "http://places2.csail.mit.edu/models_places365/vgg16_places365.pth.tar",
    "googlenet": "http://places2.csail.mit.edu/models_places365/googlenet_places365.pth.tar"
}

# --- Helper Functions ---

def convert_tensor(tensor, target_dtype=np.float16):
    """Converts a tensor/array to the target numpy dtype."""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy().astype(target_dtype)
    elif isinstance(tensor, np.ndarray):
        return tensor.astype(target_dtype)
    else:
        return np.array(tensor).astype(target_dtype)

def clean_state_dict(state_dict):
    """
    Flattens the state dictionary and removes common prefix artifacts 
    like 'module.' from DataParallel wrapping.
    """
    new_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' anywhere in the key
        name = k.replace("module.", "")
        new_dict[name] = convert_tensor(v)
    return new_dict

def get_places365_model_skeleton(arch):
    """Returns a standard PyTorch model structure for Places365."""
    if arch == "alexnet":
        return models.alexnet(num_classes=365)
    elif arch == "resnet50":
        return models.resnet50(num_classes=365)
    elif arch == "vgg16":
        return models.vgg16(num_classes=365)
    elif arch == "googlenet":
        return models.googlenet(num_classes=365, aux_logits=False)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

# --- Conversion Logic ---

def convert_torch7(filepath, target_dir):
    if torchfile is None:
        print(f"⚠️  Skipping {filepath}: 'torchfile' not installed. Run `pip install torchfile`.")
        return

    print(f"Processing Torch7 file: {filepath}")
    try:
        model_obj = torchfile.load(filepath)
        converted_state = {}

        def extract_layers(layer, prefix=""):
            if hasattr(layer, 'weight') and layer.weight is not None:
                converted_state[f"{prefix}.weight"] = convert_tensor(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                converted_state[f"{prefix}.bias"] = convert_tensor(layer.bias)
            
            if hasattr(layer, 'modules') and layer.modules:
                for i, sublayer in enumerate(layer.modules):
                    # 0-based indexing for compatibility
                    next_prefix = f"{prefix}.{i}" if prefix else f"{i}"
                    extract_layers(sublayer, next_prefix)
        
        extract_layers(model_obj)
        
        if not converted_state:
            print(f"❌ No weights found in {filepath}.")
            return

        name_base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(target_dir, f"{name_base}_t7_mlx.npz")
        np.savez(out_path, **converted_state)
        print(f"✅ Saved {out_path} ({len(converted_state)} tensors)")

    except Exception as e:
        print(f"❌ Failed to convert {filepath}: {e}")

def convert_pytorch(filepath, target_dir):
    print(f"Processing PyTorch file: {filepath}")
    try:
        checkpoint = torch.load(filepath, map_location="cpu")
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            print(f"❌ Unknown checkpoint format in {filepath}")
            return

        clean_dict = clean_state_dict(state_dict)
        
        name_base = os.path.splitext(os.path.basename(filepath))[0]
        # Avoid double extension if file was .pth.tar
        if name_base.endswith(".pth"):
            name_base = os.path.splitext(name_base)[0]
            
        out_path = os.path.join(target_dir, f"{name_base}_mlx.npz")
        np.savez(out_path, **clean_dict)
        
        size_mb = os.path.getsize(out_path) / (1024*1024)
        print(f"✅ Saved {out_path} ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"❌ Failed to convert {filepath}: {e}")

def download_and_convert_places365(arch, download_dir, target_dir):
    url = PLACES365_URLS.get(arch)
    if not url:
        print(f"No URL for {arch}")
        return

    filename = os.path.join(download_dir, os.path.basename(url))
    
    # 1. Download
    if not os.path.exists(filename):
        print(f"Downloading {arch} from {url}...")
        try:
            download_url_to_file(url, filename)
        except Exception as e:
            print(f"Download failed: {e}")
            return
    else:
        print(f"Found cached {filename}")

    # 2. Load into standard Skeleton (ensures structural correctness)
    print(f"Loading {arch} into PyTorch structure...")
    try:
        model = get_places365_model_skeleton(arch)
        checkpoint = torch.load(filename, map_location="cpu")
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # Robust Load
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except:
            model.load_state_dict(new_state_dict, strict=False)
            
        # 3. Export
        model.eval()
        final_dict = clean_state_dict(model.state_dict())
        out_path = os.path.join(target_dir, f"{arch}_places365_mlx.npz")
        np.savez(out_path, **final_dict)
        print(f"✅ Saved {out_path}")

    except Exception as e:
        print(f"Failed to process {arch}: {e}")

# --- Main CLI ---

def main():
    parser = argparse.ArgumentParser(description="DeepDream-MLX Model Converter")
    parser.add_argument("--scan", default="toConvert", help="Directory to scan for local files")
    parser.add_argument("--download", choices=["alexnet", "resnet50", "vgg16", "googlenet", "all"], 
                        help="Download and convert specific Places365 models")
    parser.add_argument("--dest", default=".", help="Output directory for .npz files")
    args = parser.parse_args()

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    # 1. Handle Downloads
    if args.download:
        if not os.path.exists(args.scan):
            os.makedirs(args.scan)
            
        targets = ["alexnet", "resnet50", "vgg16", "googlenet"] if args.download == "all" else [args.download]
        for t in targets:
            download_and_convert_places365(t, args.scan, args.dest)

    # 2. Handle Local Scan
    if os.path.exists(args.scan):
        print(f"\nScanning '{args.scan}' for local models...")
        files = glob.glob(os.path.join(args.scan, "*"))
        for f in files:
            if os.path.isdir(f): continue
            ext = os.path.splitext(f)[1].lower()
            
            if ext == ".t7":
                convert_torch7(f, args.dest)
            elif ext in [".pth", ".pt", ".tar", ".pkl"]:
                # If it looks like a downloaded places file we already processed, skip to avoid duplication
                # heuristic: if we just downloaded it. 
                convert_pytorch(f, args.dest)
            elif ext in [".caffemodel"]:
                print(f"⚠️  Skipping Caffe model {os.path.basename(f)} (Not supported)")

if __name__ == "__main__":
    main()
