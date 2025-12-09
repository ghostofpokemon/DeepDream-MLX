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

# Visualization & Registry
try:
    import mlx.core as mx
    import mlx.utils
    from deepdream import MODEL_REGISTRY
    from deepdream.visualization import traverse_model, print_header, print_footer
    VIZ_AVAILABLE = True
except ImportError as e:
    VIZ_AVAILABLE = False
    print(f"Warning: deepdream package logic missing ({e}). Visualization disabled.")

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
    "googlenet": "http://places2.csail.mit.edu/models_places365/googlenet_places365.pth.tar",
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
    "densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth"
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

# --- Visualization Logic ---

def visualize_conversion(arch, weights_path):
    if not VIZ_AVAILABLE: return
    
    print("\n")
    print_header(f"{arch.upper()} ARCHITECTURE")
    
    if arch not in MODEL_REGISTRY:
        print(f"   (Model '{arch}' not in registry, skipping tree viz)")
        return

    try:
        # Instantiate & Load
        entry = MODEL_REGISTRY[arch]
        model = entry["cls"]()
        model.load_npz(weights_path)
        
        # Viz
        traverse_model(model, arch)
        
        # Count
        flat_params = mlx.utils.tree_flatten(model.parameters())
        unique_params = {id(v): v for _, v in flat_params}
        total = sum(v.size for v in unique_params.values())
        print_footer(total)
        
    except Exception as e:
        print(f"   (Visualization Error: {e})")

def dream_and_show(arch, weights_path):
    """
    Run a quick 1-step dream and display it in the terminal.
    """
    try:
        from deepdream import run_dream, load_image
        from deepdream.ascii_art import render_image_to_string
        import os
        
        # Test Image
        test_img = "assets/test_image.jpg"
        if not os.path.exists(test_img):
            # Try to find any jpg
            jpgs = glob.glob("*.jpg")
            if jpgs: 
                test_img = jpgs[0]
            else:
                return # No image to test with

        print(f"\n   [Running Test Dream on {test_img}With {arch} ...]")
        
        # Load Model
        entry = MODEL_REGISTRY[arch]
        model = entry["cls"]()
        model.load_npz(weights_path)
        
        img_np = load_image(test_img, target_width=256) # Small for speed
        
        # Quick Dream (1 Octave, 2 Steps)
        dreamed, _ = run_dream(
            arch, 
            img_np, 
            steps=2, 
            octaves=1, 
            model_instance=model
        )
        
        # Save temp
        tmp_out = "tmp_dream_test.jpg"
        Image.fromarray(dreamed).save(tmp_out)
        
        # Render
        print("\n")
        print(render_image_to_string(tmp_out, width=60))
        print(f"   (Test Dream Generated: {tmp_out})")
        
        # Cleanup
        os.remove(tmp_out)

    except Exception as e:
        print(f"   (Dream Error: {e})")


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

def download_and_convert(arch, download_dir, target_dir):
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

    # 2. Convert based on type
    if arch in ["efficientnet_b0", "densenet121"]:
        # Direct state dict conversion
        convert_pytorch(filename, target_dir)
        # Auto-Visualize
        final_path = os.path.join(target_dir, f"{os.path.basename(filename).split('.')[0]}_mlx.npz")
        # Fixing path logic: convert_pytorch saves to {name_base}_mlx.npz
        # Using a consistent name construction would be better, but let's just reconstruct it:
        base = os.path.splitext(os.path.basename(filename))[0]
        if base.endswith(".pth"): base = base[:-4]
        predicted_path = os.path.join(target_dir, f"{base}_mlx.npz")
        
        visualize_conversion(arch, predicted_path)
        dream_and_show(arch, predicted_path)
    else:
        # Places365 Legacy path (requiring skeleton)
        bs = os.path.splitext(os.path.basename(filename))[0]
        # remove .pth if present in stem (for .pth.tar)
        if bs.endswith(".pth"): bs = bs[:-4]
        
        final_out_path = os.path.join(target_dir, f"{bs}_mlx.npz")
        if os.path.exists(final_out_path):
             print(f"Skipping {arch}, {final_out_path} already exists.")
             return

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
            np.savez(final_out_path, **final_dict)
            print(f"✅ Saved {final_out_path}")

        except Exception as e:
            print(f"Failed to process {arch}: {e}")

# --- Main CLI ---

def main():
    parser = argparse.ArgumentParser(description="DeepDream-MLX Model Converter")
    parser.add_argument("--scan", default="toConvert", help="Directory to scan for local files")
    parser.add_argument("--download", choices=["alexnet", "resnet50", "vgg16", "googlenet", "efficientnet_b0", "densenet121", "all"], 
                        help="Download and convert specific models")
    parser.add_argument("--dest", default=".", help="Output directory for .npz files")
    args = parser.parse_args()

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    # 1. Handle Downloads
    if args.download:
        if not os.path.exists(args.scan):
            os.makedirs(args.scan)
            
        targets = ["alexnet", "resnet50", "vgg16", "googlenet", "efficientnet_b0", "densenet121"] if args.download == "all" else [args.download]
        for t in targets:
            download_and_convert(t, args.scan, args.dest)

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
                # If it looks like a downloaded file we already processed, skip to avoid duplication
                # heuristic: if we just downloaded it. 
                convert_pytorch(f, args.dest)
            elif ext in [".caffemodel"]:
                print(f"⚠️  Skipping Caffe model {os.path.basename(f)} (Not supported)")

if __name__ == "__main__":
    main()
