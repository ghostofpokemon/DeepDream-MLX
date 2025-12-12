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
    import mlx.utils
    from deepdream import MODEL_REGISTRY
    from deepdream.visualization import traverse_model, print_header, print_footer
    from deepdream.term_image import print_image
    VIZ_AVAILABLE = True
except ImportError as e:
    VIZ_AVAILABLE = False
    print(f"Warning: deepdream package logic missing ({e}). Visualization disabled.")

# Optional Torchfile for .t7 support
try:
    import torchfile
except ImportError:
    torchfile = None

try:
    import timm
except ImportError:
    timm = None

# --- Configuration ---
HF_BASE = "https://huggingface.co/nickmystic/DeepDream-MLX/resolve/main/models"
# Map architecture names to their hosted filename.
# If values are None, we assume f"{arch}_mlx.npz"
PLACES365_URLS = {
    "alexnet": f"{HF_BASE}/alexnet_mlx.npz",
    "resnet50": f"{HF_BASE}/resnet50_mlx.npz",
    "vgg16": f"{HF_BASE}/vgg16_mlx.npz",
    "googlenet": f"{HF_BASE}/googlenet_mlx.npz",
    "efficientnet_b0": f"{HF_BASE}/efficientnet_b0_mlx.npz",
    "densenet121": f"{HF_BASE}/densenet121_mlx.npz"
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
        
        # Inline Image
        try:
             print_image(tmp_out)
        except: pass
        
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

    filename = os.path.join(target_dir, os.path.basename(url))
    
    # 1. Check if already exists
    if os.path.exists(filename):
        print(f"Found existing {filename}")
        return

    # 2. Download
    print(f"Downloading {arch} from {url}...")
    try:
        download_url_to_file(url, filename)
        print(f"✅ Downloaded {filename}")
        
        # If it's an NPZ, we are done (pre-converted)
        if filename.endswith(".npz"):
            size_mb = os.path.getsize(filename) / (1024*1024)
            print(f"   (Size: {size_mb:.1f} MB)")
            # Optional: Visualize
            visualize_conversion(arch, filename)
            # dream_and_show(arch, filename) # Optional
            return

    except Exception as e:
        print(f"Download failed: {e}")
        return

    # Fallback to conversion if we somehow downloaded a non-npz (legacy support)
    # This block is only reached if we changed the URL back to a .pth file
    if not filename.endswith(".npz"):
        print(f"Detected non-npz file {filename}, attempting legacy conversion...")
        # ... logic would go here, but strictly relying on NPZ for now ...
        convert_pytorch(filename, target_dir)

# --- Main CLI ---

def download_timm_weights(model_name, target_dir):
    if timm is None:
        print("❌ 'timm' library not installed. Run `pip install timm`.")
        return

    print(f"Downloading/Converting {model_name} from timm...")
    try:
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        state_dict = model.state_dict()
        clean_dict = clean_state_dict(state_dict)
        
        out_path = os.path.join(target_dir, f"{model_name}_mlx.npz")
        np.savez(out_path, **clean_dict)
        
        size_mb = os.path.getsize(out_path) / (1024*1024)
        print(f"✅ Saved {out_path} ({size_mb:.1f} MB)")
        
        if model_name == "convnextv2_tiny":
             visualize_conversion("convnext_tiny", out_path)

    except Exception as e:
        print(f"❌ Failed to process {model_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="DeepDream-MLX Model Converter")
    parser.add_argument("--scan", default="toConvert", help="Directory to scan for local files")
    parser.add_argument("--download", help="Download and convert specific models (supports timm models)")
    parser.add_argument("--dest", default="models", help="Output directory for .npz files")
    args = parser.parse_args()

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    # 1. Handle Downloads
    if args.download:
        if not os.path.exists(args.scan):
            os.makedirs(args.scan)
            
        # If it's a known non-timm model, use the old path
        legacy_models = ["alexnet", "resnet50", "vgg16", "googlenet", "efficientnet_b0", "densenet121"]
        
        if args.download == "all":
            targets = legacy_models + ["convnextv2_tiny"]
            for t in targets:
                if t == "convnextv2_tiny":
                    download_timm_weights(t, args.dest)
                else:
                    download_and_convert(t, args.scan, args.dest)
        else:
            # Check if it's a legacy supported model
            if args.download in legacy_models:
                 download_and_convert(args.download, args.scan, args.dest)
            else:
                 # Assume it's a timm model (e.g. xception71, convnextv2_tiny, etc)
                 download_timm_weights(args.download, args.dest)

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
