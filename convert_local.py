#!/usr/bin/env python3
import os
import torch
import numpy as np
import glob

try:
    import torchfile
except ImportError:
    torchfile = None

SOURCE_DIR = "toConvert"
TARGET_DIR = "." # Root directory where dream.py lives

def convert_tensor(tensor, target_dtype=np.float16):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy().astype(target_dtype)
    elif isinstance(tensor, np.ndarray):
        return tensor.astype(target_dtype)
    else:
        return np.array(tensor).astype(target_dtype)

def clean_state_dict(state_dict):
    """
    Removes 'module.' prefixes and handles other common nesting issues.
    Returns a flat dictionary of numpy arrays.
    """
    new_dict = {}
    for k, v in state_dict.items():
        name = k
        
        # Remove DataParallel 'module.' artifact anywhere
        # (Some models like Places365 AlexNet have features.module.0...)
        name = name.replace("module.", "")
            
        # Remove 'features.' prefix for VGG if needed? 
        # Actually, our MLX VGG expects keys like "features.0.weight" or mapped names?
        # Let's check mlx_vgg16.py... 
        # It uses a manual mapping: `load_weight(f"features.{i}.0.weight", ...)`
        # So preserving 'features.' is actually CORRECT for VGG-style checkpoints.
        
        new_dict[name] = convert_tensor(v)
        
    return new_dict

def convert_t7_file(filepath):
    if torchfile is None:
        print("⚠️  Skipping .t7 file: 'torchfile' library not installed.")
        return

    print(f"Processing Torch7 file {filepath}...")
    try:
        # Torch7 files (Lua) often load as a list of layers or a legacy object
        model_obj = torchfile.load(filepath)
        
        # This is experimental. Torch7 structure mapping to PyTorch/MLX is complex.
        # We will try to extract weight/bias tensors.
        # Usually model_obj.modules contains the layers.
        
        converted_state = {}
        
        def extract_layers(layer, prefix=""):
            if hasattr(layer, 'weight') and layer.weight is not None:
                converted_state[f"{prefix}.weight"] = convert_tensor(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                converted_state[f"{prefix}.bias"] = convert_tensor(layer.bias)
                
            if hasattr(layer, 'modules') and layer.modules:
                for i, sublayer in enumerate(layer.modules):
                    # Lua uses 1-based indexing, Python 0-based. 
                    # We'll use 0-based here to match PyTorch conventions if it was Sequential.
                    extract_layers(sublayer, f"{prefix}.{i}" if prefix else f"{i}")
        
        extract_layers(model_obj)
        
        if not converted_state:
            print(f"❌ No weights found in {filepath} (Structure might be non-standard)")
            return

        # Heuristic: Guess Architecture
        print(f"   Extracted {len(converted_state)} tensors.")
        
        # Save
        name_base = os.path.splitext(os.path.basename(filepath))[0]
        out_name = f"{name_base}_t7_mlx.npz"
        out_path = os.path.join(TARGET_DIR, out_name)
        np.savez(out_path, **converted_state)
        print(f"✅ Saved {out_name}")
        
    except Exception as e:
        print(f"❌ Failed to convert .t7 {filepath}: {e}")

def convert_file(filepath):
    filename = os.path.basename(filepath)
    name_base, ext = os.path.splitext(filename)
    
    if ext.lower() == '.t7':
        convert_t7_file(filepath)
        return

    # Skip non-pytorch files
    if ext.lower() not in ['.pth', '.pt', '.pkl', '.tar']:
        if ext.lower() in ['.caffemodel']:
            print(f"⚠️  Skipping {filename}: MLX cannot convert raw Caffe files without Caffe installed.")
        return

    print(f"Processing {filename}...")
    
    try:
        # Load PyTorch Checkpoint
        checkpoint = torch.load(filepath, map_location="cpu")
        
        # Extract State Dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            print(f"❌ Could not interpret format of {filename}")
            return

        # Convert to Numpy
        clean_dict = clean_state_dict(state_dict)
        
        # Heuristic: Guess Architecture to warn user
        keys = list(clean_dict.keys())
        guess = "Unknown"
        if any("inception" in k for k in keys): guess = "GoogLeNet"
        elif any("layer4" in k for k in keys): guess = "ResNet"
        elif any("features.0" in k for k in keys): guess = "VGG/AlexNet"
        
        print(f"   Detected Arch: {guess} ({len(clean_dict)} tensors)")

        # Save
        out_name = f"{name_base}_mlx.npz"
        out_path = os.path.join(TARGET_DIR, out_name)
        
        np.savez(out_path, **clean_dict)
        
        size_mb = os.path.getsize(out_path) / (1024*1024)
        print(f"✅ Saved {out_name} ({size_mb:.1f} MB)")
        
    except Exception as e:
        print(f"❌ Failed to convert {filename}: {e}")

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Directory '{SOURCE_DIR}' not found.")
        return

    files = glob.glob(os.path.join(SOURCE_DIR, "*"))
    print(f"Scanning {SOURCE_DIR} for PyTorch models...")
    
    for f in files:
        if os.path.isfile(f):
            convert_file(f)

if __name__ == "__main__":
    main()
