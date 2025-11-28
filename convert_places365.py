#!/usr/bin/env python3
import os
import argparse
import torch
import torchvision.models as models
import numpy as np
from torch.hub import download_url_to_file

# Model Definitions and URLs
# Places365-CNN models (Standard)
PLACES365_URLS = {
    "alexnet": "http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar",
    "resnet50": "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar",
    "vgg16": "http://places2.csail.mit.edu/models_places365/vgg16_places365.pth.tar",
    "googlenet": "http://places2.csail.mit.edu/models_places365/googlenet_places365.pth.tar" 
}

def get_places365_model(arch):
    # Note: We use num_classes=365.
    if arch == "alexnet":
        model = models.alexnet(num_classes=365)
    elif arch == "resnet50":
        model = models.resnet50(num_classes=365)
    elif arch == "vgg16":
        model = models.vgg16(num_classes=365)
    elif arch == "googlenet":
        # Places365 googlenet often differs slightly, but let's try standard structure
        model = models.googlenet(num_classes=365, aux_logits=False)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model

def load_state_dict_robust(model, state_dict):
    # Handle 'module.' prefix often found in parallel-trained models
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    
    # Try strict, then non-strict
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Loaded state dict (Strict)")
    except RuntimeError as e:
        print(f"Strict load failed, trying non-strict. Error: {e}")
        keys = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded with missing/unexpected keys: {keys}")
    return model

def export_to_mlx(model, model_name, suffix="_places365"):
    print(f"Exporting {model_name} to MLX format...")
    model.eval()
    state = model.state_dict()
    converted_state = {}
    target_type = np.float16 # Default to fp16 for places365 to save space

    for k, v in state.items():
        v_np = v.cpu().detach().numpy()
        
        # Normalize names if needed, but usually PyTorch standard is fine
        # MLX-Dream usually expects standard keys matching the definition
        
        converted_state[k] = v_np.astype(target_type)

    out_name = f"{model_name}{suffix}_mlx.npz"
    np.savez(out_name, **converted_state)
    
    size_mb = os.path.getsize(out_name) / (1024*1024)
    print(f"✅ Saved {out_name} ({size_mb:.1f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Convert Places365 models to MLX")
    parser.add_argument("--model", choices=["alexnet", "resnet50", "vgg16", "googlenet", "all"], default="all")
    args = parser.parse_args()

    models_to_run = ["alexnet", "resnet50", "vgg16", "googlenet"] if args.model == "all" else [args.model]

    if not os.path.exists("toConvert"):
        os.makedirs("toConvert")

    for arch in models_to_run:
        print(f"\n=== Processing {arch} ===")
        
        # 1. Setup Model Architecture
        try:
            model = get_places365_model(arch)
        except Exception as e:
            print(f"Error creating model {arch}: {e}")
            continue

        # 2. Get Weights File
        url = PLACES365_URLS.get(arch)
        if not url:
            print(f"No URL known for {arch}")
            continue
            
        filename = os.path.join("toConvert", os.path.basename(url))
        
        if not os.path.exists(filename):
            print(f"Downloading {url} to {filename}...")
            try:
                download_url_to_file(url, filename)
            except Exception as e:
                print(f"Download failed: {e}")
                continue
        else:
            print(f"Found cached file: {filename}")

        # 3. Load Weights
        print("Loading weights into PyTorch model...")
        try:
            checkpoint = torch.load(filename, map_location="cpu")
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint 
            
            load_state_dict_robust(model, state_dict)
            
            # 4. Export
            export_to_mlx(model, arch)
            
        except Exception as e:
            print(f"Failed to process {arch}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
