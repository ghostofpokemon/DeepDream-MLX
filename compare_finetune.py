#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from PIL import Image

def run_dream(args, model, input_img, weights, output_img):
    cmd = [
        sys.executable, "dream.py",
        "--input", input_img,
        "--output", output_img,
        "--model", model,
        "--weights", weights,
        "--steps", str(args.steps),
        "--lr", str(args.lr),
        "--octaves", str(args.octaves),
        "--scale", str(args.scale),
        "--jitter", str(args.jitter),
        "--smoothing", str(args.smoothing),
    ]
    if args.layers:
        cmd += ["--layers"] + args.layers
    if args.width:
        cmd += ["--width", str(args.width)]
        
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned weights")
    parser.add_argument("--model", required=True, help="Model name (e.g. vgg16)")
    parser.add_argument("--base", required=True, help="Path to base weights (e.g. vgg16_mlx_bf16.npz)")
    parser.add_argument("--tuned", required=True, help="Path to tuned weights (e.g. exports/vgg16.npz)")
    parser.add_argument("--input", required=True, help="Input image")
    parser.add_argument("--output", default="comparison.jpg", help="Output comparison image")
    
    # Dream params
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.09)
    parser.add_argument("--octaves", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1.8)
    parser.add_argument("--jitter", type=int, default=32)
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--layers", nargs="+", help="Layers to optimize")
    
    args = parser.parse_args()
    
    base_out = "tmp_base.jpg"
    tuned_out = "tmp_tuned.jpg"
    
    try:
        # Run Base
        print("=== Generatng Base Dream ===")
        run_dream(args, args.model, args.input, args.base, base_out)
        
        # Run Tuned
        print("\n=== Generating Tuned Dream ===")
        run_dream(args, args.model, args.input, args.tuned, tuned_out)
        
        # Stitch
        print("\n=== Creating Comparison ===")
        img1 = Image.open(base_out)
        img2 = Image.open(tuned_out)
        
        # Resize if needed to match heights
        if img1.height != img2.height:
            ratio = img1.height / img2.height
            img2 = img2.resize((int(img2.width * ratio), img1.height))
            
        total_width = img1.width + img2.width
        max_height = max(img1.height, img2.height)
        
        comp = Image.new('RGB', (total_width, max_height))
        comp.paste(img1, (0, 0))
        comp.paste(img2, (img1.width, 0))
        
        comp.save(args.output)
        print(f"Comparison saved to {args.output}")
        
    finally:
        if os.path.exists(base_out): os.remove(base_out)
        if os.path.exists(tuned_out): os.remove(tuned_out)

if __name__ == "__main__":
    main()
