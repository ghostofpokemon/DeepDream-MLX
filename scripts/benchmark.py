#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
from datetime import datetime
import json

# Benchmark Configuration
MODELS = ["googlenet", "vgg16", "resnet50"] # vgg19 often similar to vgg16, skipping for speed unless requested
PRECISIONS = ["int8", "bf16", "float32"]
INPUT_IMAGE = "assets/demo_googlenet.jpg" # Use a standard asset if available, or fallback
OUTPUT_DIR = "benchmark_results"

def ensure_asset():
    """Ensures a test image exists."""
    if not os.path.exists(INPUT_IMAGE):
        # Fallback if specific asset missing
        candidates = [f for f in os.listdir("assets") if f.endswith(".jpg")]
        if candidates:
            return os.path.join("assets", candidates[0])
        else:
            raise FileNotFoundError("No test image found in assets/")
    return INPUT_IMAGE

def get_weight_file(model, precision):
    """Maps model+precision to expected filename."""
    suffix = ""
    if precision == "int8":
        suffix = "_mlx_int8.npz"
    elif precision == "bf16":
        suffix = "_mlx_bf16.npz"
    elif precision == "float32":
        suffix = "_mlx.npz"
    
    return f"{model}{suffix}"

def run_benchmark():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    test_img = ensure_asset()
    results = []

    print(f"Starting Benchmark on {test_img}...")
    print(f"{ 'Model':<15} {'Precision':<10} {'Time (s)':<10} {'Status':<10}")
    print("-" * 50)

    for model in MODELS:
        for prec in PRECISIONS:
            weight_file = get_weight_file(model, prec)
            
            if not os.path.exists(weight_file):
                print(f"{model:<15} {prec:<10} {'---':<10} {'Missing Weights'}")
                continue

            # Run dream.py
            # We use a fixed seed or settings for consistency if possible, 
            # but dream.py is deterministic given same args usually.
            # We limit steps to 5 for speed, or use default 10? Default 10 is better for realistic timing.
            
            out_path = os.path.join(OUTPUT_DIR, f"bench_{model}_{prec}.jpg")
            
            cmd = [
                "python", "dream.py",
                "--input", test_img,
                "--output", out_path,
                "--model", model,
                "--weights", weight_file,
                "--steps", "10",
                "--width", "400"
            ]
            
            start_t = time.time()
            try:
                # Capture output to avoid clutter
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                duration = time.time() - start_t
                print(f"{model:<15} {prec:<10} {duration:.2f}       {'OK'}")
                results.append({
                    "model": model,
                    "precision": prec,
                    "time": duration,
                    "image": out_path
                })
            except subprocess.CalledProcessError:
                print(f"{model:<15} {prec:<10} {'Error':<10} {'Failed'}")

    # Generate Report
    generate_report(results)
    create_composite_image(results)

def generate_report(results):
    report_path = os.path.join(OUTPUT_DIR, "BENCHMARK_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# DeepDream MLX Benchmark Report\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Model | Precision | Time (s) | Result |\n")
        f.write("|-------|-----------|----------|--------|\n")
        
        for r in results:
            rel_img = os.path.basename(r['image'])
            f.write(f"| {r['model']} | {r['precision']} | {r['time']:.2f} | <img src='{rel_img}' width='100'/> |\n")
            
    print(f"\nReport generated at {report_path}")

def create_composite_image(results):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("PIL not installed, skipping composite image.")
        return

    # Organize data
    # matrix[model][precision] = image_path
    matrix = {}
    all_models = sorted(list(set(r['model'] for r in results)))
    all_precs = sorted(list(set(r['precision'] for r in results)))
    
    for r in results:
        if r['model'] not in matrix:
            matrix[r['model']] = {}
        matrix[r['model']][r['precision']] = r['image']

    if not matrix:
        return

    # Determine sizes
    # Assume all images roughly same size, read first found
    sample_img = Image.open(results[0]['image'])
    w, h = sample_img.size
    
    # Layout: Header row (precisions), Left col (models)
    padding = 50
    header_height = 60
    label_width = 120
    
    grid_w = label_width + len(all_precs) * (w + padding)
    grid_h = header_height + len(all_models) * (h + padding)
    
    composite = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(composite)
    
    # Try to load a font, else default
    try:
        font = ImageFont.truetype("Arial", 24)
    except IOError:
        font = ImageFont.load_default()

    # Draw Header
    for i, prec in enumerate(all_precs):
        x = label_width + i * (w + padding)
        draw.text((x + w//2 - 20, 20), prec, fill=(0,0,0), font=font)

    # Draw Rows
    for j, model in enumerate(all_models):
        y = header_height + j * (h + padding)
        # Model Label
        draw.text((10, y + h//2), model, fill=(0,0,0), font=font)
        
        for i, prec in enumerate(all_precs):
            x = label_width + i * (w + padding)
            if prec in matrix[model]:
                img_path = matrix[model][prec]
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    if img.size != (w, h):
                        img = img.resize((w, h))
                    composite.paste(img, (x, y))
                    
                    # Draw time
                    time_val = next(r['time'] for r in results if r['model'] == model and r['precision'] == prec)
                    draw.text((x + 5, y + h + 5), f"{time_val:.2f}s", fill=(0,0,0), font=font)

    comp_path = os.path.join(OUTPUT_DIR, "benchmark_composite.jpg")
    composite.save(comp_path)
    print(f"Composite benchmark image saved to {comp_path}")

if __name__ == "__main__":
    run_benchmark()