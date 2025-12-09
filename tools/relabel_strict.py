#!/usr/bin/env python3
"""
Relabel a dataset into a strict set of categories using OpenRouter/Gemini.
Consolidates scattered classes into a clean, high-level schema for better training.
"""

import argparse
import base64
import mimetypes
import os
import shutil
import sys
import time
from pathlib import Path
import requests

# Ensure output is flushed immediately
sys.stdout.reconfigure(line_buffering=True)

print("Initializing Relabeler...", flush=True)

# Add parent directory to path to import image_utils
try:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from image_utils import show_inline
    print("Successfully imported image_utils.", flush=True)
except Exception as e:
    print(f"Error importing image_utils: {e}", flush=True)
    sys.exit(1)

# The 5 canonical Evangelion UI categories
CATEGORIES = [
    "WARNING_EMERGENCY",  # Red/Orange, hexagons, ALERT text, heavy bold patterns
    "TACTICAL_RADAR",     # Concentric circles, maps, targeting reticles, sonar
    "DATA_SCHEMATIC",     # Wireframes, MAGI code, technical diagrams, dense text lists
    "WAVEFORM_GRAPH",     # Sine waves, histograms, heart monitors, noise patterns
    "HUD_FRAMES",         # Minimalist lines, viewfinders, corners, UI borders
]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def encode_image(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"
    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def classify_image(model: str, api_key: str, image_path: Path) -> str:
    img_data_url = encode_image(image_path)
    
    prompt = (
        f"Analyze this Evangelion-style UI image. Classify it into exactly one of these {len(CATEGORIES)} categories:\n"
        f"{', '.join(CATEGORIES)}\n\n"
        "Reply ONLY with the exact category name. Do not explain."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://deepdream-mlx.local",
        "X-Title": "DeepDream Relabeler",
    }
    
    payload = {
        "model": model,
        "max_tokens": 16,
        "temperature": 0.1, # Low temp for deterministic classification
        "messages": [
            {"role": "system", "content": "You are a classifier for Neon Genesis Evangelion user interface assets."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_data_url}},
                ],
            },
        ],
    }

    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            # Cleanup response to match category
            clean_content = content.replace(" ", "_").upper()
            # Check for partial matches if the model gets chatty
            for cat in CATEGORIES:
                if cat in clean_content:
                    return cat
            
            if clean_content in CATEGORIES:
                return clean_content
                
            print(f"\n[Warn] Model returned invalid category: '{content}'. Retrying...", flush=True)
        except Exception as e:
            print(f"\n[Error] Request failed: {e}. Retrying...", flush=True)
            time.sleep(1)
    
    return "UNKNOWN"

def main():
    parser = argparse.ArgumentParser(description="Relabel dataset into strict categories")
    parser.add_argument("--source", required=True, help="Root directory of input dataset (can be nested)")
    parser.add_argument("--output", required=True, help="Output directory for clean dataset")
    parser.add_argument("--model", default="google/gemini-2.5-flash", help="OpenRouter model ID")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set.", flush=True)
        sys.exit(1)

    src_root = Path(args.source)
    dest_root = Path(args.output)
    
    if not src_root.exists():
        print(f"Error: Source directory {src_root} does not exist.", flush=True)
        sys.exit(1)

    # Create category folders
    for cat in CATEGORIES:
        (dest_root / cat).mkdir(parents=True, exist_ok=True)
    (dest_root / "UNKNOWN").mkdir(parents=True, exist_ok=True)

    # Find all images recursively
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [
        p for p in src_root.rglob("*") 
        if p.is_file() and p.suffix.lower() in extensions
    ]

    print(f"Found {len(image_files)} images in {src_root}", flush=True)
    print(f"Target Categories: {', '.join(CATEGORIES)}", flush=True)

    success_count = 0
    
    # Manual loop instead of tqdm to ensure print control
    total = len(image_files)
    for i, img_path in enumerate(image_files):
        try:
            # Visual separator
            print("-" * 40, flush=True)
            print(f"Processing {i+1}/{total}: {img_path.name}", flush=True)
            
            category = classify_image(args.model, api_key, img_path)
            
            # Colorized Output
            color_code = "\033[1;32m" if category != "UNKNOWN" else "\033[1;31m"
            reset_code = "\033[0m"
            print(f"Classified as: {color_code}{category}{reset_code}", flush=True)
            
            # Show Image
            try:
                show_inline(img_path)
            except Exception as e:
                print(f"Failed to display image: {e}", flush=True)
            
            dest_path = dest_root / category / img_path.name
            
            # Handle duplicates
            if dest_path.exists():
                stem = dest_path.stem
                suffix = dest_path.suffix
                dest_path = dest_root / category / f"{stem}_{int(time.time())}{suffix}"

            if args.move:
                shutil.move(str(img_path), str(dest_path))
            else:
                shutil.copy2(str(img_path), str(dest_path))
            
            if category != "UNKNOWN":
                success_count += 1
                
        except KeyboardInterrupt:
            print("\nStopping...", flush=True)
            break
        except Exception as e:
            print(f"\nFailed to process {img_path.name}: {e}", flush=True)

    print(f"\nDone! {success_count}/{len(image_files)} images classified successfully.", flush=True)
    print(f"New dataset located at: {dest_root}", flush=True)

if __name__ == "__main__":
    main()