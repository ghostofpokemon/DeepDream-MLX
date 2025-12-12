#!/usr/bin/env python3
"""
Looking Glass: AI Image Labeler for DeepDream-MLX
Uses OpenRouter (Gemini 2.5/Flash) to classify images into folders.
Structure: output/split/class_name/image.jpg
Compatible with train_dream.py (ImageFolder).
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import base64
import json
import os
import shutil
import requests
import time
from typing import Optional
from deepdream.term_image import print_image

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_label_from_openrouter(image_path: str, api_key: str, model: str, prompt: str) -> Optional[str]:
    base64_image = encode_image(image_path)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ghostofpokemon/DeepDream-MLX",
        "X-Title": "DeepDream-MLX Labeler"
    }
    
    # User asked for "5 labels" context, but training requires 1 class. 
    # We will ask for the "Single Best Category" to organize folders.
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}\nReturn ONLY the label. No punctuation."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data['choices'][0]['message']['content'].strip()
        # Clean slug
        slug = "".join(ch for ch in content.lower() if ch.isalnum() or ch in (" ", "-", "_")).strip()
        slug = "-".join(slug.split())
        return slug
    except Exception as e:
        print(f"Error calling OpenRouter: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="AI Image Labeler (Folder Organizer)")
    parser.add_argument("input", help="Directory of source images")
    parser.add_argument("--output", default="dataset/labeled", help="Output root directory")
    parser.add_argument("--split", default="train", help="Dataset split (train/val)")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"), help="OpenRouter API Key")
    parser.add_argument("--model", default="google/gemini-2.0-flash-exp:free", help="Model ID")
    parser.add_argument("--prompt", default="Classify this image into a single short 1-2 word category based on its dominant style or subject.", help="Prompt")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving")
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: API Key required. Set OPENROUTER_API_KEY env var or pass --api-key.")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input {args.input} not found.")
        sys.exit(1)
        
    dest_root = os.path.join(args.output, args.split)
    os.makedirs(dest_root, exist_ok=True)

    image_files = [
        os.path.join(args.input, f) 
        for f in os.listdir(args.input) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    image_files.sort()
    
    print(f"Found {len(image_files)} images. Organizing into {dest_root}...")
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        # 1. Preview
        print("\n" + "-"*40)
        try:
            print_image(img_path)
        except:
            print(f"[Image: {filename}]")
            
        # 2. Label
        print(f"Classifying {filename}...")
        label = get_label_from_openrouter(img_path, args.api_key, args.model, args.prompt)
        
        if label:
            print(f"\033[92m -> Class: {label}\033[0m")
            
            # 3. Move/Copy to Folder
            class_dir = os.path.join(dest_root, label)
            os.makedirs(class_dir, exist_ok=True)
            target_path = os.path.join(class_dir, filename)
            
            try:
                if args.copy:
                    shutil.copy2(img_path, target_path)
                    print(f"   Copied to {class_dir}/")
                else:
                    shutil.move(img_path, target_path)
                    print(f"   Moved to {class_dir}/")
            except Exception as e:
                print(f"   Error moving file: {e}")
        else:
            print("\033[91m   Skipping (Label failed)\033[0m")
            
        time.sleep(0.5)

if __name__ == "__main__":
    main()
