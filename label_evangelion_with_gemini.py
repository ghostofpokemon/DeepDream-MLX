#!/usr/bin/env python3
"""
Auto-label Evangelion UI frames using Google's Gemini 2.5 Flash.

This script calls the Generative AI API for each image, gets a short label,
and reorganizes the dataset into ImageFolder format:
    output_root/train/<label_slug>/*.jpg

Requirements:
    pip install google-generativeai Pillow tqdm python-slugify
    export GEMINI_API_KEY="your-key"
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from PIL import Image
from slugify import slugify
from tqdm import tqdm


def describe_image(model, image_path: Path, prompt: str, max_tokens: int) -> Optional[str]:
    img = Image.open(image_path).convert("RGB")
    response = model.generate_content(
        [prompt, img],
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens, temperature=0.5, top_p=0.8
        ),
    )
    text = response.text.strip() if response.text else ""
    return text or None


def main():
    parser = argparse.ArgumentParser(description="Label Evangelion UI images with Gemini")
    parser.add_argument("--source", required=True, help="Folder with raw images (e.g. 'EvangelionUserInterfaces ')")
    parser.add_argument("--output", default="dataset/evangelion-ui-labeled", help="ImageFolder root to write")
    parser.add_argument("--split", default="train", help="Subfolder split name (train/val/etc)")
    parser.add_argument("--prompt", default="In one or two words, name the primary object or motif in this UI mockup.", help="Prompt sent to Gemini")
    parser.add_argument("--max-tokens", type=int, default=8, help="Max tokens for the label response")
    parser.add_argument("--dry-run", action="store_true", help="Print labels without copying files")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running this script.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"Source folder not found: {src}")

    output_root = Path(args.output) / args.split
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    for path in tqdm(list(src.glob("*.*")), desc="Labeling images"):
        if not path.is_file():
            continue
        label = describe_image(model, path, args.prompt, args.max_tokens)
        if not label:
            print(f"Could not get label for {path.name}, skipping.")
            continue
        label_slug = slugify(label, lowercase=True) or "unknown"
        if args.dry_run:
            print(f"{path.name} -> {label_slug}")
            continue
        target_dir = output_root / label_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / path.name
        path.replace(target_path)

    print(f"Finished labeling! Check {output_root} for ImageFolder layout.")


if __name__ == "__main__":
    main()
