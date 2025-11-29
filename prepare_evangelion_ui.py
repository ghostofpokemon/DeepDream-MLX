#!/usr/bin/env python3
"""
Prep Evangelion UI images into an ImageFolder layout with aspect-preserving letterbox or zoom+crop.
Source folder currently has a trailing space in its name; pass it explicitly.
"""

import argparse
import random
from pathlib import Path
from typing import Tuple

from PIL import Image


def pad_to_square(img: Image.Image, size: int, fill: Tuple[int, int, int], zoom: float) -> Image.Image:
    w, h = img.size
    scale = min(size * zoom / w, size * zoom / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    offset = ((size - new_w) // 2, (size - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def zoom_and_crop_center(img: Image.Image, size: int, zoom: float) -> Image.Image:
    w, h = img.size
    # scale so the smaller side covers the target, then crop center
    scale = max(size * zoom / w, size * zoom / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = max(0, (new_w - size) // 2)
    top = max(0, (new_h - size) // 2)
    right = left + size
    bottom = top + size
    return resized.crop((left, top, right, bottom))


def main():
    p = argparse.ArgumentParser(description="Prep Evangelion UI dataset with letterbox or zoom+crop")
    p.add_argument("--source", required=True, help="Source directory (accepts trailing space name)")
    p.add_argument("--dest", default="dataset/evangelion-ui", help="Output root for ImageFolder")
    p.add_argument("--class-name", default="evangelion_ui", help="Class folder name")
    p.add_argument("--size", type=int, default=256, help="Square size after padding")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Hold-out fraction for val split")
    p.add_argument("--mode", choices=["letterbox", "crop"], default="letterbox", help="Aspect strategy")
    p.add_argument("--zoom", type=float, default=1.0, help="Zoom factor before padding/cropping")
    args = p.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    dest_train = Path(args.dest) / "train" / args.class_name
    dest_val = Path(args.dest) / "val" / args.class_name
    dest_train.mkdir(parents=True, exist_ok=True)
    dest_val.mkdir(parents=True, exist_ok=True)

    images = [p for p in src.iterdir() if p.is_file()]
    random.seed(42)
    random.shuffle(images)
    if not images:
        raise SystemExit(f"No files found in {src}")

    val_cut = int(len(images) * args.val_ratio)
    val_files = set(images[:val_cut])

    fill = (123, 117, 104)  # ImageNet-ish mean to reduce border artifacts
    for idx, path in enumerate(images, 1):
        out_dir = dest_val if path in val_files else dest_train
        out_path = out_dir / path.name
        img = Image.open(path).convert("RGB")
        if args.mode == "letterbox":
            img = pad_to_square(img, args.size, fill=fill, zoom=args.zoom)
        else:
            img = zoom_and_crop_center(img, args.size, zoom=args.zoom)
        img.save(out_path)
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(images)}")

    print(f"Done. Train: {len(images) - len(val_files)}, Val: {len(val_files)}")
    print(f"Output root: {Path(args.dest).resolve()}")


if __name__ == "__main__":
    main()
