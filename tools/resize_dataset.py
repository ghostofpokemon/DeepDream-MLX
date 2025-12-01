#!/usr/bin/env python3
"""Batch-resize an imagefolder dataset to a maximum side length."""

import argparse
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = 1_000_000_000


def resize_image(image_path: Path, max_size: int):
    img = Image.open(image_path).convert("RGB")
    if img.width <= max_size and img.height <= max_size:
        return False
    scale = max_size / max(img.width, img.height)
    new_size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
    resized = img.resize(new_size, Image.LANCZOS)
    resized.save(image_path)
    return True


def iter_images(root: Path):
    for file in sorted(root.rglob("*")):
        if file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            yield file


def main():
    parser = argparse.ArgumentParser(description="Resize every image in a class-structured dataset")
    parser.add_argument("data_path", help="Path to dataset root (class subfolders expected)")
    parser.add_argument("max_size", type=int, help="Longest side length after resize (pixels)")
    args = parser.parse_args()

    root = Path(args.data_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    count = 0
    resized = 0
    for img_path in iter_images(root):
        count += 1
        if resize_image(img_path, args.max_size):
            resized += 1
            print(f"Resized {img_path.relative_to(root)}")

    print(f"Processed {count} images | resized {resized}")


if __name__ == "__main__":
    main()
