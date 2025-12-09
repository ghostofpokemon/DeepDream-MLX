#!/usr/bin/env python3
"""Utility helpers for dream output post-processing."""

import base64
import sys
from pathlib import Path
from typing import Union

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

PathLike = Union[str, Path]


def create_comparison_image(base_path: PathLike, tuned_path: PathLike, compare_path: PathLike) -> Path:
    base_path = Path(base_path)
    tuned_path = Path(tuned_path)
    compare_path = Path(compare_path)

    with Image.open(base_path) as img1, Image.open(tuned_path) as img2:
        if img1.height != img2.height:
            ratio = img1.height / img2.height
            resized_width = max(1, int(img2.width * ratio))
            img2 = img2.resize((resized_width, img1.height))
        total_width = img1.width + img2.width
        max_height = max(img1.height, img2.height)
        comparison = Image.new("RGB", (total_width, max_height))
        comparison.paste(img1, (0, 0))
        comparison.paste(img2, (img1.width, 0))
        comparison.save(compare_path)
    return compare_path


def show_inline(path: PathLike, width: int = 800):
    path = Path(path)
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    sys.stdout.write(
        f"\033]1337;File=inline=1;width={width}px;height=auto;preserveAspectRatio=1;name={path.name}:{b64}\a\n"
    )
