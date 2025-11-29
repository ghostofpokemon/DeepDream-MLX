#!/usr/bin/env python3
"""
Auto-label Evangelion UI frames via OpenRouter multimodal models.

Usage:
    export OPENROUTER_API_KEY=sk-...
    python label_evangelion_with_openrouter.py \
        --source "EvangelionUserInterfaces " \
        --output dataset/evangelion-ui-labeled \
        --split train \
        --model "google/gemini-2.5-flash"

By default, files are *moved* into ImageFolder layout
    output/split/<label_slug>/filename.png
Use --copy to keep originals and copy labeled files instead.
"""

import argparse
import base64
import mimetypes
import os
import shutil
from pathlib import Path

import requests
from typing import Optional
from tqdm import tqdm


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def encode_image(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"
    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def show_inline(path: Path):
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    import sys
    sys.stdout.write(f"\033]1337;File=inline=1;width=300px;height=auto;preserveAspectRatio=1;name={path.name}:{b64}\a\n")


def label_image(model: str, api_key: str, prompt: str, image_path: Path, max_tokens: int, referer: str, title: str) -> Optional[str]:
    img_data_url = encode_image(image_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": referer or "https://deepdream-mlx.local",
        "X-Title": title or "DeepDream-MLX Labeler",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.5,
        "messages": [
            {"role": "system", "content": "You assign short descriptive labels (1-3 words) for UI screenshots."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_data_url}},
                ],
            },
        ],
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        return None
    data = resp.json()
    message = data["choices"][0]["message"]
    parts = message.get("content", [])
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                text = part.get("text")
                if text:
                    return text.strip()
    if isinstance(parts, str):
        return parts.strip()
    # fallback to message text
    return message.get("content", "").strip()


def main():
    parser = argparse.ArgumentParser(description="Label images with OpenRouter multimodal models")
    parser.add_argument("--source", required=True, help="Folder containing unlabeled images")
    parser.add_argument("--output", default="dataset/evangelion-ui-labeled", help="Root ImageFolder directory")
    parser.add_argument("--split", default="train", help="Split into which files are placed (train/val/etc)")
    parser.add_argument("--model", default="google/gemini-2.5-flash", help="OpenRouter model ID")
    parser.add_argument("--prompt", default="In 1-3 words, name the dominant motif in this EVA-styled UI.", help="Prompt text sent to the model")
    parser.add_argument("--max-tokens", type=int, default=16, help="Max output tokens")
    parser.add_argument("--copy", action="store_true", help="Copy instead of moving files into labeled folders")
    parser.add_argument("--referer", default=os.environ.get("OPENROUTER_REFERER", ""), help="Optional HTTP Referer header")
    parser.add_argument("--title", default=os.environ.get("OPENROUTER_TITLE", "DeepDream-MLX Labeler"), help="Optional X-Title header")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set.")

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"Source directory missing: {src}")

    dest_root = Path(args.output) / args.split
    dest_root.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(src.iterdir()) if p.is_file()]
    for path in tqdm(files, desc="Labeling"):
        try:
            label = label_image(
                args.model,
                api_key,
                args.prompt,
                path,
                args.max_tokens,
                args.referer,
                args.title,
            )
        except Exception as exc:
            print(f"Failed to label {path.name}: {exc}")
            continue

        if label:
            print(f"{path.name}: {label}")
        else:
            print(f"{path.name}: <empty label>")
        show_inline(path)

        label_slug = "".join(ch for ch in label.lower() if ch.isalnum() or ch in (" ", "-", "_")).strip() if label else ""
        label_slug = "-".join(label_slug.split()) or "unknown"
        target_dir = dest_root / label_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / path.name
        try:
            if args.copy:
                shutil.copy2(path, target_path)
            else:
                shutil.move(path, target_path)
        except Exception as exc:
            print(f"Failed to place {path.name} -> {target_path}: {exc}")

    print(f"Done. Inspect {dest_root} to confirm class folders.")


if __name__ == "__main__":
    main()
