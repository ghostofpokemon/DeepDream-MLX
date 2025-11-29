#!/usr/bin/env python3
"""
Quick OpenRouter label for a single image (love.jpg).
"""

import base64
import mimetypes
import os
from pathlib import Path

import base64
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash"
PROMPT = "In 1-3 words, describe the dominant object or motif in this Evangelion-style UI mockup."
IMAGE_PATH = Path("love.jpg")


def encode_image(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def imgcat(path: Path):
    import sys
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    sys.stdout.write(f"\033]1337;File=inline=1;width=400px;height=auto;preserveAspectRatio=1;name={path.name}:{b64}\a\n")


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY first.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get(
            "OPENROUTER_REFERER", "https://deepdream-mlx.local"
        ),
        "X-Title": os.environ.get("OPENROUTER_TITLE", "DeepDream-MLX Labeler"),
    }
    payload = {
        "model": MODEL,
        "max_tokens": 24,
        "temperature": 0.4,
        "messages": [
            {
                "role": "system",
                "content": "You return short descriptive labels (1-3 words).",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image(IMAGE_PATH)},
                    },
                ],
            },
        ],
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code >= 400:
        print("Error:", resp.text)
        resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    label = ""
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                label = part.get("text", "")
    elif isinstance(content, str):
        label = content
    print("Label:", label.strip())
    imgcat(IMAGE_PATH)


if __name__ == "__main__":
    main()
