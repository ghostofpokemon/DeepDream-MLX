#!/usr/bin/env python3
"""Quick multi-run DeepDream sampler for the Vite app repo.

The first run uses a hand-tuned preset to ensure at least one "classic" dream.
Subsequent runs randomize key hyperparameters. Outputs land in ./tmp by default.
"""

import argparse
import random
import subprocess
import time
from pathlib import Path

import numpy as np


MODEL_CONFIG = {
    "inception_v3": {
        "weights": "models/inception_v3_epoch012.npz",
        "hero": {
            "layers": ["Mixed_5d", "Mixed_6b", "Mixed_7b"],
            "steps": 80,
            "lr": 0.18,
            "octaves": 4,
            "scale": 1.55,
            "jitter": 40,
            "smoothing": 0.45,
        },
        "layer_pool": [
            "Conv2d_2b_3x3",
            "Conv2d_3b_1x1",
            "Mixed_5b",
            "Mixed_5c",
            "Mixed_5d",
            "Mixed_6a",
            "Mixed_6b",
            "Mixed_6c",
            "Mixed_6d",
            "Mixed_6e",
            "Mixed_7a",
            "Mixed_7b",
            "Mixed_7c",
        ],
        "classic_layers": [
            ["Mixed_5c", "Mixed_5d", "Mixed_6b"],
            ["Mixed_6b", "Mixed_6c"],
            ["Mixed_6e", "Mixed_7b"],
            ["Mixed_5d", "Mixed_6e", "Mixed_7c"],
        ],
        "classic_ranges": {
            "steps": (48, 96),
            "lr": (0.12, 0.22),
            "octaves": (4, 6),
            "scale": (1.35, 1.6),
            "jitter": (28, 48),
            "smoothing": (0.35, 0.55),
        },
    },
    "googlenet": {
        "weights": "models/googlenet_mlx.npz",
        "hero": {
            "layers": ["inception3b", "inception4c", "inception4d"],
            "steps": 64,
            "lr": 0.2,
            "octaves": 4,
            "scale": 1.45,
            "jitter": 32,
            "smoothing": 0.5,
        },
        "layer_pool": [
            "inception3a",
            "inception3b",
            "inception4a",
            "inception4b",
            "inception4c",
            "inception4d",
            "inception4e",
            "inception5a",
            "inception5b",
        ],
        "classic_layers": [
            ["inception3b", "inception4d"],
            ["inception4c", "inception4d", "inception5a"],
            ["inception4b", "inception4e"],
        ],
        "classic_ranges": {
            "steps": (48, 96),
            "lr": (0.12, 0.24),
            "octaves": (4, 6),
            "scale": (1.35, 1.6),
            "jitter": (24, 48),
            "smoothing": (0.4, 0.6),
        },
    },
    "vgg16": {
        "weights": "models/vgg16_mlx.npz",
        "hero": {
            "layers": ["relu4_3"],
            "steps": 80,
            "lr": 0.1,
            "octaves": 4,
            "scale": 1.6,
            "jitter": 24,
            "smoothing": 0.45,
        },
        "layer_pool": [
            "relu3_3",
            "relu4_1",
            "relu4_2",
            "relu4_3",
            "relu5_1",
            "relu5_2",
            "relu5_3",
        ],
        "classic_layers": [
            ["relu4_2"],
            ["relu4_3"],
            ["relu4_3", "relu5_1"],
        ],
        "classic_ranges": {
            "steps": (60, 110),
            "lr": (0.08, 0.14),
            "octaves": (4, 6),
            "scale": (1.4, 1.65),
            "jitter": (16, 36),
            "smoothing": (0.4, 0.55),
        },
    },
    "mobilenet": {
        "weights": "models/MobileNetV3_mlx.npz",
        "hero": {
            "layers": ["layer7", "layer9", "layer11"],
            "steps": 72,
            "lr": 0.18,
            "octaves": 4,
            "scale": 1.45,
            "jitter": 28,
            "smoothing": 0.45,
        },
        "layer_pool": [
            "layer0",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "layer5",
            "layer6",
            "layer7",
            "layer8",
            "layer9",
            "layer10",
            "layer11",
        ],
        "classic_layers": [
            ["layer6", "layer8"],
            ["layer7", "layer9"],
            ["layer8", "layer10", "layer11"],
        ],
        "classic_ranges": {
            "steps": (50, 90),
            "lr": (0.12, 0.22),
            "octaves": (3, 5),
            "scale": (1.3, 1.55),
            "jitter": (20, 40),
            "smoothing": (0.35, 0.55),
        },
    },
}


def resolve_dream_script(path_override: str | None) -> Path:
    if path_override:
        return Path(path_override).resolve()
    this_dir = Path(__file__).resolve().parent
    candidates = [
        this_dir / "dream.py",
        this_dir / "dream_mlx.py",
        this_dir.parent / "dream.py",
        this_dir.parent / "dream_mlx.py",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError("dream_mlx.py not found; use --dream-script to point to it")


def ensure_dream_ready_npz(path: Path):
    with np.load(path, allow_pickle=True) as data:
        if "params" in data.files:
            raise ValueError(
                f"{path} looks like a training checkpoint (contains 'params'). "
                "Use the exported dream npz from train_dream.py or convert the checkpoint first."
            )


def run_dream(args, dream_script: Path, layers, steps, lr, octaves, scale, jitter, smoothing, suffix):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = output_dir / f"{Path(args.input).stem}_{args.model}_{suffix}_{timestamp}.jpg"

    cmd = [
        "python",
        str(dream_script),
        "--input",
        args.input,
        "--output",
        str(out_path),
        "--model",
        args.model,
        "--img_width",
        str(args.width),
        "--steps",
        str(steps),
        "--lr",
        f"{lr:.4f}",
        "--pyramid_size",
        str(octaves),
        "--pyramid_ratio",
        f"{scale:.3f}",
        "--jitter",
        str(jitter),
        "--smoothing_coefficient",
        f"{smoothing:.3f}",
        "--weights",
        args.weights,
        "--layers",
        *layers,
    ]

    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def uniform_range(lo, hi):
    return random.uniform(lo, hi)


def random_params(layer_pool, base_width: int, classic: bool, classic_layers=None, classic_ranges=None, min_dim: float = 120.0):
    for _ in range(100):
        if classic and classic_layers:
            layers = random.choice(classic_layers)
        else:
            layer_count = random.randint(1, min(3, len(layer_pool)))
            layers = random.sample(layer_pool, k=layer_count)

        if classic and classic_ranges:
            oct_lo, oct_hi = classic_ranges.get("octaves", (3, 5))
            sc_lo, sc_hi = classic_ranges.get("scale", (1.25, 1.65))
            octaves = random.randint(int(oct_lo), int(oct_hi))
            scale = uniform_range(sc_lo, sc_hi)
        else:
            octaves = random.randint(2, 5)
            scale = random.uniform(1.2, 1.65)
        smallest = base_width / (scale ** (octaves - 1))
        if smallest < min_dim:
            continue
        if classic and classic_ranges:
            steps = random.randint(*classic_ranges.get("steps", (40, 90)))
            lr = uniform_range(*classic_ranges.get("lr", (0.08, 0.2)))
            jitter = random.randint(*classic_ranges.get("jitter", (24, 48)))
            smoothing = uniform_range(*classic_ranges.get("smoothing", (0.4, 0.6)))
        else:
            steps = random.randint(12, 48)
            lr = random.uniform(0.03, 0.22)
            jitter = random.randint(16, 64)
            smoothing = random.uniform(0.2, 0.8)
        params = {
            "layers": layers,
            "steps": steps,
            "lr": lr,
            "octaves": octaves,
            "scale": scale,
            "jitter": jitter,
            "smoothing": smoothing,
        }
        return params
    raise RuntimeError("Failed to sample valid random hyperparameters; try reducing --count or adjusting min_dim")


def main():
    parser = argparse.ArgumentParser(description="Batch DeepDream sampler with random params")
    parser.add_argument("--input", default="input/love.jpg", help="Input image path")
    parser.add_argument("--model", default="inception_v3", choices=list(MODEL_CONFIG.keys()), help="Model to use")
    parser.add_argument("--weights", help="Override weights path")
    parser.add_argument("--count", type=int, default=3, help="Number of random runs after the hero run")
    parser.add_argument("--width", type=int, default=500, help="Resize width (preserve aspect ratio)")
    parser.add_argument("--output-dir", default="tmp", help="Directory where outputs go")
    parser.add_argument("--dream-script", help="Path to dream_mlx.py (auto-detects parent repo if omitted)")
    parser.add_argument("--classic", action="store_true", help="Bias random runs toward the classic 2015 DeepDream look")
    args = parser.parse_args()

    cfg = MODEL_CONFIG[args.model]
    if not args.weights:
        args.weights = cfg["weights"]

    dream_script = resolve_dream_script(args.dream_script)
    weights_path = Path(args.weights).resolve()
    ensure_dream_ready_npz(weights_path)
    args.weights = str(weights_path)

    hero = cfg["hero"]
    print("Running hero preset...")
    run_dream(
        args,
        dream_script,
        hero["layers"],
        hero["steps"],
        hero["lr"],
        hero["octaves"],
        hero["scale"],
        hero["jitter"],
        hero["smoothing"],
        suffix="hero",
    )

    for idx in range(args.count):
        params = random_params(
            cfg["layer_pool"],
            args.width,
            classic=args.classic,
            classic_layers=cfg.get("classic_layers"),
            classic_ranges=cfg.get("classic_ranges"),
        )
        print(f"Random run {idx + 1}/{args.count}: {params}")
        run_dream(
            args,
            dream_script,
            params["layers"],
            params["steps"],
            params["lr"],
            params["octaves"],
            params["scale"],
            params["jitter"],
            params["smoothing"],
            suffix=f"rand{idx + 1}",
        )


if __name__ == "__main__":
    main()
