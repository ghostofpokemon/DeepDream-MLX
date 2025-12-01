#!/usr/bin/env python3
"""Randomized DeepDream side-by-side comparisons for base vs fine-tuned weights."""

import argparse
import subprocess
import time
from pathlib import Path

from dream_params import (
    MODEL_CONFIG,
    DEFAULT_INPUT,
    resolve_dream_script,
    ensure_dream_ready_npz,
    random_params,
    apply_overrides,
)
from image_utils import create_comparison_image


def run_dream_once(
    args,
    dream_script: Path,
    layers,
    steps,
    lr,
    octaves,
    scale,
    jitter,
    smoothing,
    weights,
    suffix: str,
    timestamp: str,
) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        "--width",
        str(args.width),
        "--steps",
        str(steps),
        "--lr",
        f"{lr:.4f}",
        "--octaves",
        str(octaves),
        "--scale",
        f"{scale:.3f}",
        "--jitter",
        str(jitter),
        "--smoothing",
        f"{smoothing:.3f}",
        "--weights",
        weights,
        "--layers",
        *layers,
    ]
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_path


def run_comparison(args, dream_script: Path, params: dict, suffix: str, run_idx: int | None = None) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    label = f"{suffix}{run_idx if run_idx is not None else ''}".rstrip()
    base_path = run_dream_once(
        args,
        dream_script,
        params["layers"],
        params["steps"],
        params["lr"],
        params["octaves"],
        params["scale"],
        params["jitter"],
        params["smoothing"],
        args.base,
        suffix=f"{label}_base",
        timestamp=timestamp,
    )
    tuned_path = run_dream_once(
        args,
        dream_script,
        params["layers"],
        params["steps"],
        params["lr"],
        params["octaves"],
        params["scale"],
        params["jitter"],
        params["smoothing"],
        args.tuned,
        suffix=f"{label}_tuned",
        timestamp=timestamp,
    )
    compare_name = f"{Path(args.input).stem}_{args.model}_{label}_compare_{timestamp}.jpg"
    compare_path = Path(args.output_dir) / compare_name
    create_comparison_image(base_path, tuned_path, compare_path)
    print(f"Comparison saved to {compare_path}")
    return compare_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base vs tuned weights with randomized dream params")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input image path")
    parser.add_argument("--model", choices=list(MODEL_CONFIG.keys()), default="vgg16")
    parser.add_argument("--base", help="Path to base weights (defaults to model preset)")
    parser.add_argument("--tuned", required=True, help="Path to tuned/exported weights")
    parser.add_argument("--width", type=int, default=600, help="Resize width for dreams")
    parser.add_argument("--output-dir", default="tmp", help="Directory for per-run outputs and comparisons")
    parser.add_argument("--dream-script", help="Path to dream_mlx.py/dream.py (auto-detect if omitted)")
    parser.add_argument("--count", type=int, default=3, help="Number of randomized comparisons")
    parser.add_argument("--skip-hero", action="store_true", help="Skip the hero preset run")
    parser.add_argument("--classic", action="store_true", help="Bias random params to classic DeepDream ranges")
    parser.add_argument("--layers", nargs="+", help="Override dream layers")
    parser.add_argument("--steps", type=int, help="Override gradient steps")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--octaves", type=int, help="Override octave count")
    parser.add_argument("--scale", type=float, help="Override octave scale")
    parser.add_argument("--jitter", type=int, help="Override jitter amount")
    parser.add_argument("--smoothing", type=float, help="Override smoothing coefficient")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = MODEL_CONFIG[args.model]
    if not args.base:
        args.base = cfg["weights"]

    dream_script = resolve_dream_script(args.dream_script)
    base_path = Path(args.base).resolve()
    tuned_path = Path(args.tuned).resolve()
    ensure_dream_ready_npz(base_path)
    ensure_dream_ready_npz(tuned_path)
    args.base = str(base_path)
    args.tuned = str(tuned_path)

    hero_params = apply_overrides(cfg["hero"], args)

    if not args.skip_hero:
        print("Running hero comparison...")
        run_comparison(args, dream_script, hero_params, suffix="hero")

    for idx in range(args.count):
        params = random_params(
            cfg["layer_pool"],
            args.width,
            classic=args.classic,
            classic_layers=cfg.get("classic_layers"),
            classic_ranges=cfg.get("classic_ranges"),
        )
        params = apply_overrides(params, args)
        print(f"Random compare {idx + 1}/{args.count}: {params}")
        run_comparison(args, dream_script, params, suffix="rand", run_idx=idx + 1)
if __name__ == "__main__":
    main()
