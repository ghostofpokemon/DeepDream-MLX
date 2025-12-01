#!/usr/bin/env python3
"""Quick multi-run DeepDream sampler for the Vite app repo.

The first run uses a hand-tuned preset to ensure at least one "classic" dream.
Subsequent runs randomize key hyperparameters. Outputs land in ./tmp by default.
"""

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


def run_dream(
    args,
    dream_script: Path,
    layers,
    steps,
    lr,
    octaves,
    scale,
    jitter,
    smoothing,
    suffix,
    weights,
    timestamp: str | None = None,
):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
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
        weights,
        "--layers",
        *layers,
    ]

    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_path


def run_with_optional_comparison(
    args,
    dream_script: Path,
    layers,
    steps,
    lr,
    octaves,
    scale,
    jitter,
    smoothing,
    suffix,
):
    if args.tuned:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_out = run_dream(
            args,
            dream_script,
            layers,
            steps,
            lr,
            octaves,
            scale,
            jitter,
            smoothing,
            suffix=f"{suffix}_base",
            weights=args.weights,
            timestamp=timestamp,
        )
        tuned_out = run_dream(
            args,
            dream_script,
            layers,
            steps,
            lr,
            octaves,
            scale,
            jitter,
            smoothing,
            suffix=f"{suffix}_tuned",
            weights=args.tuned,
            timestamp=timestamp,
        )
        compare_name = f"{Path(args.input).stem}_{args.model}_{suffix}_compare_{timestamp}.jpg"
        compare_path = Path(args.output_dir) / compare_name
        create_comparison_image(base_out, tuned_out, compare_path)
        print(f"Comparison saved to {compare_path}")
    else:
        run_dream(
            args,
            dream_script,
            layers,
            steps,
            lr,
            octaves,
            scale,
            jitter,
            smoothing,
            suffix=suffix,
            weights=args.weights,
        )


def main():
    parser = argparse.ArgumentParser(description="Batch DeepDream sampler with random params")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input image path")
    parser.add_argument("--model", default="inception_v3", choices=list(MODEL_CONFIG.keys()), help="Model to use")
    parser.add_argument("--weights", help="Override weights path")
    parser.add_argument("--count", type=int, default=3, help="Number of random runs after the hero run")
    parser.add_argument("--width", type=int, default=500, help="Resize width (preserve aspect ratio)")
    parser.add_argument("--output-dir", default="tmp", help="Directory where outputs go")
    parser.add_argument("--dream-script", help="Path to dream_mlx.py (auto-detects parent repo if omitted)")
    parser.add_argument("--classic", action="store_true", help="Bias random runs toward the classic 2015 DeepDream look")
    parser.add_argument(
        "--tuned",
        help="Optional tuned/exported weights to generate side-by-side comparison outputs",
    )
    parser.add_argument("--layers", nargs="+", help="Override dream layers for hero/random runs")
    parser.add_argument("--steps", type=int, help="Override dream steps")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--octaves", type=int, help="Override octave/pyramid count")
    parser.add_argument("--scale", type=float, help="Override pyramid scaling factor")
    parser.add_argument("--jitter", type=int, help="Override jitter pixels")
    parser.add_argument(
        "--smoothing",
        type=float,
        help="Override smoothing coefficient",
    )
    args = parser.parse_args()

    cfg = MODEL_CONFIG[args.model]
    if not args.weights:
        args.weights = cfg["weights"]

    dream_script = resolve_dream_script(args.dream_script)
    weights_path = Path(args.weights).resolve()
    ensure_dream_ready_npz(weights_path)
    args.weights = str(weights_path)

    if args.tuned:
        tuned_path = Path(args.tuned).resolve()
        ensure_dream_ready_npz(tuned_path)
        args.tuned = str(tuned_path)

    hero = apply_overrides(cfg["hero"], args)
    print("Running hero preset...")
    run_with_optional_comparison(
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
        params = apply_overrides(params, args)
        print(f"Random run {idx + 1}/{args.count}: {params}")
        run_with_optional_comparison(
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
