#!/usr/bin/env python3
"""Randomized DeepDream side-by-side comparisons for base vs fine-tuned weights."""

import argparse
import json
import shutil
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
from image_utils import create_comparison_image, show_inline


def cache_dir(args: argparse.Namespace, kind: str) -> Path:
    root = Path("compare_cache") / args.model / kind
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_cache_manifest(cache_dir: Path) -> dict:
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text())
        except Exception:
            return {}
    return {}


def save_cache_manifest(cache_dir: Path, manifest: dict):
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def build_cache_key(args: argparse.Namespace, params: dict, weights: str) -> str:
    return json.dumps(
        {
            "input": str(Path(args.input).resolve()),
            "model": args.model,
            "width": args.width,
            "layers": params["layers"],
            "steps": params["steps"],
            "lr": params["lr"],
            "octaves": params["octaves"],
            "scale": params["scale"],
            "jitter": params["jitter"],
            "smoothing": params["smoothing"],
            "weights": str(Path(weights).resolve()),
        },
        sort_keys=True,
    )


def reusable_name(args: argparse.Namespace, params: dict, tag: str) -> str:
    layer_slug = "-".join(params["layers"])
    return f"{Path(args.input).stem}_{args.model}_{layer_slug}_s{params['steps']}lr{params['lr']}o{params['octaves']}sc{params['scale']}_{tag}.jpg"


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


RAINBOW = [
    (255, 0, 0),
    (255, 127, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 0, 255),
    (111, 0, 255),
    (255, 0, 255),
]


def confetti(msg: str) -> str:
    pieces = []
    for i, ch in enumerate(msg):
        r, g, b = RAINBOW[i % len(RAINBOW)]
        pieces.append(f"\033[38;2;{r};{g};{b}m{ch}")
    pieces.append("\033[0m")
    return "".join(pieces)


def run_comparison(args, dream_script: Path, params: dict, suffix: str, run_idx: int | None = None) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    label = f"{suffix}{run_idx if run_idx is not None else ''}".rstrip()
    base_cache = cache_dir(args, "base")
    tuned_cache = Path(args.output_dir) / "tuned_cache"
    tuned_cache.mkdir(parents=True, exist_ok=True)
    manifest_base = load_cache_manifest(base_cache)
    manifest_tuned = load_cache_manifest(tuned_cache)
    base_key = build_cache_key(args, params, args.base)
    tuned_key = build_cache_key(args, params, args.tuned)

    base_path = None
    if base_key in manifest_base:
        candidate = base_cache / manifest_base[base_key]
        if candidate.exists():
            base_path = candidate
            print(confetti(f"[cache hit] base -> {base_path} 🎉"))
        else:
            manifest_base.pop(base_key, None)
            save_cache_manifest(base_cache, manifest_base)

    if base_path is None:
        name = reusable_name(args, params, "base")
        base_path = base_cache / name
        tmp = run_dream_once(
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
        shutil.move(tmp, base_path)
        manifest_base[base_key] = name
        save_cache_manifest(base_cache, manifest_base)

    tuned_path = None
    if tuned_key in manifest_tuned:
        staged = tuned_cache / manifest_tuned[tuned_key]
        if staged.exists():
            tuned_path = staged
            print(confetti(f"[cache hit] tuned -> {tuned_path} 🌈"))

    if tuned_path is None:
        name = reusable_name(args, params, f"{label}_tuned")
        tuned_path = tuned_cache / name
        tmp = run_dream_once(
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
        shutil.move(tmp, tuned_path)
        manifest_tuned[tuned_key] = tuned_path.name
        save_cache_manifest(tuned_cache, manifest_tuned)
    compare_name = f"{Path(args.input).stem}_{args.model}_{label}_compare_{timestamp}.jpg"
    compare_path = Path(args.output_dir) / compare_name
    create_comparison_image(base_path, tuned_path, compare_path)
    print(f"Comparison saved to {compare_path}")
    show_inline(compare_path)
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
    parser.add_argument("--count", type=int, default=3, help="Number of non-classic randomized comparisons")
    parser.add_argument("--classic-count", type=int, default=0, help="Number of classic-style randomized comparisons")
    parser.add_argument("--skip-hero", action="store_true", help="Skip the hero preset run")
    parser.add_argument("--classic", action="store_true", help="(Legacy) treat --count runs as classic comparisons")
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

    random_count = args.count
    classic_count = args.classic_count
    if args.classic:
        classic_count = args.count
        random_count = 0

    for idx in range(random_count):
        params = random_params(
            cfg["layer_pool"],
            args.width,
            classic=False,
            classic_layers=cfg.get("classic_layers"),
            classic_ranges=cfg.get("classic_ranges"),
        )
        params = apply_overrides(params, args)
        print(f"Random compare {idx + 1}/{random_count}: {params}")
        run_comparison(args, dream_script, params, suffix="rand", run_idx=idx + 1)

    for idx in range(classic_count):
        params = random_params(
            cfg["layer_pool"],
            args.width,
            classic=True,
            classic_layers=cfg.get("classic_layers"),
            classic_ranges=cfg.get("classic_ranges"),
        )
        params = apply_overrides(params, args)
        print(f"Classic compare {idx + 1}/{classic_count}: {params}")
        run_comparison(args, dream_script, params, suffix="classic", run_idx=idx + 1)
if __name__ == "__main__":
    main()
