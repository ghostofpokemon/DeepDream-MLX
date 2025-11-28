#!/usr/bin/env bash
# Quick head-to-head timing of historic DeepDream scripts at width 400 using love.jpg.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG="${ROOT}/love.jpg"
OUT="${ROOT}/bench_versions/out"
mkdir -p "${OUT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

run() {
  local name="$1"; shift
  echo "---- ${name} ----"
  /usr/bin/time -p "$@" || echo "FAILED: ${name}"
  echo
}

cd "${ROOT}"

# Oldest: positional input
run "9f21d03_deepdream.py" \
  python bench_versions/dream_9f21d03_deepdream.py "${IMG}" --model googlenet --width 400 --steps 8 --output "${OUT}/out_9f21d03_deepdream.jpg"

# Old-ish: --input flag, img_width
run "9f21d03_dream_mlx.py" \
  python bench_versions/dream_9f21d03_dream_mlx.py --model googlenet --input "${IMG}" --img_width 400 --steps 8 --output "${OUT}/out_9f21d03_dream_mlx.jpg"

# Args refactor era
run "97b4af6.py" \
  python bench_versions/dream_97b4af6.py --model googlenet --input "${IMG}" --width 400 --steps 8 --output "${OUT}/out_97b4af6.jpg"

# Pre-auto-download
run "48a1ec7.py" \
  python bench_versions/dream_48a1ec7.py --model googlenet --input "${IMG}" --width 400 --steps 8 --output "${OUT}/out_48a1ec7.jpg"

# Current
run "current" \
  python bench_versions/dream_current.py --model googlenet --input "${IMG}" --width 400 --steps 8 --output "${OUT}/out_current.jpg"
