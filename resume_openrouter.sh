#!/bin/bash
set -euo pipefail
python label_evangelion_with_openrouter.py \
  --source "EvangelionUserInterfaces " \
  --output dataset/evangelion-ui-openrouter \
  --split train \
  --model "openrouter/nvidia/nemotron-nano-12b-v2-vl:free" \
  --copy \
  --prompt "Give a descriptive 1-2 word label for this Evangelion UI screenshot." \
  "$@"
