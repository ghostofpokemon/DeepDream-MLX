---
model_name: DeepDream-MLX
model_description: Native, hardware-accelerated DeepDream for Apple Silicon.
language: en
library_name: mlx
license: apache-2.0
tags:
- mlx
- computer-vision
- art
- generative
- deepdream
pipeline_tag: image-to-image
---

# DeepDream-MLX

<img src="assets/deepdream_header.jpg" alt="DeepDream Header" width="100%"/>

**Status:** Fast + native. **Vibe:** 2015 hallucinations, 2025 silicon.

DeepDream-MLX brings the original psychedelic computer vision look to Apple Silicon using [MLX](https://github.com/ml-explore/mlx). No Caffe relics—just clean tensor ops, ready-to-go checkpoints, and a zoom-video pipeline.

## What You Get

- MLX checkpoints for GoogLeNet (Inception v1), VGG16/VGG19, ResNet50, AlexNet, plus Places365 + bf16 variants (all `.npz`, tracked with LFS).
- `dream.py`: full DeepDream CLI with presets (`--preset nb14/nb20/nb28`), guided dreaming (`--guide`), and `--model all` for side-by-side runs.
- `dream_video.py`: zoom feedback loop using `scipy.ndimage.zoom`, outputs frames for `ffmpeg`.
- `convert.py`: scan or download `.pth`/`.t7` checkpoints and convert them into MLX format while keeping `toConvert/` clean.
- `benchmark.py` + `quantize_experiment.py`: quick speed checks and quantization experiments on Apple GPUs.

## Install

Bring your own env (conda/uv/venv/none) as you prefer:

```bash
# using uv (optional)
uv pip install -r requirements.txt

# or plain pip
pip install -r requirements.txt
```

## Run a Dream

```bash
# Classic look (GoogLeNet, default layers inception3b/4c/4d)
python dream.py --input assets/demo_googlenet.jpg --output dream.jpg \
  --model googlenet --octaves 4 --scale 1.4 --steps 16

# Painterly textures (VGG16) with a preset
python dream.py --input assets/demo_vgg16.jpg --output dream_vgg16.jpg \
  --model vgg16 --preset nb20 --steps 20

# Guided dreaming
python dream.py --input assets/demo_vgg16.jpg --guide assets/demo_googlenet.jpg \
  --model vgg16 --layers relu4_3 --steps 18 --octaves 4

# Compare everything in one go
python dream.py --input assets/demo_vgg19.jpg --model all
```

Default layers per model: VGG16 `relu4_3`, VGG19 `relu4_4`, ResNet50 `layer4_2`, AlexNet `relu5`, GoogLeNet `inception3b/4c/4d`. Override with `--layers layer1 layer2 ...` as needed.

## Weights (local + optional Hugging Face)

- `dream.py` will auto-download missing weights from Hugging Face (`NickMystic/DeepDream-MLX`) into your local cache.
- Core `.npz` weights also live in this repo via LFS. If you want a fresh copy or a variant, optionally pull manually:

```bash
pip install huggingface_hub

# Core checkpoints
huggingface-cli download NickMystic/DeepDream-MLX googlenet_mlx.npz --local-dir .
huggingface-cli download NickMystic/DeepDream-MLX vgg16_mlx.npz --local-dir .
huggingface-cli download NickMystic/DeepDream-MLX resnet50_mlx.npz --local-dir .

# Optional variants
huggingface-cli download NickMystic/DeepDream-MLX googlenet_mlx_bf16.npz --local-dir .
huggingface-cli download NickMystic/DeepDream-MLX resnet50_places365_mlx.npz --local-dir .
huggingface-cli download NickMystic/DeepDream-MLX alexnet_places365_mlx.npz --local-dir .
```

Programmatic fetch:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="NickMystic/DeepDream-MLX", filename="googlenet_mlx.npz")
print(path)  # local cache path to pass into --weights
```

## Zoom Video Loop

```bash
python dream_video.py --input assets/example_googlenet.jpg --output_dir frames \
  --model googlenet --layers inception4c --frames 120 --zoom_factor 1.05

# Assemble video (requires ffmpeg)
ffmpeg -framerate 15 -i frames/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p dream_zoom.mp4
```

## Convert or Add Checkpoints

```bash
# Convert anything already in toConvert/
python convert.py --scan toConvert/

# Download common Torch7/PyTorch models and convert automatically
python convert.py --download all
```

All large `.npz` remain in Git LFS; keep `toConvert/` free of raw blobs before publishing.

## Train / Fine-Tune

Hugging Face classification datasets with `image` + `label` columns are supported out of the box. Default example: the sketchy ImageNet variant for a weird baseline.

```bash
# Sketchy ImageNet -> fine-tune GoogLeNet, warm up fc/aux then unfreeze
python train_dream.py --hf-dataset imagenet-sketch --epochs 8 --fc-warmup-epochs 2 \
  --batch-size 24 --lr 1e-3 --export-dream exports/googlenet_sketch.npz
```

Swap `--hf-dataset` to any other HF dataset in the same format; `--val-split` can be added for validation. The trainer computes mean/std and an optional color decorrelation matrix and stores them alongside checkpoints and the dream-ready export.

Local imagefolder + aspect preserved:
```bash
python train_dream.py --hf-dataset imagefolder --data-dir dataset/evangelion-ui \
  --train-split train --val-split val --preserve-aspect
```

### Reusing this repo from a sibling app

If you want the parent DeepDream UI (one directory up) to call directly into these MLX helpers instead of maintaining duplicated scripts:

1. Install this project in editable mode so Python can import it anywhere: `pip install -e /Users/mystic/Programs/deepdream/deepdream-mlx-models`.
2. Inside the parent app, import the shared functions/classes instead of copying code. Examples:
   - Dreaming: `from dream import run_dream_for_model, load_image` and call `run_dream_for_model(model_name, args, load_image(path))`.
   - Training: `from train_dream import main as train_main` to reuse the CLI entry point or `from mlx_googlenet import GoogLeNetTrain` if you need the modules directly.
3. Share checkpoints/exports by pointing both projects to the same folders (e.g. `exports/`, `checkpoints_*`). When running from the parent app, pass the shared absolute paths so you never duplicate weights.
4. Keep `dream_mlx_app.py` thin: parse app-specific flags there, then forward them to the imported helpers from this repo. This ensures new model additions (like Inception V3 / MobileNet V3) automatically become available everywhere.

Following that pattern means this repo stays the single source of truth for dreaming/training logic while the parent UI only worries about UX.

## Benchmark & Quantize

```bash
python benchmark.py
python quantize_experiment.py --model googlenet
```

## License

Apache-2.0 (see `LICENSE`).
