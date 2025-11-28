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

```bash
python3 -m venv venv
source venv/bin/activate
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

## Download Weights from Hugging Face

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

## Benchmark & Quantize

```bash
python benchmark.py
python quantize_experiment.py --model googlenet
```

## License

Apache-2.0 (see `LICENSE`).
