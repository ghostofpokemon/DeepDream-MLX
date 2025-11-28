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

**Status:** Fast. Native. 
**Vibe:** 2015 Hallucinations // 2025 Silicon.

DeepDream-MLX brings the classic psychedelic computer vision algorithm to modern Apple Silicon, running natively on the GPU via the [MLX](https://github.com/ml-explore/mlx) framework. No Caffe, no slow conversion layers—just pure tensor operations.

## ⚡️ Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Dream (Default VGG16)
python dream.py --input assets/demo_googlenet.jpg

# 3. Explore Models
python dream.py --input assets/demo_googlenet.jpg --model googlenet --layers inception4c
```

## 🔮 The Evolution of Vision

We support the classic ancestors of modern Computer Vision.

```text
   TIMELINE       MODEL            PARAMS      PHILOSOPHY
   ──────────────────────────────────────────────────────────
   1998           LeNet-5          60K         "Digits."
     │
     ▼
   2012           AlexNet          60M         "Deep."
     │            (Available)
     │
     ├────────────┐
     ▼            ▼
   2014         2014
   VGG16        GoogLeNet          7M          "Wide & Efficient."
   138M         (Inception)
   "Deeper."
     │
     ▼
   2015
   ResNet50       25M              "Identity & Residuals."
   (Modern Standard)
```

## 🧪 Recipes

### 1. The Classic (GoogLeNet)
The original DeepDream look. Eyes, slugs, and pagodas.
```bash
python dream.py --input img.jpg --model googlenet --layers inception4c --octaves 4 --scale 1.4
```

### 2. The Painter (VGG16)
Dense, rich textures. Great for artistic style transfer-like effects.
```bash
python dream.py --input img.jpg --model vgg16 --layers relu4_3 --steps 20
```

### 3. The Modernist (ResNet50)
Sharp, geometric, and sometimes abstract architectural hallucinations.
```bash
python dream.py --input img.jpg --model resnet50 --layers layer4_2
```

## 🛠 Advanced Usage

### Converting Models
We include a universal converter that ingests standard PyTorch (`.pth`) and legacy Torch7 (`.t7`) models, optimizing them into MLX format (`float16` by default).

```bash
# Convert a local file
python convert.py --scan path/to/models

# Download & Convert Places365 (AlexNet, ResNet, etc.)
python convert.py --download all
```

### Benchmarking
Verify performance on your machine.
```bash
python benchmark.py
```

## ⚖️ Performance (M2 Max)

| Framework | Model | Precision | Speed |
| :--- | :--- | :--- | :--- |
| **MLX** | GoogLeNet | **float16** | **~3.6s** |
| PyTorch (MPS) | GoogLeNet | float32 | ~4.5s |

*Benchmarks run at 400px width, 10 iterations.*

---
*Built for the dreamers.*