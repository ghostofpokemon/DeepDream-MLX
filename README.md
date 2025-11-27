---
license: other
library_name: mlx
pipeline_tag: feature-extraction
language: en
base_model: torchvision/imagenet-1k
tags:
- deepdream
- mlx
- computer-vision
- googlenet
- vgg16
- vgg19
- feature-extraction
---

# 🔮 DeepDream MLX

**PyTorch → MLX ports of classic DeepDream models for Apple Silicon.**

Efficient, standalone implementations of VGG16, VGG19, and GoogleNet (InceptionV1) tailored for Mac.

---

## ⚡ Quick Start

The fastest way to start dreaming.

**1. Setup Environment (Optional)**
```bash
conda create -n deepdream python=3.11
conda activate deepdream
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Dream**
```bash
python deepdream.py flower.jpg
```
*Running this command uses **VGG16** with standard settings to generate a high-quality dream.*

---

## 🧠 Models & Features

We provide lightweight, standalone MLX implementations of the classic computer vision architectures.

| Model | Weights File | Description |
| :--- | :--- | :--- |
| **VGG16** | `vgg16_mlx.npz` | **(Default)** The gold standard for DeepDream aesthetics. Rich textures and object parts. |
| **VGG19** | `vgg19_mlx.npz` | Deeper variant of VGG. Similar aesthetics with slightly different feature complexity. |
| **GoogleNet** | `googlenet_mlx.npz` | (Inception V1) The original DeepDream model. Trippy eyes, animals, and geometric patterns. |

### Switch Models
To use a different model, simply pass the `--model` flag:

```bash
# Use GoogleNet (Inception)
python deepdream.py flower.jpg --model googlenet

# Use VGG19
python deepdream.py flower.jpg --model vgg19
```

### Advanced Control
Customize your dreams with additional flags:

```bash
python deepdream.py flower.jpg \
  --model vgg16 \
  --layers relu4_3 relu5_2 \
  --steps 20 \
  --width 1024 \
  --pyramid_size 6
```

*   `--layers`: Choose specific layers to maximize (e.g., `relu4_3` for VGG, `inception4c` for GoogleNet).
*   `--steps`: Gradient ascent steps per scale (default: 10).
*   `--width`: Resize input image to this width (default: 600).
*   `--pyramid_size`: Number of scales to dream on (default: 4).

---

## 🛠️ Technical Details

**Conversion Process**
These models were ported from `torchvision` pre-trained weights (ImageNet).
1.  **Export**: Weights extracted from PyTorch `state_dict`.
2.  **Convert**: Transposed Convolution weights from `[Out, In, H, W]` to MLX's `[Out, H, W, In]`.
3.  **Run**: Native MLX implementations for maximum performance on M1/M2/M3 chips.

**Repository Structure**
*   `deepdream.py`: Main script for generating images.
*   `mlx_*.py`: Model architecture definitions.
*   `*.npz`: Pre-trained weights.
*   `inference.py`: Minimal script for forward-pass inference (no dreaming).

## 📜 License

Weights are derived from `torchvision` and are subject to their respective original licenses (BSD 3-Clause). Code is provided under the MIT license.