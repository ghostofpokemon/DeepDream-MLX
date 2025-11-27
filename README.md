---
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

## ⚡️ Instant Gratification

Don't think. Just dream.

```bash
# 1. Install
pip install mlx numpy pillow scipy

# 2. Run (VGG16 Default)
python dream.py --input love.jpg

# 3. Run (All Models)
python dream.py --input love.jpg --model all
```

## 🔮 The Lineage

VGG and GoogLeNet: Cousins from the 2012 Big Bang. One went **Deep**, the other went **Wide**. We ported them all.

```text
╔═ LeNet-5 (1998) ══════════════════════════════════════════════════════╗
║ The Grandfather (Yann LeCun)                                          ║
╚════╤══════════════════════════════════════════════════════════════════╝
     │
     ▼
╔═ AlexNet (2012) ══════════════════════════════════════════════════════╗
║ The Big Bang. The first GPU craze.                                    ║
╚════╤══════════════════════════════════════════════════════════════════╝
     │
     ╠══════════════════════════════════════════════════════╦═╗
     ║                                                      ║ ║
╔════╩══════════════════════╗              ╔════════════════╩═══════════╗
║     THE OXFORD BRANCH     ║              ║     THE GOOGLE BRANCH      ║
║  (Philosophy: "Go Deeper")║              ║  (Philosophy: "Go Wider")  ║
╚════╤══════════════════════╝              ╚════════════════╤═══════════╝
     │                                                      │
╔════╩══════════════════════╗              ╔════════════════╩═══════════╗
║ VGG (Visual Geometry Grp) ║              ║   Inception (GoogLeNet)    ║
╚════╤══════════════════════╝              ╚════════════════╤═══════════╝
     │                                                      │
     ╠════════════════╦═╗                                   │
     ║                ║ ║                                   │
╔════╩══════╗   ╔═════╩═════╗              ╔════════════════╩═══════════╗
║   VGG16   ║   ║   VGG19   ║              ║        Inception V1        ║
║ (Painter) ║   ║ (Stylist) ║              ║      (Hallucinator)        ║
╚════╤══════╝   ╚═════╤═════╝              ╚════════════════╤═══════════╝
     │                │                                     │
     │                │                                     │
     ▼                ▼                                     ▼
vgg16_mlx.npz    vgg19_mlx.npz                      googlenet_mlx.npz
  (Ported)         (Ported)                            (Ported)
```

## 🧠 The Models

*   **VGG16:** *The Painter.* Rich textures, thick brushstrokes.
*   **GoogLeNet:** *The Hallucination.* Eyes, animals, geometry. The classic.
*   **VGG19:** *The Stylist.* Complex, layered patterns.
*   **ResNet50:** *The Modernist.* Sharp, deep structures.

## 🧪 Recipes

Copy-paste these to get the exact looks from the header.

### 1. The Classic Trip (GoogLeNet)
*Multi-scale hallucination targeting `inception3a`, `4e`, and `5b`.*

```bash
python dream.py --input love.jpg \
    --model googlenet \
    --steps 22 \
    --lr 0.061 \
    --octaves 4 \
    --scale 1.8 \
    --jitter 26 \
    --smoothing 0.08 \
    --layers inception3a inception4e inception5b
```

### 2. The Deep Texture (VGG16)
*Rich artistic textures targeting `relu4_2`.*

```bash
python dream.py --input love.jpg \
    --model vgg16 \
    --steps 24 \
    --lr 0.07 \
    --octaves 4 \
    --scale 1.8 \
    --jitter 36 \
    --smoothing 0.19 \
    --layers relu4_2
```

### 3. The Quick Study (VGG19)
*Aggressive, shallow run on `relu5_2`.*

```bash
python dream.py --input love.jpg \
    --model vgg19 \
    --steps 14 \
    --lr 0.045 \
    --octaves 2 \
    --scale 1.5 \
    --jitter 27 \
    --smoothing 0.41 \
    --layers relu5_2
```

## 💾 Weight Conversion

We didn't just wrap existing libs. We wrote custom exporters (`export_*.py`) to rip weights from standard PyTorch/Torchvision archives and serialize them into optimized MLX `.npz` arrays.

This unlocks the classic Caffe-era architectures for the Apple Unified Memory architecture. No bloat, just tensors.

---
*NickMystic*