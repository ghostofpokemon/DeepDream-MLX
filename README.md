--- license: mit
tags:
- mlx
- computer-vision
- art
- generative
pipeline_tag: image-to-image
---

# DeepDream-MLX

Native, hardware-accelerated DeepDream for Apple Silicon.
We ripped out the slow parts and baked the compute graph directly into the GPU.

**Status:** Fast. 
**Vibe:** 2015 Aesthetics // 2025 Hardware.

![DeepDream Header](assets/deepdream_header.jpg)

## The Lineage

VGG and GoogLeNet are cousins, evolving from AlexNet (2012) but taking different paths: one went **Deep**, the other went **Wide**.

```text
THE CONVOLUTIONAL ANCESTRY
==========================

[ LeNet-5 (1998) ]  <-- The Grandfather (Yann LeCun)
       \
       v
[ AlexNet (2012) ]  <-- The Big Bang. The first GPU craze.
       \
       ├───────────────────────────────────────────────┐
       │                                               │
[ THE OXFORD BRANCH ]                           [ THE GOOGLE BRANCH ]
(Philosophy: "Go Deeper")                       (Philosophy: "Go Wider")
       │                                               │
       │                                               │
[ VGG (Visual Geometry Group) ]                 [ Inception (GoogLeNet) ]
       │                                               │
       ├─ [ VGG16 ]                                    └─ [ Inception V1 ]
       │   │                                               │
       │   └─ vgg16_mlx.npz (Our Port)                     ├─ bvlc_googlenet.caffemodel
       │                                                   │  (Berkeley's Caffe Ref.)
       └─ [ VGG19 ]                                        │
           │                                               └─ googlenet_mlx.npz (Our Port)
           └─ vgg19_mlx.npz (Our Port)
```

## Quick Start

```bash
# Needs typical scientific stack + mlx
pip install mlx numpy pillow scipy

# Dream with default VGG16
python dream.py --input love.jpg
```

## The Models

*   **VGG16:** The Painter. Rich textures, thick brushstrokes.
*   **GoogLeNet (InceptionV1):** The Hallucination. Eyes, animals, geometry.
*   **VGG19:** The Stylist. Complex, layered patterns.
*   **ResNet50:** The Modernist. Sharp, deep structures.

### Advanced Usage

Customize the dream parameters:

```bash
python dream.py --input love.jpg \
    --output dreamy_love.jpg \
    --model googlenet \
    --width 800 \
    --layers inception4c \
    --steps 20 \
    --octaves 6 \
    --jitter 32
```

### Arguments

*   `--input`: Path to the input image (required).
*   `--output`: Path to save the output image (optional, auto-generated if omitted).
*   `--model`: Model architecture to use: `vgg16` (default), `vgg19`, `googlenet`, `resnet50`.
*   `--width`: Resize input image to this width (maintains aspect ratio). Default: uses original size.
*   `--layers`: Specific layers to maximize activations for.
*   `--steps`: Number of gradient ascent steps per octave (default: 10).
*   `--lr`: Learning rate (step size) (default: 0.09).
*   `--octaves`: Number of scales in the image pyramid (default: 4).
*   `--scale`: Scale factor between pyramid levels (default: 1.8).
*   `--jitter`: Amount of random image shift (jitter) to apply (default: 32).
*   `--smoothing`: Gradient smoothing strength (default: 0.5).

## Recipes

Here are the exact commands used to generate the header images:

### 1. GoogLeNet (The Classic)
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

### 2. VGG16 (The Painter)
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

### 3. VGG19 (The Stylist)
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

## Weight Conversion

We didn't just wrap existing libs. We wrote custom exporters (`export_*.py`) to rip weights from standard PyTorch/Torchvision archives and serialize them into optimized MLX `.npz` arrays. 

This unlocks the classic Caffe-era architectures for the Apple Unified Memory architecture. No bloat, just tensors.

---
*NickMystic*
