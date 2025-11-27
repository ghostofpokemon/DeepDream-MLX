---
license: mit
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

## Quick Start

```bash
# Needs typical scientific stack + mlx
pip install mlx numpy pillow scipy

# Dream
python dream.py --input love.jpg --model vgg16
```

## The Models

We support the heavy hitters. Weights are converted and ready.

*   **VGG16:** The Painter. Rich textures, thick brushstrokes.
*   **GoogLeNet (InceptionV1):** The Hallucination. Eyes, animals, geometry.
*   **ResNet50:** The Modernist. Sharp, deep structures.

## Weight Conversion

We didn't just wrap existing libs. We wrote custom exporters (`export_*.py`) to rip weights from standard PyTorch/Torchvision archives and serialize them into optimized MLX `.npz` arrays. 

This unlocks the classic Caffe-era architectures for the Apple Unified Memory architecture. No bloat, just tensors.

## Advanced

Everything is tunable.

```bash
python dream.py \
    --input assets/love.jpg \
    --model googlenet \
    --steps 20 \
    --jitter 32 \
    --pyramid_size 6
```

## File Structure

*   `dream.py`: The engine. Compiled graph execution.
*   `mlx_*.py`: Model definitions ported to native MLX.
*   `*.npz`: The weights (ported by us).
*   `export_*.py`: The bridge scripts that brought these models here.

---
*NickMystic*