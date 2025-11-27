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

## ⚡️ Instant Gratification

```bash
# 1. Install Dependencies
pip install mlx numpy pillow scipy

# 2. Dream (VGG16 Default)
python dream.py --input love.jpg

# 3. Dream (All Models)
python dream.py --input love.jpg --model all
```

## 🔮 The Lineage

VGG and GoogLeNet: Cousins from the 2012 Big Bang. One went **Deep**, the other went **Wide**. We ported them all.

```text
╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                          THE CONVOLUTIONAL ANCESTRY                                                 ║
╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                     ║
║          ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓                                                                               ║
║          ┃      LeNet-5 (1998)      ┃  (The Grandfather)                                                            ║
║          ┗━━━━━━━━━━━━┳━━━━━━━━━━━━━┛                                                                               ║
║                       │                                                                                             ║
║                       ▼                                                                                             ║
║          ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓                                                                               ║
║          ┃      AlexNet (2012)      ┃  (The Ignition)                                                               ║
║          ┗━━━━━━━━━━━━┳━━━━━━━━━━━━━┛                                                                               ║
║                       │                                                                                             ║
║    ╔══════════════════╩════════════════════════════════════════════════════════════════════════════════╗            ║
║    ║                                                                                                   ║            ║
║    ▼                                              ▼                                                    ▼            ║
║                                                                                                                     ║
║ ╔══════════════════════════════════╗    ╔══════════════════════════════════╗    ╔═════════════════════════════════╗ ║
║ ║        THE OXFORD BRANCH         ║    ║        THE GOOGLE BRANCH         ║    ║     THE RESIDUAL REVOLUTION     ║ ║
║ ║      (Philosophy: "Deeper")      ║    ║      (Philosophy: "Wider")       ║    ║     (Philosophy: "Identity")    ║ ║
║ ╚═════════════════╦════════════════╝    ╚═════════════════╦════════════════╝    ╚════════════════════╦════════════╝ ║
║                   │                                       │                                          │              ║
║         ┌─────────┴─────────┐                             │                                          │              ║
║         │                   │                             │                                          │              ║
║    ┏━━━━▼━━━━┓         ┏━━━━▼━━━━┓                   ┏━━━━▼━━━━┓                                ┏━━━━▼━━━━┓         ║
║    ┃  VGG16  ┃         ┃  VGG19  ┃                   ┃Inception┃                                ┃ ResNet  ┃         ║
║    ┃         ┃         ┃         ┃                   ┃   V1    ┃                                ┃   50    ┃         ║
║    ┗━━━━┳━━━━┛         ┗━━━━┳━━━━┛                   ┗━━━━┳━━━━┛                                ┗━━━━┳━━━━┛         ║
║         │                   │                             │                                          │              ║
║    (The Painter)       (The Stylist)               (The Hallucinator)                             (The Modernist)   ║
║         │                   │                             │                                          │              ║
║         ▼                   ▼                             ▼                                          ▼              ║
║   vgg16_mlx.npz       vgg19_mlx.npz               googlenet_mlx.npz                          resnet50_mlx.npz       ║
║                                                                                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

## 🧠 The Models

*   **VGG16:** General purpose image features.
*   **GoogLeNet (InceptionV1):** The classic DeepDream model.
*   **VGG19:** Deeper VGG features.
*   **ResNet50:** Modern deep features.

## 🧪 Recipes

Copy-paste these to get the exact looks from the header.

### 1. Classic Inception Patterns (GoogLeNet)
*This setup targets various Inception layers for recognizable DeepDream shapes.*

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

### 2. Rich Textures (VGG16)
*A VGG16 run for detailed, painterly results.*

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

### 3. Layered Patterns (VGG19)
*A VGG19 run for complex, stylized outputs.*

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

### 4. Different VGG16 Vision
*Another VGG16 setting, exploring alternative features.*

```bash
python dream.py --input love.jpg \
    --model vgg16 \
    --steps 24 \
    --lr 0.069 \
    --octaves 4 \
    --scale 1.8 \
    --jitter 10 \
    --smoothing 0.41 \
    --layers relu5_1
```

### 5. Sharp Abstract Forms (ResNet50)
*Modern features from ResNet50 for distinct, edgy results.*

```bash
python dream.py --input love.jpg \
    --model resnet50 \
    --steps 22 \
    --lr 0.13 \
    --octaves 4 \
    --scale 2 \
    --jitter 83 \
    --smoothing 0.47 \
    --layers layer3_2 layer3_5
```

## 💾 Weight Conversion & Efficiency

We didn't just wrap existing libs. We wrote custom exporters (`export_*.py`) to rip weights from standard PyTorch/Torchvision archives and serialize them into optimized MLX `.npz` arrays. 

### 50% Smaller Weights (FP16)
We now support **Float16** (Half-Precision) weights by default. This cuts model size in half with zero visual loss for DeepDreaming.
*   **VGG16:** 528MB → **264MB**
*   **ResNet50:** 98MB → **49MB**

`dream.py` automatically detects and loads `_bf16.npz` files if present.

## 🔎 Where to find models?

You can convert *any* standard PyTorch model to run here.
1.  **Torchvision:** The source of our VGG/GoogLeNet/ResNet weights.
2.  **Hugging Face Hub:** Massive repo of pretrained models.
3.  **Caffe Model Zoo (Historical):** If you have `.caffemodel` files, load them into PyTorch (using tools like `load_caffe`) and then export.

## 🎓 Training & Fine-Tuning (TODO)

Want your DeepDream to see things *differently*? (e.g., dogs instead of slugs?)
You need to fine-tune the base model on a new dataset.

**Current Workflow:**
1.  Train your model in PyTorch (standard ImageNet training or custom dataset).
2.  Save the `.pth` checkpoint.
3.  Modify our `export_*.py` scripts to load your custom checkpoint.
4.  Export to `.npz`.
5.  Dream.

*A dedicated `train_dream.py` script is on the roadmap.*

---
*NickMystic*
