#!/usr/bin/env python3
import argparse
import os
import time
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import scipy.ndimage as nd
from PIL import Image

from mlx_googlenet import GoogLeNet
from mlx_resnet50 import ResNet50
from mlx_vgg16 import VGG16
from mlx_vgg19 import VGG19
from mlx_alexnet import AlexNet

IMAGENET_MEAN = mx.array([0.485, 0.456, 0.406])
IMAGENET_STD = mx.array([0.229, 0.224, 0.225])
LOWER_IMAGE_BOUND = (-IMAGENET_MEAN / IMAGENET_STD).reshape(1, 1, 1, 3)
UPPER_IMAGE_BOUND = ((1.0 - IMAGENET_MEAN) / IMAGENET_STD).reshape(1, 1, 1, 3)


def load_image(path, target_width=None):
    img = Image.open(path).convert("RGB")
    if target_width:
        w, h = img.size
        scale = target_width / w
        new_h = int(h * scale)
        img = img.resize((target_width, new_h), Image.LANCZOS)
    return np.array(img)


def preprocess(img_np):
    x = mx.array(img_np, dtype=mx.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = x[None, ...]  # NHWC
    return x


def deprocess(x):
    x = x[0]
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = mx.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(mx.uint8)
    return np.array(x)


def resize_bilinear(x, new_h, new_w):
    b, h, w, c = x.shape
    out = mx.zeros((b, new_h, new_w, c))
    for bi in range(b):
        for ci in range(c):
            out[bi, :, :, ci] = mx.array(
                nd.zoom(np.array(x[bi, :, :, ci]), zoom=(new_h / h, new_w / w), order=1)
            )
    return out


def gaussian_kernel(sigma, truncate=4.0, fixed_radius=None):
    """Generates a 1D Gaussian kernel."""
    if fixed_radius is not None:
        radius = fixed_radius
    else:
        radius = int(truncate * sigma + 0.5)

    x = mx.arange(-radius, radius + 1)
    kernel = mx.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur_2d(x, sigma, fixed_radius=None):
    """Applies Gaussian blur using separable 1D convolutions in MLX."""
    kernel = gaussian_kernel(sigma, fixed_radius=fixed_radius)
    kernel = kernel.astype(x.dtype)
    k_size = kernel.shape[0]
    C = x.shape[-1]

    k_x = kernel.reshape(1, 1, k_size, 1)
    k_x = mx.repeat(k_x, C, axis=0)
    k_y = kernel.reshape(1, k_size, 1, 1)
    k_y = mx.repeat(k_y, C, axis=0)

    pad = k_size // 2

    x = mx.conv2d(x, k_x, stride=1, padding=(0, pad), groups=C)
    x = mx.conv2d(x, k_y, stride=1, padding=(pad, 0), groups=C)
    return x


def smooth_gradients(grad, sigma, fixed_radius=None):
    """Cascade 3 Gaussian blurs (sigma multipliers 0.5/1/2) using native MLX ops."""
    sigmas = [sigma * 0.5, sigma * 1.0, sigma * 2.0]
    smoothed = []
    for s in sigmas:
        smoothed.append(gaussian_blur_2d(grad, s, fixed_radius=fixed_radius))

    g_total = smoothed[0]
    for i in range(1, len(smoothed)):
        g_total = g_total + smoothed[i]
    return g_total / len(smoothed)


def get_pyramid_shapes(base_shape, num_octaves, scale):
    h, w = base_shape
    shapes = []
    for level in range(num_octaves):
        exponent = level - num_octaves + 1
        nh = max(1, int(round(h * (scale**exponent))))
        nw = max(1, int(round(w * (scale**exponent))))
        shapes.append((nh, nw))
    return shapes


def deepdream(
    model,
    img_np,
    layers,
    steps,
    lr,
    num_octaves,
    scale,
    jitter=32,
    smoothing=0.5,
    guide_img_np=None,
):
    img = preprocess(img_np)
    base_h, base_w = img.shape[1:3]
    pyramid_shapes = get_pyramid_shapes((base_h, base_w), num_octaves, scale)

    for level, (nh, nw) in enumerate(pyramid_shapes):
        img = resize_bilinear(img, nh, nw)

        guide_features = {}
        if guide_img_np is not None:
            guide_resized = resize_bilinear(preprocess(guide_img_np), nh, nw)
            _, guide_features = model.forward_with_endpoints(guide_resized)

        def loss_fn(x):
            endpoints = model.forward_with_endpoints(x)[1]
            loss = mx.zeros(())
            for name in layers:
                act = endpoints[name]
                if guide_img_np is not None:
                    guide_act = guide_features[name]
                    loss = loss + mx.mean(act * guide_act)
                else:
                    loss = loss + mx.mean(act * act)
            return loss / len(layers)

        # Calculate max radius needed for static compilation
        max_effective_sigma = 2.0 * (2.0 + smoothing)
        fixed_radius = int(4.0 * max_effective_sigma + 0.5)

        @mx.compile
        def update_step(x, sigma):
            loss, grads = mx.value_and_grad(loss_fn)(x)
            g = smooth_gradients(grads, sigma, fixed_radius=fixed_radius)
            g = g - mx.mean(g)
            g = g / (mx.std(g) + 1e-8)
            x = x + lr * g
            x = mx.minimum(mx.maximum(x, LOWER_IMAGE_BOUND), UPPER_IMAGE_BOUND)
            return x, loss

        for it in range(steps):
            ox, oy = np.random.randint(-jitter, jitter + 1, 2)
            rolled = mx.roll(mx.roll(img, ox, axis=1), oy, axis=2)

            sigma_val = ((it + 1) / steps) * 2.0 + smoothing

            rolled, loss = update_step(rolled, mx.array(sigma_val))

            img = mx.roll(mx.roll(rolled, -ox, axis=1), -oy, axis=2)

    return deprocess(img)


def get_weights_path(model_name, explicit_path=None):


    if explicit_path:


        return explicit_path


        


    # 1. Try int8 (Maximum Efficiency / Smallest)


    int8_path = f"{model_name}_mlx_int8.npz"


    if os.path.exists(int8_path):


        return int8_path





    # 2. Try bf16 (Standard Efficient)


    bf16_path = f"{model_name}_mlx_bf16.npz"


    if os.path.exists(bf16_path):


        return bf16_path


        


    # 3. Try standard float32


    fp32_path = f"{model_name}_mlx.npz"


    if os.path.exists(fp32_path):


        return fp32_path


        


    return int8_path # Return preferred default for error message context


def run_dream_for_model(model_name, args, img_np):
    print(f"--- Running DeepDream with {model_name} ---")

    # ... (PRESETS dict remains here) ...
    # Notebook presets
    PRESETS = {
        "nb14": {
            "layers": ["relu3_3"],
            "steps": 10,
            "lr": 0.06,
            "octaves": 6,
            "scale": 1.4,
            "jitter": 32,
            "smoothing": 0.5,
        },
        "nb20": {
            "layers": ["relu4_2"],
            "steps": 10,
            "lr": 0.06,
            "octaves": 6,
            "scale": 1.4,
            "jitter": 32,
            "smoothing": 0.5,
        },
        "nb28": {
            "layers": ["relu5_3"],
            "steps": 10,
            "lr": 0.06,
            "octaves": 6,
            "scale": 1.4,
            "jitter": 32,
            "smoothing": 0.5,
        },
    }

    # Set up model, weights, and defaults
    current_layers = args.layers
    current_steps = args.steps
    current_lr = args.lr
    current_octaves = args.octaves
    current_scale = args.scale
    current_jitter = args.jitter
    current_smoothing = args.smoothing

    if model_name == "vgg16":
        model = VGG16()
        weights = get_weights_path("vgg16", args.weights)
        default_layers = ["relu4_3"]
        if args.preset:
            p = PRESETS[args.preset]
            # Apply preset overrides
            current_layers = p["layers"]
            current_steps = p["steps"]
            current_lr = p["lr"]
            current_octaves = p["octaves"]
            current_scale = p["scale"]
            current_jitter = p["jitter"]
            current_smoothing = p["smoothing"]

    elif model_name == "vgg19":
        model = VGG19()
        weights = get_weights_path("vgg19", args.weights)
        default_layers = ["relu4_4"]
        if args.preset and args.preset in PRESETS:
            p = PRESETS[args.preset]
            current_layers = p["layers"]
            current_steps = p["steps"]
            current_lr = p["lr"]
            current_octaves = p["octaves"]
            current_scale = p["scale"]
            current_jitter = p["jitter"]
            current_smoothing = p["smoothing"]

    elif model_name == "resnet50":
        model = ResNet50()
        weights = get_weights_path("resnet50", args.weights)
        default_layers = ["layer4_2"]

    elif model_name == "alexnet":
        model = AlexNet()
        weights = get_weights_path("alexnet", args.weights)
        default_layers = ["relu5"]

    else:  # googlenet
        model = GoogLeNet()
        weights = get_weights_path("googlenet", args.weights)
        default_layers = ["inception3b", "inception4c", "inception4d"]

    if not os.path.exists(weights):
        print(f"Error: Weights NPZ not found: {weights}. Skipping {model_name}.")
        return

    print(f"Loading weights from: {weights}")
    model.load_npz(weights)

    guide_img_np = None
    if args.guide:
        print(f"Using guide image: {args.guide}")
        guide_img_np = load_image(args.guide, args.width)

    start_time = time.time()
    start_timestamp = datetime.now()

    dreamed = deepdream(
        model,
        img_np,
        layers=current_layers or default_layers,
        steps=current_steps,
        lr=current_lr,
        num_octaves=current_octaves,
        scale=current_scale,
        jitter=current_jitter,
        smoothing=current_smoothing,
        guide_img_np=guide_img_np,
    )

    end_time = time.time()
    elapsed = end_time - start_time

    if args.output:
        out = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        formatted_time = f"{elapsed:.2f}s"
        formatted_date = start_timestamp.strftime("%m%d")
        formatted_timestamp = start_timestamp.strftime("%H%M%S")
        out = f"{base_name}_dream_{model_name}_{formatted_time}_{formatted_date}_{formatted_timestamp}.jpg"

    Image.fromarray(dreamed).save(out)
    print(f"Saved {out}\n")


def parse_args():
    p = argparse.ArgumentParser(description="DeepDream with MLX (Compiled)")
    p.add_argument("--input", required=True, help="Input image path")
    p.add_argument("--output", help="Output image path (optional)")
    p.add_argument("--guide", help="Guide image for guided dreaming")

    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Resize input to width (maintains aspect ratio)",
    )
    p.add_argument(
        "--img_width", type=int, help="Alias for --width", dest="width"
    )  # Alias

    p.add_argument(
        "--model",
        choices=["vgg16", "vgg19", "googlenet", "resnet50", "alexnet", "all"],
        default="vgg16",
        help="Model to use. 'all' runs all models.",
    )
    p.add_argument("--preset", choices=["nb14", "nb20", "nb28"], help="VGG16 presets")

    p.add_argument("--layers", nargs="+", help="Layers to maximize")
    p.add_argument(
        "--steps", type=int, default=10, help="Gradient ascent steps per octave"
    )
    p.add_argument("--lr", type=float, default=0.09, help="Learning rate (step size)")

    p.add_argument("--octaves", type=int, default=4, help="Number of image octaves")
    p.add_argument(
        "--pyramid_size", type=int, dest="octaves", help="Alias for --octaves"
    )  # Alias

    p.add_argument("--scale", type=float, default=1.8, help="Octave scale factor")
    p.add_argument(
        "--pyramid_ratio", type=float, dest="scale", help="Alias for --scale"
    )  # Alias
    p.add_argument(
        "--octave_scale", type=float, dest="scale", help="Alias for --scale"
    )  # Alias

    p.add_argument("--jitter", type=int, default=32, help="Jitter amount (pixels)")

    p.add_argument(
        "--smoothing", type=float, default=0.5, help="Gradient smoothing strength"
    )
    p.add_argument(
        "--smoothing_coefficient",
        type=float,
        dest="smoothing",
        help="Alias for --smoothing",
    )  # Alias

    p.add_argument("--weights", help="Custom weights path")

    return p.parse_args()


def main():
    args = parse_args()
    img_np = load_image(args.input, args.width)

    if args.model == "all":
        models = ["vgg16", "vgg19", "googlenet", "resnet50", "alexnet"]
        if args.output:
            print(
                "Warning: --output argument ignored because --model='all' was selected."
            )
            args.output = None
        for m in models:
            run_dream_for_model(m, args, img_np)
    else:
        run_dream_for_model(args.model, args, img_np)


if __name__ == "__main__":
    main()
