import argparse
import os
import time
import random
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
from mlx_mobilenet import MobileNetV3Small_Defined
from mlx_inception_v3 import InceptionV3

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


def smooth_gradients(grad, sigma):
    """Cascade 3 Gaussian blurs (sigma multipliers 0.5/1/2) like deepdream.py."""
    g_np = np.array(grad)
    sigmas = [sigma * 0.5, sigma * 1.0, sigma * 2.0]
    smoothed = []
    for s in sigmas:
        smoothed.append(
            nd.gaussian_filter(g_np, sigma=(0, s, s, 0), mode="reflect")
        )
    g_np = sum(smoothed) / len(smoothed)
    return mx.array(g_np, dtype=grad.dtype)


def get_pyramid_shapes(base_shape, num_octaves, scale):
    h, w = base_shape
    shapes = []
    for level in range(num_octaves):
        exponent = level - num_octaves + 1
        nh = max(1, int(round(h * (scale**exponent))))
        nw = max(1, int(round(w * (scale**exponent))))
        
        # Ensure minimum dimension to avoid pooling errors
        # GoogLeNet/VGG need at least ~16-32px for deep layers
        if nh < 32 or nw < 32:
            continue
            
        shapes.append((nh, nw))
    
    # Ensure the original size is always included as the last step if the loop logic allows,
    # but with this logic, the last step (level = num_octaves - 1) gives exponent 0 -> scale^0 = 1.
    # So the last shape IS the original shape.
    # If all shapes were filtered out, we must at least run on the original size.
    if not shapes:
        shapes.append((h, w))
        
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

        for it in range(steps):
            ox, oy = np.random.randint(-jitter, jitter + 1, 2)
            rolled = mx.roll(mx.roll(img, ox, axis=1), oy, axis=2)

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

            loss, grads = mx.value_and_grad(loss_fn)(rolled)
            
            sigma_val = ((it + 1) / steps) * 2.0 + smoothing
            g = smooth_gradients(grads, sigma=sigma_val)
            
            g = g - mx.mean(g)
            g = g / (mx.std(g) + 1e-8)
            
            rolled = rolled + lr * g
            rolled = mx.minimum(mx.maximum(rolled, LOWER_IMAGE_BOUND), UPPER_IMAGE_BOUND)
            
            img = mx.roll(mx.roll(rolled, -ox, axis=1), -oy, axis=2)

    return deprocess(img)


def get_weights_path(model_name, explicit_path=None):
    if explicit_path:
        return explicit_path
        
    # Search locations: current dir, models/, deepdream-mlx-models/
    prefixes = ["", "models/", "deepdream-mlx-models/"]
    suffixes = ["_mlx_int8.npz", "_mlx_bf16.npz", "_mlx.npz"]
    
    for prefix in prefixes:
        for suffix in suffixes:
            path = f"{prefix}{model_name}{suffix}"
            if os.path.exists(path):
                return path

    # Fallback default
    return f"models/{model_name}_mlx.npz"


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

    if args.random:
        print("--- Randomizing Parameters ---")
        current_octaves = random.randint(3, 8)
        current_scale = round(random.uniform(1.3, 1.6), 2)
        current_steps = random.randint(10, 40)
        current_lr = round(random.uniform(0.04, 0.12), 3)
        current_jitter = random.randint(16, 64)
        current_smoothing = round(random.uniform(0.1, 0.8), 2)
        print(f"Randomized: octaves={current_octaves}, scale={current_scale}, steps={current_steps}, lr={current_lr}, jitter={current_jitter}, smoothing={current_smoothing}")

    if model_name == "vgg16":
        model = VGG16()
        weights = get_weights_path("vgg16", args.weights)
        default_layers = ["relu4_3"]
        # ... (existing VGG16 setup) ...
        if args.random:
             possible_layers = ["relu3_3", "relu4_1", "relu4_2", "relu4_3", "relu5_1", "relu5_2", "relu5_3"]
             current_layers = random.sample(possible_layers, k=random.randint(1, 2))
             print(f"Randomized Layers: {current_layers}")
        elif args.preset:
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
        if args.random:
             possible_layers = ["relu3_3", "relu4_1", "relu4_2", "relu4_3", "relu4_4", "relu5_1", "relu5_2", "relu5_3", "relu5_4"]
             current_layers = random.sample(possible_layers, k=random.randint(1, 2))
             print(f"Randomized Layers: {current_layers}")
        elif args.preset and args.preset in PRESETS:
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
        if args.random:
             # ResNet50 layers
             possible_layers = [f"layer{i}_{j}" for i in range(1, 5) for j in range(3)] + ["layer3_3", "layer3_4", "layer3_5"]
             # Simple filter for valid ones based on standard ResNet50 blocks [3, 4, 6, 3]
             # layer1: 0-2, layer2: 0-3, layer3: 0-5, layer4: 0-2
             valid_resnet = []
             for l in range(1, 5):
                 valid_resnet.append(f"layer{l}")
             for j in range(3): valid_resnet.append(f"layer1_{j}")
             for j in range(4): valid_resnet.append(f"layer2_{j}")
             for j in range(6): valid_resnet.append(f"layer3_{j}")
             for j in range(3): valid_resnet.append(f"layer4_{j}")
             
             current_layers = random.sample(valid_resnet, k=random.randint(1, 2))
             print(f"Randomized Layers: {current_layers}")

    elif model_name == "googlenet":
        model = GoogLeNet()
        weights = get_weights_path("googlenet", args.weights)
        default_layers = ["inception3b", "inception4c", "inception4d"]
        if args.random:
             possible_layers = ["inception3a", "inception3b", "inception4a", "inception4b", "inception4c", "inception4d", "inception4e", "inception5a", "inception5b"]
             current_layers = random.sample(possible_layers, k=random.randint(1, 3))
             print(f"Randomized Layers: {current_layers}")

    elif model_name == "mobilenet":
        model = MobileNetV3Small_Defined()
        weights = get_weights_path("MobileNetV3", args.weights) # MobileNetV3_mlx.npz
        default_layers = ["layer4", "layer9"]
        if args.random:
             possible_layers = [f"layer{i}" for i in range(13)]
             current_layers = random.sample(possible_layers, k=random.randint(1, 3))
             print(f"Randomized Layers: {current_layers}")

    elif model_name == "inception_v3":
        model = InceptionV3()
        weights = get_weights_path("inception_v3", args.weights)
        default_layers = ["Mixed_5b", "Mixed_6b", "Mixed_7b"]
        if args.random:
             possible_layers = ["Conv2d_2b_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3", 
                                "Mixed_5b", "Mixed_5c", "Mixed_5d", 
                                "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e", 
                                "Mixed_7a", "Mixed_7b", "Mixed_7c"]
             current_layers = random.sample(possible_layers, k=random.randint(1, 3))
             print(f"Randomized Layers: {current_layers}")

    # Validation: Check if current_layers are valid for the model to prevent KeyError
    # If invalid, revert to default_layers
    layers_to_check = current_layers or default_layers
    is_valid = True
    if model_name == "googlenet" and any("relu" in l or "layer" in l for l in layers_to_check): is_valid = False
    if model_name.startswith("vgg") and any("inception" in l or "layer" in l for l in layers_to_check): is_valid = False
    if model_name == "resnet50" and any("inception" in l or "relu" in l for l in layers_to_check): is_valid = False
    if model_name == "mobilenet" and any("inception" in l or "relu" in l for l in layers_to_check): is_valid = False
    if model_name == "inception_v3" and any("layer" in l or "relu" in l for l in layers_to_check): is_valid = False
    
    if not is_valid:
        print(f"Warning: Layers {layers_to_check} seem invalid for {model_name}. Reverting to defaults: {default_layers}")
        current_layers = default_layers

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
        choices=["vgg16", "vgg19", "googlenet", "resnet50", "mobilenet", "inception_v3", "all"],
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

    p.add_argument("--random", action="store_true", help="Randomize parameters (layers, octaves, scale, etc.)")

    return p.parse_args()


def main():
    args = parse_args()
    img_np = load_image(args.input, args.width)

    if args.model == "all":
        models = ["vgg16", "vgg19", "googlenet", "resnet50"]
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
