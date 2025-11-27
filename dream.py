
import argparse
import os
import time
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import scipy.ndimage as nd
from mlx_resnet50 import ResNet50
from PIL import Image

from mlx_googlenet import GoogLeNet
from mlx_vgg16 import VGG16
from mlx_vgg19 import VGG19

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
        # If sigma is a tracer (mx.array), this int() conversion would fail during tracing
        # So we must use fixed_radius when compiling with dynamic sigma
        radius = int(truncate * sigma + 0.5)
        
    x = mx.arange(-radius, radius + 1)
    kernel = mx.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur_2d(x, sigma, fixed_radius=None):
    """Applies Gaussian blur using separable 1D convolutions in MLX."""
    # If sigma is 0, we should return x. But inside compiled graph with dynamic sigma, 
    # we can't conditionally return based on value (creates graph branch). 
    # However, gaussian with sigma close to 0 is just a delta spike.
    # Let's relying on the kernel math: exp(-huge) -> 0. 
    # We should add a small epsilon to sigma to avoid div by zero if sigma can be 0.
    # But here sigma is always >= 0.5 * 0.5 = 0.25.
    
    # Generate kernel
    kernel = gaussian_kernel(sigma, fixed_radius=fixed_radius)
    kernel = kernel.astype(x.dtype)
    k_size = kernel.shape[0]
    
    C = x.shape[-1]
    
    # Horizontal kernel: [C, 1, k, 1]
    k_x = kernel.reshape(1, 1, k_size, 1)
    k_x = mx.repeat(k_x, C, axis=0)
    
    # Vertical kernel: [C, k, 1, 1]
    k_y = kernel.reshape(1, k_size, 1, 1)
    k_y = mx.repeat(k_y, C, axis=0)
    
    pad = k_size // 2
    
    x = mx.conv2d(x, k_x, stride=1, padding=(0, pad), groups=C)
    x = mx.conv2d(x, k_y, stride=1, padding=(pad, 0), groups=C)
    
    return x


def smooth_gradients(grad, sigma, fixed_radius=None):
    """Cascade 3 Gaussian blurs (sigma multipliers 0.5/1/2) using native MLX ops."""
    # If fixed_radius is used, we must ensure it's large enough for the LARGEST sigma (sigma * 2.0).
    # We assume the caller handled this or we just pass it through.
    
    sigmas = [sigma * 0.5, sigma * 1.0, sigma * 2.0]
    smoothed = []
    for s in sigmas:
        smoothed.append(gaussian_blur_2d(grad, s, fixed_radius=fixed_radius))
    
    g_total = smoothed[0]
    for i in range(1, len(smoothed)):
        g_total = g_total + smoothed[i]
        
    return g_total / len(smoothed)


def get_pyramid_shapes(base_shape, pyramid_size, pyramid_ratio):
    h, w = base_shape
    shapes = []
    for level in range(pyramid_size):
        exponent = level - pyramid_size + 1
        nh = max(1, int(round(h * (pyramid_ratio**exponent))))
        nw = max(1, int(round(w * (pyramid_ratio**exponent))))
        shapes.append((nh, nw))
    return shapes


def deepdream(
    model,
    img_np,
    layers,
    steps,
    lr,
    pyramid_size,
    pyramid_ratio,
    jitter=32,
    smoothing_coefficient=0.5,
    guide_img_np=None,
):
    img = preprocess(img_np)
    base_h, base_w = img.shape[1:3]
    pyramid_shapes = get_pyramid_shapes((base_h, base_w), pyramid_size, pyramid_ratio)

    for level, (nh, nw) in enumerate(pyramid_shapes):
        img = resize_bilinear(img, nh, nw)

        # Prepare guide features for this level if guide is provided
        guide_features = {}
        if guide_img_np is not None:
            guide_resized = resize_bilinear(preprocess(guide_img_np), nh, nw)
            _, guide_features = model.forward_with_endpoints(guide_resized)
        
        # Define the loss function closure
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
        # Max sigma used in loop is roughly 2.0 + smoothing_coeff (when it=steps)
        # smooth_gradients multiplies this by 2.0 again for the largest blur.
        # So max_effective_sigma = 2.0 * (2.0 + smoothing_coefficient)
        max_effective_sigma = 2.0 * (2.0 + smoothing_coefficient)
        fixed_radius = int(4.0 * max_effective_sigma + 0.5)

        # Compile the update step
        # We compile a function that takes: rolled_img, sigma (tensor)
        # And returns: updated_rolled_img
        @mx.compile
        def update_step(x, sigma):
            loss, grads = mx.value_and_grad(loss_fn)(x)
            
            # Pass fixed_radius to ensure kernel shapes are static
            # while kernel values depend on dynamic 'sigma'
            g = smooth_gradients(grads, sigma, fixed_radius=fixed_radius)
            
            # Normalize
            g = g - mx.mean(g)
            g = g / (mx.std(g) + 1e-8)
            
            # Update
            x = x + lr * g
            
            # Clip
            x = mx.minimum(mx.maximum(x, LOWER_IMAGE_BOUND), UPPER_IMAGE_BOUND)
            return x, loss

        for it in range(steps):
            # Jitter (outside compiled block)
            ox, oy = np.random.randint(-jitter, jitter + 1, 2)
            rolled = mx.roll(mx.roll(img, ox, axis=1), oy, axis=2)
            
            # Calculate sigma for this step
            sigma_val = ((it + 1) / steps) * 2.0 + smoothing_coefficient
            
            # Run compiled update with dynamic sigma (as tensor)
            rolled, loss = update_step(rolled, mx.array(sigma_val))
            
            # Un-Jitter
            img = mx.roll(mx.roll(rolled, -ox, axis=1), -oy, axis=2)
            
            # Force eval to ensure computation happens (optional but good for benchmark/progress)
            # mx.eval(img)

    return deprocess(img)





def run_dream_for_model(model_name, args, img_np):
    print(f"--- Running DeepDream with {model_name} ---")
    
    # Notebook presets (from notebookfc42c6db41.ipynb)
    PRESETS = {
        "nb14": {
            "layers": ["relu3_3"],
            "steps": 10,
            "lr": 0.06,
            "pyramid_size": 6,
            "pyramid_ratio": 1.4,
            "jitter": 32,
            "smoothing_coefficient": 0.5,
        },
        "nb20": {
            "layers": ["relu4_2"],
            "steps": 10,
            "lr": 0.06,
            "pyramid_size": 6,
            "pyramid_ratio": 1.4,
            "jitter": 32,
            "smoothing_coefficient": 0.5,
        },
        "nb28": {
            "layers": ["relu5_3"],
            "steps": 10,
            "lr": 0.06,
            "pyramid_size": 6,
            "pyramid_ratio": 1.4,
            "jitter": 32,
            "smoothing_coefficient": 0.5,
        },
    }

    # Set up model, weights, and defaults
    # We create local copies of parameters that might be overridden by presets
    current_layers = args.layers
    current_steps = args.steps
    current_lr = args.lr
    current_pyramid_size = args.octaves or args.pyramid_size
    current_pyramid_ratio = args.octave_scale or args.pyramid_ratio
    current_jitter = args.jitter
    current_smoothing = args.smoothing_coefficient

    if model_name == "vgg16":
        model = VGG16()
        weights = args.weights or "vgg16_mlx.npz"
        default_layers = ["relu4_3"]
        if args.preset:
            preset = PRESETS[args.preset]
            current_layers = preset["layers"]
            current_steps = preset["steps"]
            current_lr = preset["lr"]
            current_pyramid_size = preset["pyramid_size"]
            current_pyramid_ratio = preset["pyramid_ratio"]
            current_jitter = preset["jitter"]
            current_smoothing = preset["smoothing_coefficient"]
            
    elif model_name == "vgg19":
        model = VGG19()
        weights = args.weights or "vgg19_mlx.npz"
        default_layers = ["relu4_4"]
        if args.preset and args.preset in PRESETS:
            preset = PRESETS[args.preset]
            # Presets technically for VGG16 but we can apply them
            current_layers = preset["layers"]
            current_steps = preset["steps"]
            current_lr = preset["lr"]
            current_pyramid_size = preset["pyramid_size"]
            current_pyramid_ratio = preset["pyramid_ratio"]
            current_jitter = preset["jitter"]
            current_smoothing = preset["smoothing_coefficient"]
            
    elif model_name == "resnet50":
        model = ResNet50()
        weights = args.weights or "resnet50_mlx.npz"
        default_layers = ["layer4_2"]
        
    else: # googlenet (InceptionV1)
        model = GoogLeNet()
        weights = args.weights or "googlenet_mlx.npz"
        default_layers = ["inception3b", "inception4c", "inception4d"]

    if not os.path.exists(weights):
        print(f"Error: Weights NPZ not found: {weights}. Skipping {model_name}.")
        return

    model.load_npz(weights)

    guide_img_np = None
    if args.guide:
        print(f"Using guide image: {args.guide}")
        guide_img_np = load_image(args.guide, args.img_width)

    start_time = time.time()
    start_timestamp = datetime.now()

    dreamed = deepdream(
        model,
        img_np,
        layers=current_layers or default_layers,
        steps=current_steps,
        lr=current_lr,
        pyramid_size=current_pyramid_size,
        pyramid_ratio=current_pyramid_ratio,
        jitter=current_jitter,
        smoothing_coefficient=current_smoothing,
        guide_img_np=guide_img_np,
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    if args.output:
        # If explicit output is set, use it (but if running 'all', this overwrites)
        # If running multiple models, we should probably append model name if 'all' was used.
        # But here we just use what's passed. The caller loop handles distinct args if needed?
        # Actually, let's just append model name if output is auto-generated.
        # If user provided explicit output, they better know what they are doing or we overwrite.
        # But for 'all', we ideally want distinct names.
        # We'll handle that logic in main loop by not passing explicit output for 'all', 
        # or ignoring it.
        out = args.output
    else:
        # Elegant filename: input_dream_MODEL_ELAPSEDs_MMDD_HHMMSS.jpg
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        formatted_time = f"{elapsed:.2f}s"
        formatted_date = start_timestamp.strftime("%m%d")
        formatted_timestamp = start_timestamp.strftime("%H%M%S")
        out = f"{base_name}_dream_{model_name}_{formatted_time}_{formatted_date}_{formatted_timestamp}.jpg"

    Image.fromarray(dreamed).save(out)
    print(f"Saved {out}\n")


def parse_args():
    p = argparse.ArgumentParser(description="DeepDream with MLX (Compiled)")
    p.add_argument("--input", required=True, help="Input image")
    p.add_argument("--output", help="Output path (ignored if model='all')")
    p.add_argument("--guide", help="Guide image for guided dreaming")
    p.add_argument("--img_width", type=int, default=None)
    p.add_argument(
        "--model",
        choices=["vgg16", "vgg19", "googlenet", "resnet50", "all"],
        default="vgg16",
        help="Backbone model. 'googlenet' is InceptionV1. 'all' runs all models.",
    )
    # ... rest of args same as before ...
    p.add_argument(
        "--preset",
        choices=["nb14", "nb20", "nb28"],
        help="Notebook-inspired presets (VGG16 only)",
    )
    p.add_argument(
        "--layers",
        nargs="+",
        help="Layers to maximise",
    )
    p.add_argument(
        "--steps", type=int, default=10, help="Gradient ascent steps per scale"
    )
    p.add_argument("--lr", type=float, default=0.09)
    p.add_argument(
        "--pyramid_size",
        type=int,
        default=4,
        help="Number of pyramid levels",
    )
    p.add_argument(
        "--pyramid_ratio",
        type=float,
        default=1.8,
        help="Scale factor between pyramid levels",
    )
    p.add_argument("--octaves", type=int, help="Alias for pyramid_size")
    p.add_argument(
        "--octave_scale", type=float, help="Alias for pyramid_ratio"
    )
    p.add_argument("--weights", help="NPZ weights path")
    p.add_argument("--jitter", type=int, default=32)
    p.add_argument("--smoothing_coefficient", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    
    img_np = load_image(args.input, args.img_width)

    if args.model == 'all':
        models = ["vgg16", "vgg19", "googlenet", "resnet50"]
        # Ignore explicit output path to avoid overwriting
        if args.output:
            print("Warning: --output argument ignored because --model='all' was selected.")
            args.output = None 
        
        for m in models:
            run_dream_for_model(m, args, img_np)
    else:
        run_dream_for_model(args.model, args, img_np)


if __name__ == "__main__":
    main()
