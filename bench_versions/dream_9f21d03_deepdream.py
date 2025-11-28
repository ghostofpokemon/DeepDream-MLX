"""
Simple DeepDream with MLX.
Defaults to VGG16.

Usage:
    python deepdream.py input.jpg
    python deepdream.py input.jpg --model googlenet
"""

import argparse
import os
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

# Import local model definitions
from mlx_googlenet import GoogLeNet
from mlx_vgg16 import VGG16
from mlx_vgg19 import VGG19

IMAGENET_MEAN = mx.array([0.485, 0.456, 0.406])
IMAGENET_STD = mx.array([0.229, 0.224, 0.225])
LOWER_IMAGE_BOUND = ((-IMAGENET_MEAN / IMAGENET_STD)).reshape(1, 1, 1, 3)
UPPER_IMAGE_BOUND = (((1.0 - IMAGENET_MEAN) / IMAGENET_STD)).reshape(1, 1, 1, 3)


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
    # Input x is (B, H, W, C) MLX array
    # Convert to numpy, then PIL, resize, back to MLX
    # This is slower than native GPU resize but avoids Scipy dependency
    # and is "fast enough" for the few resize steps in DeepDream.
    
    x_np = np.array(x)
    b, h, w, c = x_np.shape
    out = np.zeros((b, new_h, new_w, c), dtype=x_np.dtype)
    
    for i in range(b):
        # Convert to PIL Image (handling float range if necessary, but here we just resize)
        # Since we are in "normalized" space, we can't just cast to uint8.
        # We resize each channel or treat as float image if possible.
        # PIL floats are tricky. Let's resize channel-wise or use opencv if available?
        # Actually, simple loop over channels with PIL.Image.fromarray(..., mode='F') works.
        
        for ci in range(c):
            # Extract channel (H, W)
            chan = x_np[i, :, :, ci]
            img_pil = Image.fromarray(chan, mode='F')
            # Resize
            img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
            out[i, :, :, ci] = np.array(img_resized)
            
    return mx.array(out)


def smooth_gradients(grad, sigma):
    """
    Smoothes gradients to produce more coherent hallucinations.
    Approximates Gaussian blur using a simple box filter or coordinate shifting
    if we want to avoid scipy.
    
    However, to stay dependency-light WITHOUT scipy, we can implement a simple
    Gaussian blur kernel in MLX or numpy.
    """
    # Simple Numpy implementation of Gaussian blur to avoid scipy
    # We can use a small kernel convolution or just keep it simple.
    # For "lowest friction", let's implement a basic convolution in pure MLX/Numpy 
    # or just rely on the fact that scipy is actually quite standard.
    
    # BUT, the user wanted "lowest friction".
    # Let's use a simple separating kernel (1D convs) in numpy for now.
    
    g_np = np.array(grad)
    k_size = int(4 * sigma + 1) | 1 # Odd kernel size
    if k_size < 3: return grad
    
    # Create 1D Gaussian kernel
    x = np.arange(k_size) - k_size // 2
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    
    # Apply to height and width
    # (B, H, W, C)
    for i in range(3): # Multi-stage blur approximation like original
         # Simple separable convolution using numpy apply_along_axis or similar is slow.
         # Let's just stick to a very simple box blur approximation if scipy is gone,
         # OR bring back scipy.ndimage for JUST gaussian_filter if we accept one extra dep.
         
         # WAIT: The user asked for "lowest friction". 
         # Installing scipy IS friction (compilation sometimes).
         # Let's use a custom simple blur.
         pass

    # Actually, for TRUE DeepDream quality, Gaussian blur is critical.
    # I will re-implement a minimal gaussian_filter using numpy to avoid scipy.
    
    def gaussian_filter_1d(arr, sigma, axis):
        # Very basic implementation
        wd = int(4 * sigma + 1) | 1
        rad = wd // 2
        x = np.arange(wd) - rad
        k = np.exp(-x**2 / (2*sigma**2))
        k /= k.sum()
        return np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), axis=axis, arr=arr)

    sigmas = [sigma * 0.5, sigma * 1.0, sigma * 2.0]
    smoothed = []
    for s in sigmas:
        # Blur H (axis 1) then W (axis 2)
        b1 = gaussian_filter_1d(g_np, s, axis=1)
        b2 = gaussian_filter_1d(b1, s, axis=2)
        smoothed.append(b2)
        
    g_np = sum(smoothed) / len(smoothed)
    return mx.array(g_np, dtype=grad.dtype)


def get_pyramid_shapes(base_shape, pyramid_size, pyramid_ratio):
    h, w = base_shape
    shapes = []
    for level in range(pyramid_size):
        exponent = level - pyramid_size + 1
        nh = max(1, int(round(h * (pyramid_ratio ** exponent))))
        nw = max(1, int(round(w * (pyramid_ratio ** exponent))))
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
):
    img = preprocess(img_np)
    base_h, base_w = img.shape[1:3]
    pyramid_shapes = get_pyramid_shapes((base_h, base_w), pyramid_size, pyramid_ratio)

    print(f"Dreaming on {len(layers)} layers: {layers}")

    for level, (nh, nw) in enumerate(pyramid_shapes):
        img = resize_bilinear(img, nh, nw)
        print(f"Scale {level+1}/{len(pyramid_shapes)}: {nw}x{nh}")

        for it in range(steps):
            # Jitter
            ox, oy = np.random.randint(-jitter, jitter + 1, 2)
            rolled = mx.roll(mx.roll(img, ox, axis=1), oy, axis=2)

            def loss_fn(x):
                endpoints = model.forward_with_endpoints(x)[1]
                loss = mx.zeros(())
                for name in layers:
                    act = endpoints[name]
                    # Maximize L2 norm of activations
                    loss = loss + mx.mean(act * act)
                return loss / len(layers)

            loss, grads = mx.value_and_grad(loss_fn)(rolled)
            
            # Normalize gradients
            g = smooth_gradients(grads, sigma=((it + 1) / steps) * 2.0 + smoothing_coefficient)
            g = g - mx.mean(g)
            g = g / (mx.std(g) + 1e-8)
            
            # Ascent step
            rolled = rolled + lr * g

            # Clip
            rolled = mx.minimum(mx.maximum(rolled, LOWER_IMAGE_BOUND), UPPER_IMAGE_BOUND)
            
            # Un-Jitter
            img = mx.roll(mx.roll(rolled, -ox, axis=1), -oy, axis=2)
            
            print(f"  Step {it+1}/{steps} loss: {loss.item():.4f}", end="\r")
        print() # Newline after steps

    return deprocess(img)


def parse_args():
    p = argparse.ArgumentParser(description="DeepDream with MLX")
    p.add_argument("input", help="Input image path")
    p.add_argument("--output", help="Output image path")
    p.add_argument("--model", choices=["vgg16", "vgg19", "googlenet"], default="vgg16", help="Model to use")
    p.add_argument("--layers", nargs="+", help="Layers to maximize (overrides defaults)")
    p.add_argument("--width", type=int, default=600, help="Resize input to this width")
    p.add_argument("--steps", type=int, default=10, help="Gradient ascent steps per scale")
    p.add_argument("--lr", type=float, default=0.09, help="Learning rate")
    p.add_argument("--pyramid_size", type=int, default=4, help="Number of scales")
    p.add_argument("--pyramid_ratio", type=float, default=1.8, help="Scale factor")
    p.add_argument("--jitter", type=int, default=32, help="Jitter amount (pixels)")
    p.add_argument("--smoothing_coefficient", type=float, default=0.5, help="Gradient smoothing strength")
    return p.parse_args()


def main():
    args = parse_args()

    # Defaults
    if args.model == "vgg16":
        model = VGG16()
        weights = "vgg16_mlx.npz"
        default_layers = ["relu4_3"]
    elif args.model == "vgg19":
        model = VGG19()
        weights = "vgg19_mlx.npz"
        default_layers = ["relu4_4"]
    elif args.model == "googlenet":
        model = GoogLeNet()
        weights = "googlenet_mlx.npz"
        default_layers = ["inception4c", "inception4d"]
    
    # Check weights
    if not os.path.exists(weights):
        print(f"Error: Weights file '{weights}' not found.")
        print(f"Please ensure you are in the correct directory or copy '{weights}' here.")
        return

    print(f"Loading {args.model} weights from {weights}...")
    model.load_npz(weights)
    print("Model loaded.")

    # Output filename
    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = f"{base}_dream_{args.model}_{timestamp}{ext}"

    # Load Image
    try:
        img_np = load_image(args.input, args.width)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Run DeepDream
    result = deepdream(
        model,
        img_np,
        layers=args.layers or default_layers,
        steps=args.steps,
        lr=args.lr,
        pyramid_size=args.pyramid_size,
        pyramid_ratio=args.pyramid_ratio,
        jitter=args.jitter,
        smoothing_coefficient=args.smoothing_coefficient,
    )

    Image.fromarray(result).save(out_path)
    print(f"Dream saved to: {out_path}")


if __name__ == "__main__":
    main()
