"""Shared DeepDream core for MLX CLI and apps."""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np
import scipy.ndimage as nd
from PIL import Image

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover - optional dependency
    hf_hub_download = None

from mlx_alexnet import AlexNet
from mlx_googlenet import GoogLeNet
from mlx_inception_v3 import InceptionV3
from mlx_mobilenet import MobileNetV3Small_Defined
from mlx_resnet50 import ResNet50
from mlx_vgg16 import VGG16
from mlx_vgg19 import VGG19

IMAGENET_MEAN = mx.array([0.485, 0.456, 0.406])
IMAGENET_STD = mx.array([0.229, 0.224, 0.225])
LOWER_IMAGE_BOUND = (-IMAGENET_MEAN / IMAGENET_STD).reshape(1, 1, 1, 3)
UPPER_IMAGE_BOUND = ((1.0 - IMAGENET_MEAN) / IMAGENET_STD).reshape(1, 1, 1, 3)

VGG_PRESETS = {
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

MODEL_REGISTRY = {
    "vgg16": {
        "cls": VGG16,
        "default_layers": ["relu4_3"],
        "weights_key": "vgg16",
        "supports_presets": True,
    },
    "vgg19": {
        "cls": VGG19,
        "default_layers": ["relu4_4"],
        "weights_key": "vgg19",
        "supports_presets": True,
    },
    "resnet50": {
        "cls": ResNet50,
        "default_layers": ["layer4_2"],
        "weights_key": "resnet50",
        "supports_presets": False,
    },
    "alexnet": {
        "cls": AlexNet,
        "default_layers": ["relu5"],
        "weights_key": "alexnet",
        "supports_presets": False,
    },
    "googlenet": {
        "cls": GoogLeNet,
        "default_layers": ["inception3b", "inception4c", "inception4d"],
        "weights_key": "googlenet",
        "supports_presets": False,
    },
    "inception_v3": {
        "cls": InceptionV3,
        "default_layers": ["Mixed_5d", "Mixed_6e", "Mixed_7c"],
        "weights_key": "inception_v3",
        "supports_presets": False,
    },
    "mobilenet_v3": {
        "cls": MobileNetV3Small_Defined,
        "default_layers": ["layer7", "layer9", "layer11"],
        "weights_key": "MobileNetV3",
        "supports_presets": False,
    },
}


def list_models() -> List[str]:
    return sorted(MODEL_REGISTRY.keys())


def load_image(path: str, target_width: Optional[int] = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if target_width:
        w, h = img.size
        scale = target_width / w
        new_h = int(h * scale)
        img = img.resize((target_width, new_h), Image.LANCZOS)
    return np.array(img)


def preprocess(img_np: np.ndarray) -> mx.array:
    x = mx.array(img_np, dtype=mx.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x[None, ...]


def deprocess(x: mx.array) -> np.ndarray:
    x = x[0]
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = mx.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(mx.uint8)
    return np.array(x)


def resize_bilinear(x: mx.array, new_h: int, new_w: int) -> mx.array:
    b, h, w, c = x.shape
    out = mx.zeros((b, new_h, new_w, c))
    for bi in range(b):
        for ci in range(c):
            out[bi, :, :, ci] = mx.array(
                nd.zoom(np.array(x[bi, :, :, ci]), zoom=(new_h / h, new_w / w), order=1)
            )
    return out


def gaussian_kernel(sigma: float, truncate: float = 4.0, fixed_radius: Optional[int] = None) -> mx.array:
    radius = fixed_radius if fixed_radius is not None else int(truncate * sigma + 0.5)
    x = mx.arange(-radius, radius + 1)
    kernel = mx.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_blur_2d(x: mx.array, sigma: float, fixed_radius: Optional[int] = None) -> mx.array:
    kernel = gaussian_kernel(sigma, fixed_radius=fixed_radius).astype(x.dtype)
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


def smooth_gradients(grad: mx.array, sigma: float, fixed_radius: Optional[int] = None) -> mx.array:
    sigmas = [sigma * 0.5, sigma * 1.0, sigma * 2.0]
    smoothed = [gaussian_blur_2d(grad, s, fixed_radius=fixed_radius) for s in sigmas]
    out = smoothed[0]
    for extra in smoothed[1:]:
        out = out + extra
    return out / len(smoothed)


def get_pyramid_shapes(base_shape: Tuple[int, int], num_octaves: int, scale: float) -> List[Tuple[int, int]]:
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
    img_np: np.ndarray,
    layers: List[str],
    steps: int,
    lr: float,
    num_octaves: int,
    scale: float,
    jitter: int,
    smoothing: float,
    guide_img_np: Optional[np.ndarray] = None,
) -> np.ndarray:
    img = preprocess(img_np)
    base_h, base_w = img.shape[1:3]
    pyramid_shapes = get_pyramid_shapes((base_h, base_w), num_octaves, scale)

    for nh, nw in pyramid_shapes:
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
                if guide_features:
                    loss = loss + mx.mean(act * guide_features[name])
                else:
                    loss = loss + mx.mean(act * act)
            return loss / len(layers)

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
            rolled, _ = update_step(rolled, mx.array(sigma_val))
            img = mx.roll(mx.roll(rolled, -ox, axis=1), -oy, axis=2)

    return deprocess(img)


def get_weights_path(weights_key: str, explicit_path: Optional[str], logger: Optional[Callable[[str], None]] = None) -> str:
    repo_id = "NickMystic/DeepDream-MLX"
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    else:
        candidates.append(f"{weights_key}_mlx.npz")
        candidates.append(f"{weights_key}_mlx_bf16.npz")

    for path in candidates:
        if os.path.exists(path):
            return path

    if hf_hub_download is None:
        raise FileNotFoundError(
            f"Missing weights for {weights_key}. Install huggingface_hub or place {candidates[0]} locally."
        )

    errors = []
    for filename in candidates:
        try:
            if logger:
                logger(f"Downloading {os.path.basename(filename)} from {repo_id}...")
            dl_path = hf_hub_download(
                repo_id=repo_id,
                filename=os.path.basename(filename),
                resume_download=True,
                cache_dir=None,
            )
            return dl_path
        except Exception as exc:  # pragma: no cover - network path
            errors.append(f"{filename}: {exc}")

    raise FileNotFoundError(f"Could not resolve weights for {weights_key}. Tried {candidates}. Errors: {errors}")


def run_dream(
    model_name: str,
    *,
    image_np: np.ndarray,
    layers: Optional[List[str]] = None,
    steps: int = 10,
    lr: float = 0.09,
    octaves: int = 4,
    scale: float = 1.8,
    jitter: int = 32,
    smoothing: float = 0.5,
    guide_image_np: Optional[np.ndarray] = None,
    preset: Optional[str] = None,
    weights: Optional[str] = None,
    logger: Optional[Callable[[str], None]] = print,
) -> Tuple[np.ndarray, Dict[str, str]]:
    name = model_name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {', '.join(list_models())}")

    info = MODEL_REGISTRY[name]
    effective_layers = layers or info["default_layers"]
    effective_steps = steps
    effective_lr = lr
    effective_octaves = octaves
    effective_scale = scale
    effective_jitter = jitter
    effective_smoothing = smoothing

    if preset:
        if not info["supports_presets"] or preset not in VGG_PRESETS:
            raise ValueError(f"Preset '{preset}' not supported for model '{model_name}'")
        preset_cfg = VGG_PRESETS[preset]
        effective_layers = preset_cfg["layers"]
        effective_steps = preset_cfg["steps"]
        effective_lr = preset_cfg["lr"]
        effective_octaves = preset_cfg["octaves"]
        effective_scale = preset_cfg["scale"]
        effective_jitter = preset_cfg["jitter"]
        effective_smoothing = preset_cfg["smoothing"]

    weights_path = get_weights_path(info["weights_key"], weights, logger)
    if logger:
        logger(f"Loading weights from: {weights_path}")

    model = info["cls"]()
    model.load_npz(weights_path)

    dreamed = deepdream(
        model,
        image_np,
        layers=effective_layers,
        steps=effective_steps,
        lr=effective_lr,
        num_octaves=effective_octaves,
        scale=effective_scale,
        jitter=effective_jitter,
        smoothing=effective_smoothing,
        guide_img_np=guide_image_np,
    )

    meta = {
        "weights": weights_path,
        "layers": ",".join(effective_layers),
    }
    return dreamed, meta


__all__ = [
    "MODEL_REGISTRY",
    "VGG_PRESETS",
    "list_models",
    "load_image",
    "run_dream",
]
