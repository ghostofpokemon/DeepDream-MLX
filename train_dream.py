#!/usr/bin/env python3
"""
Native MLX fine-tuning for DeepDream.

Targets:
- Hugging Face image classification datasets with columns: image, label
- GoogLeNet (Inception v1) with aux heads + optional freezing of early towers

Defaults aim to preserve the 2015 DeepDream look:
- Freeze conv1/conv2/inception3a/inception3b
- Optionally warm up fc/aux-only before unfreezing higher blocks
- Compute dataset mean/std and optional color decorrelation matrix for dreaming
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import wandb
from datasets import ClassLabel, Value, load_dataset
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

from mlx_googlenet import GoogLeNetTrain
from mlx_inception_v3 import InceptionV3Train, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE
from mlx_vgg16 import VGG16Train
from mlx_mobilenet import MobileNetV3SmallTrain

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_CONFIG = {
    "googlenet": {
        "class": GoogLeNetTrain,
        "freeze_prefixes": [
            "conv1",
            "conv2",
            "inception3a",
            "inception3b",
        ],
        "warmup_freeze_prefixes": [
            "conv1",
            "conv2",
            "inception3a",
            "inception3b",
            "inception4a",
            "inception4b",
            "inception4c",
            "inception4d",
            "inception4e",
            "inception5a",
            "inception5b",
            "aux1",
            "aux2",
            "fc",
        ],
        "warmup_trainable": ["aux1", "aux2", "fc"],
    },
    "inception_v3": {
        "class": InceptionV3Train,
        "freeze_prefixes": [
            "Conv2d_1a_3x3",
            "Conv2d_2a_3x3",
            "Conv2d_2b_3x3",
            "Conv2d_3b_1x1",
            "Conv2d_4a_3x3",
            "Mixed_5b",
            "Mixed_5c",
            "Mixed_5d",
            "Mixed_6a",
            "Mixed_6b",
            "Mixed_6c",
            "Mixed_6d",
            "Mixed_6e",
            "Mixed_7a",
        ],
        "warmup_freeze_prefixes": [
            "Conv2d_1a_3x3",
            "Conv2d_2a_3x3",
            "Conv2d_2b_3x3",
            "Conv2d_3b_1x1",
            "Conv2d_4a_3x3",
            "Mixed_5b",
            "Mixed_5c",
            "Mixed_5d",
            "Mixed_6a",
            "Mixed_6b",
            "Mixed_6c",
            "Mixed_6d",
            "Mixed_6e",
            "Mixed_7a",
            "Mixed_7b",
            "Mixed_7c",
            "aux_classifier",
            "fc",
        ],
        "warmup_trainable": ["fc", "aux_classifier"],
    },
    "vgg16": {
        "class": VGG16Train,
        "freeze_prefixes": [
            "layers.0",  # conv1_1
            "layers.2",  # conv1_2
            "layers.5",  # conv2_1
            "layers.7",  # conv2_2
            "layers.10", # conv3_1
            "layers.12", # conv3_2
            "layers.14", # conv3_3
        ],
        "warmup_freeze_prefixes": [
            "layers.0",
            "layers.2",
            "layers.5",
            "layers.7",
            "layers.10",
            "layers.12",
            "layers.14",
            "layers.17", # conv4_1
            "layers.19", # conv4_2
            "layers.21", # conv4_3
            "layers.24", # conv5_1
            "layers.26", # conv5_2
            "layers.28", # conv5_3
            "classifier",
        ],
        "warmup_trainable": ["classifier"],
    },
    "mobilenet": {
        "class": MobileNetV3SmallTrain,
        "freeze_prefixes": [
            "layer0",
            "layer1",
            "layer2",
            "layer3",
        ],
        "warmup_freeze_prefixes": [
            "layer0",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "layer5",
            "layer6",
            "layer7",
            "layer8",
            "layer9",
            "layer10",
            "layer11",
            "layer12",
            "last_conv",
            "classifier",
        ],
        "warmup_trainable": ["classifier"],
    },
}

GOOGLENET_LAYER_ORDER = [
    "conv1",
    "conv2",
    "conv3",
    "inception3a",
    "inception3b",
    "inception4a",
    "inception4b",
    "inception4c",
    "inception4d",
    "inception4e",
    "inception5a",
    "inception5b",
]

VGG16_LAYER_ORDER = [
    "layers.0",  # conv1_1
    "layers.2",  # conv1_2
    "layers.5",  # conv2_1
    "layers.7",  # conv2_2
    "layers.10", # conv3_1
    "layers.12", # conv3_2
    "layers.14", # conv3_3
    "layers.17", # conv4_1
    "layers.19", # conv4_2
    "layers.21", # conv4_3
    "layers.24", # conv5_1
    "layers.26", # conv5_2
    "layers.28", # conv5_3
]

MOBILENET_LAYER_ORDER = [
    "layer0",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "layer5",
    "layer6",
    "layer7",
    "layer8",
    "layer9",
    "layer10",
    "layer11",
    "layer12",
    "last_conv",
]

INCEPTION_V3_LAYER_ORDER = [
    "Conv2d_1a_3x3",
    "Conv2d_2a_3x3",
    "Conv2d_2b_3x3",
    "Conv2d_3b_1x1",
    "Conv2d_4a_3x3",
    "Mixed_5b",
    "Mixed_5c",
    "Mixed_5d",
    "Mixed_6a",
    "Mixed_6b",
    "Mixed_6c",
    "Mixed_6d",
    "Mixed_6e",
    "Mixed_7a",
    "Mixed_7b",
    "Mixed_7c",
]

def aux_prefixes(idx: int, target: str) -> List[str]:
    mapping = {
        "loss_conv": ["aux{idx}.proj"],
        "loss_fc": ["aux{idx}.proj", "aux{idx}.fc1"],
        "loss_classifier": ["aux{idx}.proj", "aux{idx}.fc1", "aux{idx}.fc2"],
    }
    order = ["loss_conv", "loss_fc", "loss_classifier"]
    prefixes: List[str] = []
    for name in order:
        for p in mapping[name]:
            prefix = p.format(idx=idx)
            if prefix not in prefixes:
                prefixes.append(prefix)
        if name == target:
            break
    return prefixes



def to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")
    if hasattr(image, "convert"):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def resize_and_norm(
    image,
    image_size: Optional[int],
    mean: np.ndarray,
    std: np.ndarray,
    preserve_aspect: bool,
) -> np.ndarray:
    pil_img = to_pil(image)
    target = image_size if image_size and image_size > 0 else None
    if target:
        if preserve_aspect:
            pil_img = pad_to_square(pil_img, target, fill=tuple((mean * 255).astype(np.uint8)))
        else:
            pil_img = pil_img.resize((target, target), Image.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    return arr


def pad_to_square(img: Image.Image, size: int, fill=(0, 0, 0)) -> Image.Image:
    w, h = img.size
    if w == h == size:
        return img
    scale = min(size / w, size / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def crop_or_pad(img: Image.Image, target_w: int, target_h: int, fill=(0, 0, 0)) -> Image.Image:
    w, h = img.size
    if w > target_w or h > target_h:
        left = max(0, (w - target_w) // 2)
        top = max(0, (h - target_h) // 2)
        img = img.crop((left, top, left + target_w, top + target_h))
        w, h = img.size
    if w < target_w or h < target_h:
        canvas = Image.new("RGB", (target_w, target_h), fill)
        canvas.paste(img, ((target_w - w) // 2, (target_h - h) // 2))
        img = canvas
    return img


def apply_augmentations(
    img: Image.Image,
    fill_color=(0, 0, 0),
    rotate_deg: float = 0.0,
    zoom_range: float = 0.0,
    brightness: float = 0.0,
    horizontal_flip: bool = True,
) -> Image.Image:
    orig_w, orig_h = img.size
    out = img
    if horizontal_flip and random.random() < 0.5:
        out = ImageOps.mirror(out)
    if rotate_deg and rotate_deg > 0:
        angle = random.uniform(-rotate_deg, rotate_deg)
        out = out.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=fill_color)
        out = crop_or_pad(out, orig_w, orig_h, fill_color)
    if zoom_range and zoom_range > 0:
        lower = max(0.01, 1.0 - zoom_range)
        scale = random.uniform(lower, 1.0 + zoom_range)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        out = out.resize((new_w, new_h), Image.BILINEAR)
        out = crop_or_pad(out, orig_w, orig_h, fill_color)
    if brightness and brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        factor = max(0.1, factor)
        enhancer = ImageEnhance.Brightness(out)
        out = enhancer.enhance(factor)
    return out


def compute_mean_std(
    dataset,
    image_key: str,
    image_size: Optional[int],
    sample_limit: Optional[int],
    preserve_aspect: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    sum_rgb = np.zeros(3, dtype=np.float64)
    sum_sq = np.zeros(3, dtype=np.float64)
    total = 0

    for idx, row in enumerate(tqdm(dataset, desc="Computing mean/std")):
        if sample_limit and idx >= sample_limit:
            break
        pil_img = to_pil(row[image_key])
        target = image_size if image_size and image_size > 0 else None
        if target:
            pil_img = pad_to_square(
                pil_img, target, fill=(0, 0, 0)
            ) if preserve_aspect else pil_img.resize((target, target), Image.BILINEAR)
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        sum_rgb += flat.sum(axis=0)
        sum_sq += (flat * flat).sum(axis=0)
        total += flat.shape[0]

    mean = sum_rgb / total
    var = (sum_sq / total) - (mean * mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def compute_color_decorrelation(
    dataset,
    image_key: str,
    image_size: Optional[int],
    sample_limit: Optional[int],
    preserve_aspect: bool,
) -> np.ndarray:
    sum_rgb = np.zeros(3, dtype=np.float64)
    sum_outer = np.zeros((3, 3), dtype=np.float64)
    total = 0

    for idx, row in enumerate(tqdm(dataset, desc="Computing color decorrelation")):
        if sample_limit and idx >= sample_limit:
            break
        pil_img = to_pil(row[image_key])
        target = image_size if image_size and image_size > 0 else None
        if target:
            pil_img = pad_to_square(
                pil_img, target, fill=(0, 0, 0)
            ) if preserve_aspect else pil_img.resize((target, target), Image.BILINEAR)
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        sum_rgb += flat.sum(axis=0)
        sum_outer += flat.T @ flat
        total += flat.shape[0]

    mean = sum_rgb / total
    cov = (sum_outer - total * np.outer(mean, mean)) / max(total - 1, 1)
    u, s, _ = np.linalg.svd(cov)
    svd_sqrt = u @ np.diag(np.sqrt(s + 1e-10))
    return svd_sqrt.astype(np.float32)


def compute_class_weights(train_ds, label_key: str, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float64)
    labels = train_ds[label_key]
    for lbl in labels:
        counts[int(lbl)] += 1
    counts[counts == 0] = 1.0
    total = counts.sum()
    weights = total / (num_classes * counts)
    # Clamp extremes so rare classes don't explode gradients, then normalize mean to 1.
    weights = np.clip(weights, 0.2, 5.0)
    weights = weights / np.mean(weights)
    return weights.astype(np.float32)


def iter_batches(
    dataset,
    image_key: str,
    label_key: str,
    image_size: Optional[int],
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    shuffle: bool,
    preserve_aspect: bool,
    augment: bool = False,
    aug_params: Optional[Dict] = None,
    fill_color=(0, 0, 0),
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    aug_params = aug_params or {}
    rotate_deg = aug_params.get("rotate", 0.0)
    zoom_range = aug_params.get("zoom", 0.0)
    brightness = aug_params.get("brightness", 0.0)
    horizontal_flip = aug_params.get("hflip", True)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        imgs: List[np.ndarray] = []
        labels: List[int] = []
        for idx in batch_idx:
            row = dataset[idx]
            pil_img = to_pil(row[image_key])
            if augment:
                pil_img = apply_augmentations(
                    pil_img,
                    fill_color=fill_color,
                    rotate_deg=rotate_deg,
                    zoom_range=zoom_range,
                    brightness=brightness,
                    horizontal_flip=horizontal_flip,
                )
            imgs.append(resize_and_norm(pil_img, image_size, mean, std, preserve_aspect))
            labels.append(int(row[label_key]))
        yield np.stack(imgs), np.array(labels, dtype=np.int32)


def mask_frozen(tree, frozen_prefixes: List[str], path: Optional[List[str]] = None):
    if path is None:
        path = []
    if isinstance(tree, dict):
        return {k: mask_frozen(v, frozen_prefixes, path + [k]) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(mask_frozen(v, frozen_prefixes, path + [str(i)]) for i, v in enumerate(tree))
    key = ".".join(path)
    if any(key.startswith(p) for p in frozen_prefixes):
        return mx.zeros_like(tree)
    return tree


def accuracy_from_logits(logits, labels):
    preds = mx.argmax(logits, axis=-1)
    return mx.mean((preds == labels).astype(mx.float32))


def weighted_cross_entropy(logits, labels, class_weights=None):
    max_logit = mx.max(logits, axis=-1, keepdims=True)
    stabilized = logits - max_logit
    logsumexp = mx.log(mx.sum(mx.exp(stabilized), axis=-1) + 1e-8)
    log_probs = stabilized - logsumexp[:, None]
    # One-hot encoding
    if hasattr(mx, "one_hot"):
        one_hot = mx.one_hot(labels, logits.shape[-1])
    else:
        # Fallback for older MLX versions: create identity matrix and index it
        one_hot = mx.eye(logits.shape[-1])[labels]

    ce = -mx.sum(one_hot * log_probs, axis=-1)
    if class_weights is not None:
        sample_weights = mx.take(class_weights, labels)
        ce = ce * sample_weights
    return mx.mean(ce)


from mlx_vgg16 import VGG16Train

def export_dream_npz(model, path: Path):
    data: Dict[str, np.ndarray] = {}

    def to_numpy(arr):
        return np.array(arr, dtype=np.float32)

    def dump_seq_conv_bn(prefix, seq):
        conv = seq.layers[0]
        bn = seq.layers[1]
        w = to_numpy(conv.weight)
        data[f"{prefix}.conv.weight"] = np.transpose(w, (0, 3, 1, 2))
        data[f"{prefix}.bn.weight"] = to_numpy(bn.weight)
        data[f"{prefix}.bn.bias"] = to_numpy(bn.bias)
        data[f"{prefix}.bn.running_mean"] = to_numpy(bn.running_mean)
        data[f"{prefix}.bn.running_var"] = to_numpy(bn.running_var)

    def export_googlenet(mdl: GoogLeNetTrain):
        def dump_inception(prefix, module):
            dump_seq_conv_bn(f"{prefix}.branch1", module.branch1)
            dump_seq_conv_bn(f"{prefix}.branch2.0", module.branch2_1)
            dump_seq_conv_bn(f"{prefix}.branch2.1", module.branch2_2)
            dump_seq_conv_bn(f"{prefix}.branch3.0", module.branch3_1)
            dump_seq_conv_bn(f"{prefix}.branch3.1", module.branch3_2)
            dump_seq_conv_bn(f"{prefix}.branch4.1", module.branch4_2)

        dump_seq_conv_bn("conv1", mdl.conv1)
        dump_seq_conv_bn("conv2", mdl.conv2)
        dump_seq_conv_bn("conv3", mdl.conv3)
        dump_inception("inception3a", mdl.inception3a)
        dump_inception("inception3b", mdl.inception3b)
        dump_inception("inception4a", mdl.inception4a)
        dump_inception("inception4b", mdl.inception4b)
        dump_inception("inception4c", mdl.inception4c)
        dump_inception("inception4d", mdl.inception4d)
        dump_inception("inception4e", mdl.inception4e)
        dump_inception("inception5a", mdl.inception5a)
        dump_inception("inception5b", mdl.inception5b)

    def dump_basic_conv(prefix, module):
        w = to_numpy(module.conv.weight)
        data[f"{prefix}.conv.weight"] = np.transpose(w, (0, 3, 1, 2))
        data[f"{prefix}.bn.weight"] = to_numpy(module.bn.weight)
        data[f"{prefix}.bn.bias"] = to_numpy(module.bn.bias)
        data[f"{prefix}.bn.running_mean"] = to_numpy(module.bn.running_mean)
        data[f"{prefix}.bn.running_var"] = to_numpy(module.bn.running_var)

    def export_inception_v3(mdl: InceptionV3Train):
        dump_basic_conv("Conv2d_1a_3x3", mdl.Conv2d_1a_3x3)
        dump_basic_conv("Conv2d_2a_3x3", mdl.Conv2d_2a_3x3)
        dump_basic_conv("Conv2d_2b_3x3", mdl.Conv2d_2b_3x3)
        dump_basic_conv("Conv2d_3b_1x1", mdl.Conv2d_3b_1x1)
        dump_basic_conv("Conv2d_4a_3x3", mdl.Conv2d_4a_3x3)

        def dump_block(block, prefix):
            if isinstance(block, InceptionA):
                dump_basic_conv(f"{prefix}.branch1x1", block.branch1x1)
                dump_basic_conv(f"{prefix}.branch5x5_1", block.branch5x5_1)
                dump_basic_conv(f"{prefix}.branch5x5_2", block.branch5x5_2)
                dump_basic_conv(f"{prefix}.branch3x3dbl_1", block.branch3x3dbl_1)
                dump_basic_conv(f"{prefix}.branch3x3dbl_2", block.branch3x3dbl_2)
                dump_basic_conv(f"{prefix}.branch3x3dbl_3", block.branch3x3dbl_3)
                dump_basic_conv(f"{prefix}.branch_pool", block.branch_pool)
            elif isinstance(block, InceptionB):
                dump_basic_conv(f"{prefix}.branch3x3", block.branch3x3)
                dump_basic_conv(f"{prefix}.branch3x3dbl_1", block.branch3x3dbl_1)
                dump_basic_conv(f"{prefix}.branch3x3dbl_2", block.branch3x3dbl_2)
                dump_basic_conv(f"{prefix}.branch3x3dbl_3", block.branch3x3dbl_3)
            elif isinstance(block, InceptionC):
                dump_basic_conv(f"{prefix}.branch1x1", block.branch1x1)
                dump_basic_conv(f"{prefix}.branch7x7_1", block.branch7x7_1)
                dump_basic_conv(f"{prefix}.branch7x7_2", block.branch7x7_2)
                dump_basic_conv(f"{prefix}.branch7x7_3", block.branch7x7_3)
                dump_basic_conv(f"{prefix}.branch7x7dbl_1", block.branch7x7dbl_1)
                dump_basic_conv(f"{prefix}.branch7x7dbl_2", block.branch7x7dbl_2)
                dump_basic_conv(f"{prefix}.branch7x7dbl_3", block.branch7x7dbl_3)
                dump_basic_conv(f"{prefix}.branch7x7dbl_4", block.branch7x7dbl_4)
                dump_basic_conv(f"{prefix}.branch7x7dbl_5", block.branch7x7dbl_5)
                dump_basic_conv(f"{prefix}.branch_pool", block.branch_pool)
            elif isinstance(block, InceptionD):
                dump_basic_conv(f"{prefix}.branch3x3_1", block.branch3x3_1)
                dump_basic_conv(f"{prefix}.branch3x3_2", block.branch3x3_2)
                dump_basic_conv(f"{prefix}.branch7x7x3_1", block.branch7x7x3_1)
                dump_basic_conv(f"{prefix}.branch7x7x3_2", block.branch7x7x3_2)
                dump_basic_conv(f"{prefix}.branch7x7x3_3", block.branch7x7x3_3)
                dump_basic_conv(f"{prefix}.branch7x7x3_4", block.branch7x7x3_4)
            elif isinstance(block, InceptionE):
                dump_basic_conv(f"{prefix}.branch1x1", block.branch1x1)
                dump_basic_conv(f"{prefix}.branch3x3_1", block.branch3x3_1)
                dump_basic_conv(f"{prefix}.branch3x3_2a", block.branch3x3_2a)
                dump_basic_conv(f"{prefix}.branch3x3_2b", block.branch3x3_2b)
                dump_basic_conv(f"{prefix}.branch3x3dbl_1", block.branch3x3dbl_1)
                dump_basic_conv(f"{prefix}.branch3x3dbl_2", block.branch3x3dbl_2)
                dump_basic_conv(f"{prefix}.branch3x3dbl_3a", block.branch3x3dbl_3a)
                dump_basic_conv(f"{prefix}.branch3x3dbl_3b", block.branch3x3dbl_3b)
                dump_basic_conv(f"{prefix}.branch_pool", block.branch_pool)
            else:
                raise ValueError(f"Unsupported Inception block type: {type(block)}")

        for name in [
            "Mixed_5b",
            "Mixed_5c",
            "Mixed_5d",
            "Mixed_6a",
            "Mixed_6b",
            "Mixed_6c",
            "Mixed_6d",
            "Mixed_6e",
            "Mixed_7a",
            "Mixed_7b",
            "Mixed_7c",
        ]:
            dump_block(getattr(mdl, name), name)

    def export_vgg16(mdl: VGG16Train):
        conv_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        for idx in conv_indices:
            layer = mdl.layers[idx] # nn.Conv2d
            w = to_numpy(layer.weight)
            b = to_numpy(layer.bias)
            # PyTorch style: features.{idx}.weight
            # We need to transpose MLX [Out, H, W, In] -> PyTorch [Out, In, H, W]
            # But wait, MLX Conv2d is [Out, H, W, In]
            # The load_npz expects [Out, In, H, W] from PyTorch and transposes to MLX
            # So here we should export in PyTorch format [Out, In, H, W] so it's compatible with other tools?
            # OR should we export in MLX format?
            # The other export functions here are doing `np.transpose(w, (0, 3, 1, 2))` which converts MLX to PyTorch.
            # So we should do the same.
            data[f"features.{idx}.weight"] = np.transpose(w, (0, 3, 1, 2))
            data[f"features.{idx}.bias"] = b

    if isinstance(model, GoogLeNetTrain):
        export_googlenet(model)
    elif isinstance(model, InceptionV3Train):
        export_inception_v3(model)
    elif isinstance(model, VGG16Train):
        export_vgg16(model)
    else:
        raise ValueError(f"Dream export unsupported for model type {type(model)}")

    np.savez(path, **data)


def save_checkpoint(
    model: GoogLeNetTrain,
    opt_state,
    metadata: Dict,
    path: Path,
):
    tree = model.trainable_parameters()

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(convert(v) for v in obj)
        try:
            return np.array(obj)
        except Exception:
            return np.array(obj, dtype=object)

    payload = {
        "params": convert(tree),
        "optimizer": convert(opt_state),
        "metadata": metadata,
    }
    np.savez(path, **payload)


def build_freeze_prefixes(always_frozen: List[str], warmup_freeze: Optional[List[str]], trainable_only: Optional[List[str]]) -> List[str]:
    if trainable_only is None:
        return always_frozen
    base = warmup_freeze if warmup_freeze is not None else always_frozen
    return [p for p in base if p not in trainable_only]


RAINBOW_COLORS = [
    "\033[38;5;196m",
    "\033[38;5;202m",
    "\033[38;5;226m",
    "\033[38;5;46m",
    "\033[38;5;21m",
    "\033[38;5;93m",
    "\033[38;5;201m",
]


def rainbow_text(text: str) -> str:
    return "".join(RAINBOW_COLORS[i % len(RAINBOW_COLORS)] + ch for i, ch in enumerate(text)) + "\033[0m"


def prompt_for_pretrained(reason: str) -> Optional[Path]:
    prompt = (
        f"{rainbow_text(reason)}\n"
        "[1] Enter path to pretrained weights\n"
        "[2] YOLO: continue from random weights\n"
        "[3] Abort this training run\n> "
    )
    while True:
        try:
            choice = input(prompt).strip().lower()
        except EOFError:
            choice = "3"
        if choice in ("1", "y", "yes"):
            path = input("Enter pretrained path: ").strip()
            if not path:
                continue
            candidate = Path(path).expanduser().resolve()
            if candidate.exists():
                return candidate
            print(rainbow_text(f"No file at {candidate}. Try again."))
        elif choice in ("2", "n", "no"):
            print(rainbow_text("Proceeding from thin air. You monster."))
            return None
        elif choice in ("3", "q", "quit", "exit", ""):
            print(rainbow_text("Bailing out. Come back when you're prepared."))
            sys.exit(1)
        else:
            print(rainbow_text("Pick 1, 2, or 3, you chaos gremlin."))


def run_checkpoint_compare(args, epoch: int, final: bool):
    if not args.compare_input or not args.export_dream:
        return
    base_weights = args.compare_base_weights or args.pretrained
    if not base_weights:
        return
    compare_script = Path(__file__).resolve().parent / "compare_dreams.py"
    if not compare_script.exists():
        print("compare_dreams.py not found; skipping checkpoint compare.")
        return

    base_weights_path = Path(base_weights).expanduser().resolve()
    if not base_weights_path.exists():
        print(f"Compare base weights missing: {base_weights_path}; skipping checkpoint compare.")
        return

    tuned_weights_path = Path(args.export_dream).expanduser().resolve()
    input_path = Path(args.compare_input).expanduser().resolve()
    if not input_path.exists():
        print(f"Compare input missing: {input_path}; skipping checkpoint compare.")
        return

    output_dir = Path(args.compare_output_dir).expanduser().resolve() / f"epoch{epoch:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(compare_script),
        "--model",
        args.model,
        "--base",
        str(base_weights_path),
        "--tuned",
        str(tuned_weights_path),
        "--input",
        str(input_path),
        "--width",
        str(args.compare_width),
        "--count",
        "0",
        "--output-dir",
        str(output_dir),
    ]

    if args.compare_dream_script:
        cmd += ["--dream-script", args.compare_dream_script]
    overrides = [
        ("--layers", args.compare_layers),
        ("--steps", args.compare_steps),
        ("--lr", args.compare_lr),
        ("--octaves", args.compare_octaves),
        ("--scale", args.compare_scale),
        ("--jitter", args.compare_jitter),
        ("--smoothing", args.compare_smoothing),
    ]
    for flag, value in overrides:
        if value is None:
            continue
        if isinstance(value, list):
            cmd += [flag] + [str(v) for v in value]
        else:
            cmd += [flag, str(value)]

    print("[checkpoint compare]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Checkpoint compare failed: {exc}")
    # else:
    #     if not final and not args.compare_keep_history:
    #         shutil.rmtree(output_dir, ignore_errors=True)




def main():
    parser = argparse.ArgumentParser(description="Fine-tune MLX models for DeepDream")
    parser.add_argument("--model", choices=list(MODEL_CONFIG.keys()), default="googlenet", help="Model architecture to fine-tune.")
    parser.add_argument("--hf-dataset", default="imagenet-sketch", help="Hugging Face dataset id with image+label columns")
    parser.add_argument("--data-dir", default=None, help="Optional local data_dir (e.g., imagefolder)")
    parser.add_argument("--train-split", default="train", help="Dataset split for training")
    parser.add_argument("--val-split", default=None, help="Optional validation split")
    parser.add_argument("--image-key", default="image", help="Column containing images")
    parser.add_argument("--label-key", default="label", help="Column containing class ids")
    parser.add_argument("--image-size", type=int, default=224, help="Resize square side (set 0 to keep native size)")
    parser.add_argument("--preserve-aspect", action="store_true", help="Letterbox to square instead of squashing")
    parser.add_argument("--augment", action="store_true", help="Enable random flip/rotate/zoom/brightness augmentations on training images")
    parser.add_argument("--aug-rotate", type=float, default=15.0, help="Max rotation (degrees) when --augment is set")
    parser.add_argument("--aug-zoom", type=float, default=0.15, help="Zoom jitter fraction (e.g., 0.2 for ±20%) when --augment is set")
    parser.add_argument("--aug-brightness", type=float, default=0.1, help="Brightness jitter fraction when --augment is set")
    parser.add_argument("--no-aug-hflip", action="store_true", help="Disable random horizontal flips during --augment")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--no-aux", action="store_true", help="Disable auxiliary heads")
    parser.add_argument("--delete-branches", action="store_true", help="Drop auxiliary branches completely")
    parser.add_argument("--fc-warmup-epochs", type=int, default=0, help="Train only fc+aux for N epochs")
    parser.add_argument("--stat-samples", type=int, default=2048, help="Samples for mean/std/color decorrelation (0=all)")
    parser.add_argument("--skip-color-decorrelation", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Where to write checkpoints")
    parser.add_argument("--save-every", type=int, default=10, help="Epoch cadence for checkpoints")
    parser.add_argument("--export-dream", default=None, help="Path to write dream-ready npz (feature tower only)")
    parser.add_argument("--resume", default=None, help="Resume training from checkpoint npz")
    parser.add_argument("--balance-classes", action="store_true", help="Weight loss by inverse class frequency")
    parser.add_argument("--wandb-project", default=None, help="WandB project name to log metrics")
    parser.add_argument("--pretrained", default="googlenet_mlx.npz", help="Path to pretrained feature weights (e.g. ImageNet)")
    parser.add_argument("--freeze-to", choices=[
        "none",
        "conv1",
        "conv2",
        "conv3",
        "inception3a",
        "inception3b",
        "inception4a",
        "inception4b",
        "inception4c",
        "inception4d",
        "inception4e",
        "inception5a",
        "inception5b",
        # Inception V3 layers
        "Conv2d_1a_3x3",
        "Conv2d_2a_3x3",
        "Conv2d_2b_3x3",
        "Conv2d_3b_1x1",
        "Conv2d_4a_3x3",
        "Mixed_5b",
        "Mixed_5c",
        "Mixed_5d",
        "Mixed_6a",
        "Mixed_6b",
        "Mixed_6c",
        "Mixed_6d",
        "Mixed_6e",
        "Mixed_7a",
        "Mixed_7b",
        "Mixed_7c",
        # VGG16 layers
        "layers.0",
        "layers.2",
        "layers.5",
        "layers.7",
        "layers.10",
        "layers.12",
        "layers.14",
        "layers.17",
        "layers.19",
        "layers.21",
        "layers.24",
        "layers.26",
        "layers.28",
    ], default="none", help="Freeze layers up to this block (inclusive)")
    parser.add_argument("--freeze-aux1-to", choices=["none", "loss_conv", "loss_fc", "loss_classifier"], default="none")
    parser.add_argument("--freeze-aux2-to", choices=["none", "loss_conv", "loss_fc", "loss_classifier"], default="none")
    parser.add_argument("--compare-input", help="Run a checkpoint comparison dream on this image after every export")
    parser.add_argument("--compare-width", type=int, default=400, help="Width used for checkpoint compare dreams")
    parser.add_argument("--compare-layers", nargs="+", help="Layers to use for checkpoint compare dreams")
    parser.add_argument("--compare-steps", type=int)
    parser.add_argument("--compare-lr", type=float)
    parser.add_argument("--compare-octaves", type=int)
    parser.add_argument("--compare-scale", type=float)
    parser.add_argument("--compare-jitter", type=int)
    parser.add_argument("--compare-smoothing", type=float)
    parser.add_argument("--compare-output-dir", default="checkpoint_compares", help="Where checkpoint compare images go")
    parser.add_argument("--compare-base-weights", help="Override base weights for checkpoint compares (defaults to --pretrained)")
    parser.add_argument("--compare-dream-script", help="Custom dream.py path for checkpoint compares")
    parser.add_argument("--compare-keep-history", action="store_true", help="Keep every compare result instead of deleting intermediate ones")
    parser.add_argument("--max-checkpoints", type=int, default=5, help="Keep only the last N checkpoints (0=keep all)")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    os.environ.setdefault("HF_DATASETS_CACHE", str((Path(".hf_cache")).resolve()))

    load_kwargs = {"data_dir": args.data_dir} if args.data_dir else {}

    if args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args))

    export_path = None
    if args.export_dream:
        export_path = Path(args.export_dream).resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        args.export_dream = str(export_path)

    # Track saved checkpoints for rotation
    saved_checkpoints = []

    def load_split(name: str):
        try:
            return load_dataset(args.hf_dataset, split=name, **load_kwargs)
        except ValueError as e:
            if name == "val":
                try:
                    print('Split "val" not found; falling back to "validation"')
                    return load_dataset(args.hf_dataset, split="validation", **load_kwargs)
                except Exception:
                    pass
            raise e

    train_ds = load_split(args.train_split)
    val_ds = load_split(args.val_split) if args.val_split else None
    model_name = args.model

    features = train_ds.features
    label_key = args.label_key if args.label_key in features else None
    if label_key is None:
        for k, v in features.items():
            if isinstance(v, ClassLabel):
                label_key = k
                break
        if label_key is None:
            for k, v in features.items():
                if isinstance(v, Value) and v.dtype.startswith("int"):
                    label_key = k
                    break
    if label_key is None:
        # Single-class fallback: add label column of zeros
        print("No label column found; assuming single-class dataset and adding label=0.")
        train_ds = train_ds.add_column("label", [0] * len(train_ds))
        if val_ds:
            val_ds = val_ds.add_column("label", [0] * len(val_ds))
        label_key = "label"
        num_classes = 1
        class_names = ["class0"]
    else:
        label_feature = features[label_key]
        if hasattr(label_feature, "num_classes"):
            num_classes = label_feature.num_classes
            class_names = label_feature.names
        else:
            num_classes = int(max(train_ds[label_key])) + 1
            class_names = None

    print(
        f"Loaded dataset: {len(train_ds)} train / {len(val_ds) if val_ds else 0} val | num_classes={num_classes}"
    )
    class_weights_mx = None
    if args.balance_classes and num_classes > 1:
        weights_np = compute_class_weights(train_ds, label_key, num_classes)
        class_weights_mx = mx.array(weights_np)

    resume_epoch = 0
    resume_meta = None
    resume_params = None
    resume_opt_state = None

    if args.resume:
        ckpt_data = np.load(args.resume, allow_pickle=True)
        resume_params = ckpt_data["params"].item()
        resume_opt_state = ckpt_data["optimizer"].item()
        resume_meta = ckpt_data["metadata"].item()
        resume_epoch = resume_meta.get("epoch", 0)
        resume_model = resume_meta.get("model", model_name)
        if resume_model and resume_model != model_name:
            print(f"Checkpoint specifies model '{resume_model}'. Overriding --model {model_name}.")
            model_name = resume_model

    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Unsupported model '{model_name}'. Available: {list(MODEL_CONFIG.keys())}")
    model_cfg = MODEL_CONFIG[model_name]
    model_cls = model_cfg["class"]
    use_aux = not (args.no_aux or args.delete_branches)
    model = model_cls(num_classes=num_classes, aux_weight=args.aux_weight, use_aux=use_aux)
    if args.optimizer == "sgd":
        opt = optim.SGD(learning_rate=args.lr, momentum=args.momentum)
    else:
        opt = optim.Adam(learning_rate=args.lr)

    pretrained_path = None
    if not args.resume:
        if not args.pretrained:
            reason = "HOW THE FUCK ARE YOU GOING TO FINE-TUNE SOMETHING WHEN YOU DON'T EVEN HAVE A (tuned) MODEL ? RETARD"
            pretrained_path = prompt_for_pretrained(reason)
        else:
            candidate = Path(args.pretrained).expanduser().resolve()
            if candidate.exists():
                pretrained_path = candidate
            else:
                reason = f"HOW THE FUCK ARE YOU GOING TO FINE-TUNE SOMETHING WHEN {candidate} DOESN'T EVEN FUCKING EXIST?"
                pretrained_path = prompt_for_pretrained(reason)
        if pretrained_path is None:
            print(rainbow_text("No pretrained weights selected. Initializing from random."))
        else:
            args.pretrained = str(pretrained_path)
            print(f"Loading pretrained weights from {pretrained_path}...")
            try:
                model.load_npz(str(pretrained_path))
            except Exception as e:
                raise RuntimeError(f"Failed to load pretrained weights {pretrained_path}: {e}")
            print("Pretrained weights loaded successfully (features only).")

    def numpy_to_mx(obj):
        if isinstance(obj, dict):
            return {k: numpy_to_mx(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(numpy_to_mx(v) for v in obj)
        if isinstance(obj, np.ndarray):
            return mx.array(obj)
        return obj

    if resume_params is not None:
        model.update(numpy_to_mx(resume_params))
        opt.state = numpy_to_mx(resume_opt_state)
        opt._initialized = True
        if resume_epoch >= args.epochs:
            print(f"Checkpoint epoch {resume_epoch} already >= target epochs {args.epochs}. Nothing to do.")
            return
        print(f"Resuming from epoch {resume_epoch} (checkpoint {args.resume}).")

    stat_limit = args.stat_samples if args.stat_samples > 0 else None
    mean, std = compute_mean_std(train_ds, args.image_key, args.image_size, stat_limit, args.preserve_aspect)
    if args.skip_color_decorrelation:
        color_matrix = None
    else:
        color_matrix = compute_color_decorrelation(train_ds, args.image_key, args.image_size, stat_limit, args.preserve_aspect)
    if resume_meta:
        mean = np.array(resume_meta.get("mean", mean), dtype=np.float32)
        std = np.array(resume_meta.get("std", std), dtype=np.float32)
        if resume_meta.get("color_decorrelation") is not None:
            color_matrix = np.array(resume_meta["color_decorrelation"], dtype=np.float32)

    fill_color = tuple(int(np.clip(round(float(c) * 255.0), 0, 255)) for c in mean)
    aug_params = {
        "rotate": args.aug_rotate,
        "zoom": args.aug_zoom,
        "brightness": args.aug_brightness,
        "hflip": not args.no_aug_hflip,
    }

    base_frozen = list(model_cfg.get("freeze_prefixes", []))
    warmup_freeze = model_cfg.get("warmup_freeze_prefixes", base_frozen)
    train_only = model_cfg.get("warmup_trainable", [])

    if model_name == "googlenet" and args.freeze_to != "none":
        try:
            idx = GOOGLENET_LAYER_ORDER.index(args.freeze_to)
            base_frozen = GOOGLENET_LAYER_ORDER[: idx + 1]
            warmup_freeze = base_frozen
        except ValueError:
            pass
    
    if model_name == "inception_v3" and args.freeze_to != "none":
        try:
            idx = INCEPTION_V3_LAYER_ORDER.index(args.freeze_to)
            base_frozen = INCEPTION_V3_LAYER_ORDER[: idx + 1]
            warmup_freeze = base_frozen
        except ValueError:
            pass

    if model_name == "vgg16" and args.freeze_to != "none":
        try:
            idx = VGG16_LAYER_ORDER.index(args.freeze_to)
            base_frozen = VGG16_LAYER_ORDER[: idx + 1]
            warmup_freeze = base_frozen
        except ValueError:
            pass

    if model_name == "mobilenet" and args.freeze_to != "none":
        try:
            idx = MOBILENET_LAYER_ORDER.index(args.freeze_to)
            base_frozen = MOBILENET_LAYER_ORDER[: idx + 1]
            warmup_freeze = base_frozen
        except ValueError:
            pass

    if model_name == "googlenet" and not args.delete_branches:
        if args.freeze_aux1_to != "none":
            base_frozen += aux_prefixes(1, args.freeze_aux1_to)
        if args.freeze_aux2_to != "none":
            base_frozen += aux_prefixes(2, args.freeze_aux2_to)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)


    def loss_fn(mdl, x, y):
        logits, a1, a2 = mdl.forward_logits(x, train=True)
        loss = weighted_cross_entropy(logits, y, class_weights_mx)
        if a1 is not None:
            loss = loss + args.aux_weight * weighted_cross_entropy(a1, y, class_weights_mx)
        if a2 is not None:
            loss = loss + args.aux_weight * weighted_cross_entropy(a2, y, class_weights_mx)
        return loss

    def add_weight_decay(grads, params, wd):
        if wd <= 0.0:
            return grads
        if isinstance(grads, dict):
            return {k: add_weight_decay(grads[k], params[k], wd) for k in grads}
        if isinstance(grads, (list, tuple)):
            return type(grads)(add_weight_decay(g, p, wd) for g, p in zip(grads, params))
        return grads + params * wd

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Initialize training history locally
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(resume_epoch + 1, args.epochs + 1):
        start_time = time.time() # Start timer for the epoch
        warmup = epoch <= args.fc_warmup_epochs and args.fc_warmup_epochs > 0
        frozen = build_freeze_prefixes(base_frozen, warmup_freeze, train_only if warmup else None)

        train_losses = []
        train_accs = []
        for images_np, labels_np in iter_batches(
            train_ds,
            args.image_key,
            label_key,
            args.image_size,
            mean,
            std,
            args.batch_size,
            shuffle=True,
            preserve_aspect=args.preserve_aspect,
            augment=args.augment,
            aug_params=aug_params if args.augment else None,
            fill_color=fill_color,
        ):
            images = mx.array(images_np)
            labels = mx.array(labels_np)
            loss, grads = loss_and_grad_fn(model, images, labels)
            params = model.trainable_parameters()
            grads = add_weight_decay(grads, params, args.weight_decay)
            grads = mask_frozen(grads, frozen)
            opt.update(model, grads)
            train_losses.append(float(loss))
            logits, _, _ = model.forward_logits(images, train=False)
            train_accs.append(float(accuracy_from_logits(logits, labels)))

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        avg_train_acc = float(np.mean(train_accs)) if train_accs else 0.0

        if val_ds:
            val_losses = []
            val_accs = []
            for images_np, labels_np in iter_batches(
                val_ds,
                args.image_key,
                label_key,
                args.image_size,
                mean,
                std,
                args.batch_size,
                shuffle=False,
                preserve_aspect=args.preserve_aspect,
                augment=False,
                fill_color=fill_color,
            ):
                images = mx.array(images_np)
                labels = mx.array(labels_np)
                logits, a1, a2 = model.forward_logits(images, train=False)
                loss = weighted_cross_entropy(logits, labels, class_weights_mx)
                if a1 is not None:
                    loss = loss + args.aux_weight * weighted_cross_entropy(a1, labels, class_weights_mx)
                if a2 is not None:
                    loss = loss + args.aux_weight * weighted_cross_entropy(a2, labels, class_weights_mx)
                val_losses.append(float(loss))
                val_accs.append(float(accuracy_from_logits(logits, labels)))
            avg_val_loss = float(np.mean(val_losses))
            avg_val_acc = float(np.mean(val_accs))
        else:
            avg_val_loss = None
            avg_val_acc = None

        epoch_time = time.time() - start_time # Calculate elapsed time
        print(
            f"[epoch {epoch:03d}] time={epoch_time:.2f}s | train_loss={avg_train_loss:.4f} "
            f"train_acc={avg_train_acc:.4f} "
            + (f"val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.4f}" if avg_val_loss is not None else "")
            + (f" warmup_fc_only={warmup}" if warmup else "")
        )

        # --- PLOTTING ---
        try:
            import matplotlib.pyplot as plt
            
            history['loss'].append(avg_train_loss)
            history['accuracy'].append(avg_train_acc)
            if val_ds:
                history['val_loss'].append(avg_val_loss)
                history['val_accuracy'].append(avg_val_acc)
            
            # Plot Loss
            plt.figure(figsize=(10, 6))
            plt.plot(history['loss'], label='train loss')
            if val_ds:
                plt.plot(history['val_loss'], label='val loss')
            plt.title(f'Model Loss (Epoch {epoch})')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            plt.savefig(ckpt_dir / 'LossVal_loss.png')
            plt.close()

            # Plot Accuracy
            plt.figure(figsize=(10, 6))
            plt.plot(history['accuracy'], label='train acc')
            if val_ds:
                plt.plot(history['val_accuracy'], label='val acc')
            plt.title(f'Model Accuracy (Epoch {epoch})')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            plt.savefig(ckpt_dir / 'AccVal_acc.png')
            plt.close()
            
        except ImportError:
            pass # Matplotlib not installed
        except Exception as e:
            print(f"Plotting failed: {e}")
        # ----------------

        if args.wandb_project:
            metrics = {
                "epoch": epoch,
                "time": epoch_time,
                "train_loss": avg_train_loss,
                "train_acc": avg_train_acc,
            }
            if avg_val_loss is not None:
                metrics["val_loss"] = avg_val_loss
                metrics["val_acc"] = avg_val_acc
            wandb.log(metrics)

        save_now = epoch % args.save_every == 0 or epoch == args.epochs
        if save_now:
            meta = {
                "epoch": epoch,
                "model": model_name,
                "hf_dataset": args.hf_dataset,
                "train_split": args.train_split,
                "val_split": args.val_split,
                "image_key": args.image_key,
                "label_key": label_key,
                "image_size": args.image_size,
                "num_classes": num_classes,
                "class_names": class_names,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "color_decorrelation": color_matrix.tolist() if color_matrix is not None else None,
                "aux_weight": args.aux_weight,
                "use_aux": not args.no_aux,
            }
            ckpt_path = ckpt_dir / f"{model_name}_epoch{epoch:03d}.npz"
            save_checkpoint(model, opt.state, meta, ckpt_path)
            if args.export_dream:
                export_dream_npz(model, Path(args.export_dream))
                meta_path = Path(args.export_dream).with_suffix(".json")
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"Saved dream export to {args.export_dream} (+ {meta_path})")
            print(f"Saved checkpoint to {ckpt_path}")
            
            # Checkpoint rotation
            saved_checkpoints.append(ckpt_path)
            if args.max_checkpoints > 0 and len(saved_checkpoints) > args.max_checkpoints:
                oldest = saved_checkpoints.pop(0)
                try:
                    if oldest.exists():
                        oldest.unlink()
                        print(f"Deleted old checkpoint {oldest.name} (max_checkpoints={args.max_checkpoints})")
                except Exception as e:
                    print(f"Failed to delete old checkpoint {oldest}: {e}")

            run_checkpoint_compare(args, epoch, final=(epoch == args.epochs))


if __name__ == "__main__":
    main()
