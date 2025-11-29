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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from datasets import ClassLabel, Value, load_dataset
from PIL import Image
from tqdm import tqdm

from mlx_googlenet import GoogLeNetTrain
from mlx_inception_v3 import InceptionV3Train, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE

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
}


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
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        imgs: List[np.ndarray] = []
        labels: List[int] = []
        for idx in batch_idx:
            row = dataset[idx]
            imgs.append(resize_and_norm(row[image_key], image_size, mean, std, preserve_aspect))
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

    if isinstance(model, GoogLeNetTrain):
        export_googlenet(model)
    elif isinstance(model, InceptionV3Train):
        export_inception_v3(model)
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--no-aux", action="store_true", help="Disable auxiliary heads")
    parser.add_argument("--fc-warmup-epochs", type=int, default=0, help="Train only fc+aux for N epochs")
    parser.add_argument("--stat-samples", type=int, default=2048, help="Samples for mean/std/color decorrelation (0=all)")
    parser.add_argument("--skip-color-decorrelation", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Where to write checkpoints")
    parser.add_argument("--save-every", type=int, default=1, help="Epoch cadence for checkpoints")
    parser.add_argument("--export-dream", default=None, help="Path to write dream-ready npz (feature tower only)")
    parser.add_argument("--resume", default=None, help="Resume training from checkpoint npz")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    os.environ.setdefault("HF_DATASETS_CACHE", str((Path(".hf_cache")).resolve()))

    load_kwargs = {"data_dir": args.data_dir} if args.data_dir else {}

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
    model = model_cls(num_classes=num_classes, aux_weight=args.aux_weight, use_aux=not args.no_aux)
    if args.optimizer == "sgd":
        opt = optim.SGD(learning_rate=args.lr, momentum=args.momentum)
    else:
        opt = optim.Adam(learning_rate=args.lr)

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

    base_frozen = model_cfg.get("freeze_prefixes", [])
    warmup_freeze = model_cfg.get("warmup_freeze_prefixes", base_frozen)
    train_only = model_cfg.get("warmup_trainable", [])
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def loss_fn(mdl, x, y):
        logits, a1, a2 = mdl.forward_logits(x, train=True)
        loss = mx.mean(nn.losses.cross_entropy(logits, y))
        if a1 is not None:
            loss = loss + args.aux_weight * mx.mean(nn.losses.cross_entropy(a1, y))
        if a2 is not None:
            loss = loss + args.aux_weight * mx.mean(nn.losses.cross_entropy(a2, y))
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

    for epoch in range(resume_epoch + 1, args.epochs + 1):
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
            ):
                images = mx.array(images_np)
                labels = mx.array(labels_np)
                logits, a1, a2 = model.forward_logits(images, train=False)
                loss = mx.mean(nn.losses.cross_entropy(logits, labels))
                if a1 is not None:
                    loss = loss + args.aux_weight * mx.mean(nn.losses.cross_entropy(a1, labels))
                if a2 is not None:
                    loss = loss + args.aux_weight * mx.mean(nn.losses.cross_entropy(a2, labels))
                val_losses.append(float(loss))
                val_accs.append(float(accuracy_from_logits(logits, labels)))
            avg_val_loss = float(np.mean(val_losses))
            avg_val_acc = float(np.mean(val_accs))
        else:
            avg_val_loss = None
            avg_val_acc = None

        print(
            f"[epoch {epoch:03d}] train_loss={avg_train_loss:.4f} "
            f"train_acc={avg_train_acc:.4f} "
            + (f"val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.4f}" if avg_val_loss is not None else "")
            + (f" warmup_fc_only={warmup}" if warmup else "")
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
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
                export_path = Path(args.export_dream)
                export_path.parent.mkdir(parents=True, exist_ok=True)
                export_dream_npz(model, export_path)
                meta_path = export_path.with_suffix(".json")
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"Saved dream export to {export_path} (+ {meta_path})")
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
