"""TF-Slim InceptionV1 forward callable for TF2 (no KerasTensor issues)."""

import os
from typing import Iterable, Tuple, Callable, List

import tensorflow as tf
import tf_slim as slim
from tf_slim.nets import inception_v1

WEIGHTS_URL = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"
DEFAULT_LAYER_NAMES = (
    "Mixed_4b",
    "Mixed_4c",
    "Mixed_4d",
)


def _download_checkpoint_if_needed(weights_path: str = None) -> str:
    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights path does not exist: {weights_path}")
        return weights_path

    tar_path = tf.keras.utils.get_file(
        origin=WEIGHTS_URL,
        fname=os.path.basename(WEIGHTS_URL),
        extract=True,
        cache_dir=os.path.expanduser("~/.keras"),
    )
    ckpt_dir = os.path.join(os.path.dirname(tar_path), "inception_v1_2016_08_28")
    ckpt_path = os.path.join(ckpt_dir, "inception_v1.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found after download: {ckpt_path}")
    return ckpt_path


def _preprocess_fn(x: tf.Tensor) -> tf.Tensor:
    """Match TF-Slim InceptionV1 preprocessing: scale to [-1, 1]."""
    x = tf.cast(x, tf.float32)
    return (x / 127.5) - 1.0


def build_inception_v1_callable(
    layer_names: Iterable[str] = DEFAULT_LAYER_NAMES, weights_path: str = None
) -> Tuple[Callable[[tf.Tensor], List[tf.Tensor]], Callable[[tf.Tensor], tf.Tensor]]:
    """
    Returns:
        forward_fn: callable taking NHWC float tensor -> list of endpoints
        preprocess_fn: preprocessing callable
    """

    layer_names = tuple(layer_names)
    scope_name = "InceptionV1"

    @tf.function
    def forward_fn(x: tf.Tensor) -> List[tf.Tensor]:
        with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
                _, endpoints = inception_v1.inception_v1(
                    x,
                    num_classes=1001,
                    is_training=False,
                    spatial_squeeze=False,
                )
        return [endpoints[name] for name in layer_names]

    # Build variables by a dummy call
    _ = forward_fn(tf.zeros([1, 224, 224, 3], dtype=tf.float32))

    ckpt_path = _download_checkpoint_if_needed(weights_path)
    var_list = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope_name
    )
    name_map = {v.name.split(":")[0]: v for v in var_list}
    ckpt = tf.train.Checkpoint(**name_map)
    ckpt.restore(ckpt_path).expect_partial()

    return forward_fn, _preprocess_fn

