# DeepDream-MLX Development Notes

## Unified Architecture (Updated 2025-11-29)

The project has consolidated the DeepDream pipeline into a single source of truth in this repository.

1.  **Core Logic (`dream_core.py`)**: This file houses the shared model registry, weight resolution, and the `run_dream()` function. It is the central engine for all dreaming operations.
2.  **CLI (`dream.py`)**: A thin wrapper that imports `dream_core.run_dream`. This is the shared CLI for both the model repo and the app.
3.  **Usage**: Application workflows should call this CLI (via `python dream.py ...`) or import `run_dream` from `dream_core.py`.

## Optimization Learnings

- **Resume Capability**: `train_dream.py` understands `--resume`. Point it at any saved checkpoint (e.g., `--resume checkpoints_evangelion_multi/googlenet_epoch006.npz`) and it loads the params/optimizer state and restarts automatically.
- **New Backbones**: InceptionV3 and MobileNetV3 are supported.
  - Weights are resolved automatically (e.g., `inception_v3_mlx.npz`).
  - Default layers are set sensibly (`Mixed_5d`, `Mixed_6e` for Inception).
- **Model Registry**: Registry dictionary in `dream_core.py` consolidates VGG, GoogLeNet, ResNet50, Inception, MobileNet, etc.

## Integration Guide

To use this repository as a submodule in a parent application (like the Svelte app):

1.  **Add Submodule**:
    ```bash
    git submodule add https://github.com/NickMystic/deepdream-mlx-models.git deepdream_models
    ```

2.  **Importing**:
    In your parent python scripts, you can now import directly:
    ```python
    import sys
    sys.path.append("deepdream_models")
    from deepdream_models.dream_core import run_dream, list_models
    ```

3.  **Shared Weights**:
    Ensure the parent app points to the weights in this folder, or symlinks them. `dream_core.py` is smart enough to look in `models/` relative to execution if configured, but passing explicit paths via `--weights` is safer for complex setups.
