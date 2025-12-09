"""Configuration for the DeepDream TUI."""

PRESETS = [
    {
        "name": "subtle",
        "steps": 8,
        "lr": 0.07,
        "pyramid_size": 4,
        "pyramid_ratio": 1.8,
        "layers": ["relu4_3"],
        "smoothing_coefficient": 0.8,
        "jitter": 24,
    },
    {
        "name": "classic",
        "steps": 10,
        "lr": 0.09,
        "pyramid_size": 4,
        "pyramid_ratio": 1.8,
        "layers": ["relu4_3"],
        "smoothing_coefficient": 0.5,
        "jitter": 32,
    },
    {
        "name": "bold_layers",
        "steps": 14,
        "lr": 0.12,
        "pyramid_size": 5,
        "pyramid_ratio": 1.6,
        "layers": ["relu3_3", "relu4_3"],
        "smoothing_coefficient": 0.35,
        "jitter": 24,
    },
    {
        "name": "sharp_low_smooth",
        "steps": 16,
        "lr": 0.1,
        "pyramid_size": 5,
        "pyramid_ratio": 1.5,
        "layers": ["relu4_3"],
        "smoothing_coefficient": 0.3,
        "jitter": 20,
    },
]

ARG_MAPPING = {
    "lr": "lr",
    "steps": "steps",
    "jitter": "jitter",
    "pyramid_size": "pyramid-size",
    "pyramid_ratio": "pyramid-ratio",
    "smoothing_coefficient": "smoothing-coefficient",
    "layers": "layers",
}
