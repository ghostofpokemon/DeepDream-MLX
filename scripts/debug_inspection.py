
import sys
import os
import mlx.core as mx
import mlx.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from deepdream.models.inception_v3 import InceptionV3

model = InceptionV3()
print("Dir:", dir(model))
if hasattr(model, "children"):
    print("Children keys:", model.children().keys())

conv1 = model.Conv2d_1a_3x3.conv
print("Leaf vars:", list(vars(conv1).keys()))
if hasattr(conv1, "parameters"):
    print("Leaf params keys:", conv1.parameters().keys())
