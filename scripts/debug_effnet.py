
import sys
import os
import mlx.core as mx
import mlx.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from deepdream.models.efficientnet import EfficientNetB0

model = EfficientNetB0()
print("Model children keys:", model.children().keys())

print("Features type:", type(model.features))
if hasattr(model.features, "children"):
    print("Features children keys:", model.features.children().keys())

if hasattr(model.features, "layers"):
    print("Features layers length:", len(model.features.layers))
    print("Layer 0 type:", type(model.features.layers[0]))
    # Check if layer 0 has children
    if hasattr(model.features.layers[0], "children"):
         print("Layer 0 children:", model.features.layers[0].children().keys())

# Test accessing what load_npz accesses
try:
    print("Accessing stem[0]:", model.stem.layers[0])
except Exception as e:
    print("Stem access failed:", e)

try:
    print("Accessing features[0]:", model.features.layers[0])
except Exception as e:
    print("Features access failed:", e)
    
try:
    print("Accessing head[0]:", model.head.layers[0])
except Exception as e:
    print("Head access failed:", e)
