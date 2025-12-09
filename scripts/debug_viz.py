
import sys
import os
import mlx.core as mx
import mlx.nn as nn

# Ensure we can import deepdream
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from deepdream.models.mobilenet import MobileNetV3Small_Defined

def debug():
    print("Instantiating...")
    model = MobileNetV3Small_Defined()
    print("Children keys:", model.children().keys())
    l1 = getattr(model, "layer1", None)
    if l1:
        print("layer1 type:", type(l1))
        # print("layer1 vars:", list(vars(l1).keys()))
    pass
    
    # Check parameters
    params = model.parameters()
    print("Total params via .parameters():", sum(v.size for v in params.values()))

if __name__ == "__main__":
    debug()
