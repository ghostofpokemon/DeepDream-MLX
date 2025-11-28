


import mlx.core as mx

import mlx.nn as nn

import numpy as np

from mlx_googlenet import GoogLeNet

import os



def main():

    print("--- Attempting Extreme Quantization (4-bit / 8-bit) ---")

    

    # Load standard model

    model = GoogLeNet()

    model.load_npz("googlenet_mlx_bf16.npz") 

    

    print("Original Weights Loaded.")

    

    print("\nStrategy: Quantize weights to INT8 (Storage Optimization)")

    # We will effectively store weights as (int8_weight, float16_scale)

    # On load, we will do: weight = int8_weight.astype(fp16) * scale

    

    state = model.parameters()

    compressed_state = {}

    

    total_original = 0

    total_compressed = 0

    

    for k, v in state.items():

        # Flatten keys for parameters() which returns nested dicts if using trees, 

        # but model.parameters() returns nested dict of arrays? 

        # No, mlx model.parameters() returns a dict of {name: array} if flattened?

        # Actually model.parameters() returns a generator or dict?

        # model.parameters() returns a dict of arrays recursively?

        # Let's use flatten logic manually or just iterate what we have.

        pass



    # Actually model.state_dict() is better for flat keys

    # Wait, MLX doesn't have state_dict() like PyTorch exactly?

    # mlx.nn.utils.tree_flatten(model.parameters()) gives list.

    

    # Let's assume we work on the flattened dict structure we used for saving npz

    # Our export script did: np.savez(out, **{k: v})

    # Our load_npz in models does: data[key]

    

    # So we should load the .npz FILE directly and process it, 

    # rather than traversing the model object which might be complex.

    

    data = np.load("googlenet_mlx_bf16.npz")

    

    for k in data.files:

        v = mx.array(data[k])

        

        # Check if it's a weight (conv or linear)

        # Heuristic: name ends in ".weight" and ndim >= 2

        if "weight" in k and v.ndim >= 2:

            # Quantize to INT8

            v_abs = mx.abs(v)

            v_max = mx.max(v_abs)

            

            # Scale to range [-127, 127]

            # Avoid div by zero

            scale = v_max / 127.0

            scale = mx.where(scale == 0, 1.0, scale)

            

            v_int8 = (v / scale).astype(mx.int8)

            

            # Save components

            compressed_state[f"{k}_int8"] = np.array(v_int8)

            compressed_state[f"{k}_scale"] = np.array(scale.astype(mx.float16))

            

            original_bytes = v.nbytes

            new_bytes = v_int8.nbytes + 2 # scale size

            

            total_original += original_bytes

            total_compressed += new_bytes

            

        else:

            # Save as is (float16)

            compressed_state[k] = np.array(v.astype(mx.float16))

            total_original += v.nbytes

            total_compressed += v.nbytes



    out_name = "googlenet_mlx_int8.npz"

    np.savez(out_name, **compressed_state)

    

    print(f"\n✅ Saved {out_name}")

    print(f"   Original Size: {total_original / (1024*1024):.2f} MB")

    print(f"   Quantized Size: {total_compressed / (1024*1024):.2f} MB")

    print(f"   Reduction: {100 * (1 - total_compressed/total_original):.1f}%")



if __name__ == "__main__":

    main()
