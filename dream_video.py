#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import mlx.core as mx
import scipy.ndimage as nd
from PIL import Image
from dream import deepdream, load_image, deprocess, get_weights_path
from mlx_googlenet import GoogLeNet
from mlx_resnet50 import ResNet50
from mlx_vgg16 import VGG16
from mlx_vgg19 import VGG19
from mlx_alexnet import AlexNet

def run_video_dream(args):
    print(f"--- DeepDream Video Generator ---")
    print(f"Model: {args.model}")
    print(f"Zoom: {args.zoom_factor}")
    print(f"Frames: {args.frames}")
    
    # 1. Load Model
    if args.model == "vgg16":
        model = VGG16()
        default_layers = ["relu4_3"]
    elif args.model == "vgg19":
        model = VGG19()
        default_layers = ["relu4_4"]
    elif args.model == "resnet50":
        model = ResNet50()
        default_layers = ["layer4_2"]
    elif args.model == "alexnet":
        model = AlexNet()
        default_layers = ["relu5"]
    else:
        model = GoogLeNet()
        default_layers = ["inception4c"]

    weights = get_weights_path(args.model, args.weights)
    if not os.path.exists(weights):
        print(f"Error: Weights {weights} not found.")
        return
    
    print(f"Loading weights: {weights}")
    model.load_npz(weights)

    # 2. Prepare Input
    img_np = load_image(args.input, args.width)
    
    # 3. Prepare Output Dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    current_img = img_np.astype(np.float32)
    
    # 4. Loop
    for i in range(args.frames):
        start_t = time.time()
        
        # Dream
        dreamed = deepdream(
            model,
            current_img,
            layers=args.layers or default_layers,
            steps=args.steps,
            lr=args.lr,
            num_octaves=args.octaves,
            scale=args.scale,
            jitter=args.jitter,
            smoothing=args.smoothing
        )
        
        # Save Frame
        frame_name = f"frame_{i:04d}.jpg"
        out_path = os.path.join(args.output_dir, frame_name)
        Image.fromarray(dreamed).save(out_path)
        
        elapsed = time.time() - start_t
        print(f"Frame {i+1}/{args.frames}: {frame_name} ({elapsed:.2f}s)")
        
        # Transform for next frame (Zoom)
        # Zooming involves:
        # 1. Scaling up by zoom_factor
        # 2. Cropping back to original size (center crop)
        
        if i < args.frames - 1:
            # dreamed is (H, W, 3) uint8
            # Convert back to float for zoom to avoid precision loss
            next_input = dreamed.astype(np.float32)
            
            # Scipy Zoom (order=1 is bilinear, usually sufficient and fast)
            # Zoom H and W dimensions, keep Channel dimension (zoom=1)
            zf = args.zoom_factor
            next_input = nd.zoom(next_input, (zf, zf, 1), order=1)
            
            # Crop Center
            h_new, w_new, _ = next_input.shape
            h_orig, w_orig, _ = img_np.shape
            
            start_h = (h_new - h_orig) // 2
            start_w = (w_new - w_orig) // 2
            
            current_img = next_input[start_h:start_h+h_orig, start_w:start_w+w_orig, :]

    print(f"\nDone! Frames saved to {args.output_dir}/\n")
    print(f"To create video (requires ffmpeg):")
    print(f"ffmpeg -framerate 15 -i {args.output_dir}/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p video.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="frames")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--zoom_factor", type=float, default=1.05)
    
    # Shared dream args
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--model", default="googlenet")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--layers", nargs="+ ")
    parser.add_argument("--steps", type=int, default=5) # Fewer steps for video usually smoother
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--octaves", type=int, default=2) # Fewer octaves for speed
    parser.add_argument("--scale", type=float, default=1.4)
    parser.add_argument("--jitter", type=int, default=32)
    parser.add_argument("--smoothing", type=float, default=0.5)
    
    args = parser.parse_args()
    run_video_dream(args)
