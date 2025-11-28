import mlx.core as mx
import numpy as np
from PIL import Image
from dream import get_weights_path
from mlx_googlenet import GoogLeNet
# from mlx_vgg16 import VGG16 # Uncomment to use VGG16
# from mlx_vgg19 import VGG19 # Uncomment to use VGG19

def preprocess_image(image_path: str, target_size=(224, 224)):
    """
    Loads and preprocesses an image for MLX models.
    Resizes, normalizes, and converts to HWC MLX array.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image = np.array(image, dtype=np.float32) / 255.0 # Scale to [0, 1]

    # Normalize
    image = (image - mean) / std

    # Add batch dimension (B, H, W, C) and convert to MLX array
    image = mx.array(image[np.newaxis, ...])
    return image

def main():
    # Path to a dummy input image. You might need to create one or use an existing one.
    # For example, you can create a dummy 224x224 black image with:
    # `convert -size 224x224 xc:black dummy_input.png` (if ImageMagick is installed)
    # Or simply have an image named 'dummy_input.png' in the same directory.
    input_image_path = "dummy_input.png"

    # --- Load and preprocess image ---
    try:
        input_image = preprocess_image(input_image_path)
        print(f"Preprocessed image shape: {input_image.shape}")
    except FileNotFoundError:
        print(f"Error: Input image '{input_image_path}' not found.")
        print("Please create a dummy_input.png or replace the path with an existing image.")
        return

    # --- Load GoogleNet model and weights ---
    print("Loading GoogleNet model...")
    model = GoogLeNet()
    weights_path = get_weights_path("googlenet")
    model.load_npz(weights_path)
    print(f"GoogleNet weights loaded from: {weights_path}")

    # --- Perform inference ---
    print("Performing inference...")
    # The GoogleNet model returns a dictionary of activations for DeepDream
    activations = model(input_image)
    print("Inference complete.")

    # --- Display some output ---
    print("\nGoogleNet Activations (Layer Names and Shapes):")
    for layer_name, output_tensor in activations.items():
        print(f"  {layer_name}: {output_tensor.shape}")

    # You can uncomment and use VGG16/VGG19 similarly:
    # print("\n--- VGG16 Example (uncomment to run) ---")
    # vgg_model = VGG16()
    # vgg_model.load_npz("vgg16_mlx.npz")
    # vgg_activations = vgg_model(input_image)
    # print("VGG16 Activations (Layer Names and Shapes):")
    # for layer_name, output_tensor in vgg_activations.items():
    #     print(f"  {layer_name}: {output_tensor.shape}")


if __name__ == "__main__":
    main()
