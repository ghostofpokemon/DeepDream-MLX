import argparse
import os
import sys
import time
from datetime import datetime

from PIL import Image

from deepdream import list_models, load_image, run_dream
from deepdream.term_image import print_image
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({
    "info": "cyan",
    "warning": "magenta",
    "danger": "bold red",
    "success": "bold green",
    "title": "bold purple",
    "highlight": "bold pink1",
})
console = Console(theme=custom_theme)


def parse_args():
    p = argparse.ArgumentParser(description="DeepDream with MLX (Compiled)")
    p.add_argument("--input", required=True, help="Input image path")
    p.add_argument("--output", help="Output image path (optional)")
    p.add_argument("--guide", help="Guide image for guided dreaming")

    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Resize input to width (maintains aspect ratio)",
    )
    p.add_argument(
        "--img_width", type=int, help="Alias for --width", dest="width"
    )  # Alias

    model_choices = list_models()
    p.add_argument(
        "--model",
        choices=model_choices + ["all"],
        default=model_choices[0],
        help="Model to use. 'all' runs all models.",
    )
    p.add_argument("--preset", choices=["nb14", "nb20", "nb28"], help="VGG16 presets")

    p.add_argument("--layers", nargs="+", help="Layers to maximize")
    p.add_argument(
        "--steps", type=int, default=10, help="Gradient ascent steps per octave"
    )
    p.add_argument("--lr", type=float, default=0.09, help="Learning rate (step size)")

    p.add_argument("--octaves", type=int, default=4, help="Number of image octaves")
    p.add_argument(
        "--pyramid_size", type=int, dest="octaves", help="Alias for --octaves"
    )  # Alias

    p.add_argument("--scale", type=float, default=1.8, help="Octave scale factor")
    p.add_argument(
        "--pyramid_ratio", type=float, dest="scale", help="Alias for --scale"
    )  # Alias
    p.add_argument(
        "--octave_scale", type=float, dest="scale", help="Alias for --scale"
    )  # Alias

    p.add_argument("--jitter", type=int, default=32, help="Jitter amount (pixels)")

    p.add_argument(
        "--smoothing", type=float, default=0.5, help="Gradient smoothing strength"
    )
    p.add_argument(
        "--smoothing_coefficient",
        type=float,
        dest="smoothing",
        help="Alias for --smoothing",
    )  # Alias

    p.add_argument("--weights", help="Custom weights path")

    return p.parse_args()


def run_dream_for_model(model_name, args, img_np):
    console.print(f"[bold cyan]--- Running DeepDream with [white]{model_name}[/white] ---[/bold cyan]")
    if args.preset and model_name not in ("vgg16", "vgg19"):
        raise ValueError(f"Presets only supported for VGG models, not '{model_name}'")

    guide_np = load_image(args.guide, args.width) if args.guide else None
    if args.guide:
        console.print(f"[italic]Using guide image: {args.guide}[/italic]")

    start_time = time.time()
    start_timestamp = datetime.now()

    dreamed, meta = run_dream(
        model_name,
        image_np=img_np,
        layers=args.layers,
        steps=args.steps,
        lr=args.lr,
        octaves=args.octaves,
        scale=args.scale,
        jitter=args.jitter,
        smoothing=args.smoothing,
        guide_image_np=guide_np,
        preset=args.preset,
        weights=args.weights,
    )

    elapsed = time.time() - start_time
    if args.output:
        out_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        formatted_time = f"{elapsed:.2f}s"
        formatted_date = start_timestamp.strftime("%m%d")
        formatted_timestamp = start_timestamp.strftime("%H%M%S")
        out_path = f"{base_name}_dream_{model_name}_{formatted_time}_{formatted_date}_{formatted_timestamp}.jpg"

    Image.fromarray(dreamed).save(out_path)
    console.print(f"[success]Saved {out_path}[/success]")
    console.print(f"[dim]Layers: {meta['layers']} | Weights: {meta['weights']}[/dim]\n")
    
    # Print inline image if compatible terminal
    if sys.stdout.isatty():
        print_image(out_path)


from deepdream.visualization import print_header

def main():
    # print_header("DEEP DREAM MLX") # Using rich header instead
    console.print("[bold pink1]╔══════════════════════════════════════════════╗[/bold pink1]")
    console.print("[bold pink1]║             [cyan]DEEP DREAM MLX[/cyan]                   ║[/bold pink1]")
    console.print("[bold pink1]╚══════════════════════════════════════════════╝[/bold pink1]")
    
    args = parse_args()
    img_np = load_image(args.input, args.width)

    if args.model == "all":
        models = list_models()
        if args.output:
            console.print("[warning]Warning: --output ignored when --model=all; generating unique names instead.[/warning]")
            args.output = None
        for m in models:
            run_dream_for_model(m, args, img_np)
    else:
        run_dream_for_model(args.model, args, img_np)


if __name__ == "__main__":
    main()
