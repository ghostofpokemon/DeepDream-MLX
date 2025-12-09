#!/usr/bin/env python3
"""
🔮 ＤＥＥＰＤＲＥＡＭ // ＩＮＴＥＲＡＣＴＩＶＥ
   Aesthetic CLI for iterating through neural hallucinations.
"""

import sys
import time
import shutil
import subprocess
import random
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent))
try:
    from image_utils import show_inline
except ImportError:
    def show_inline(path, width=600):
        print(f"[Image: {path}]")

# --- ＡＥＳＴＨＥＴＩＣＳ ---

CYAN = "\033[38;5;51m"
PURPLE = "\033[38;5;129m"
PINK = "\033[38;5;213m"
GREEN = "\033[38;5;46m"
YELLOW = "\033[38;5;226m"
RED = "\033[38;5;196m"
GREY = "\033[38;5;240m"
RESET = "\033[0m"
BOLD = "\033[1m"

BANNER = f"""
{PURPLE}      .           .
    /' \         / `\\
   /   |  {PINK}.---.{PURPLE}  |   \\
  |    | {PINK}/ {CYAN}● {PINK}\ {PURPLE}|    |
  |    | {PINK}\ {CYAN}● {PINK}/ {PURPLE}|    |
   \   |  {PINK}'---'{PURPLE}  |   /
    \./         \./  
{PINK}  ＤＥＥＰＤＲＥＡＭ{RESET}
 {CYAN}ＩＮＴＥＲＡＣＴＩＶＥ{RESET}
"""

def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

def box_print(title, content, color=CYAN):
    width = 60
    print(f"{color}╔{'═' * (width-2)}╗{RESET}")
    print(f"{color}║ {BOLD}{title.center(width-4)}{RESET}{color} ║{RESET}")
    print(f"{color}╠{'═' * (width-2)}╣{RESET}")
    for line in content.split('\n'):
        if line.strip():
            # Split key/value by '::' for alignment if present
            if "::" in line:
                key, val = line.split("::", 1)
                print(f"{color}║{RESET} {key.strip():<10} {GREY}::{RESET} {val.strip().ljust(width-18)} {color}║{RESET}")
            else:
                print(f"{color}║{RESET} {line.ljust(width-4)} {color}║{RESET}")
    print(f"{color}╚{'═' * (width-2)}╝{RESET}")

def loading_bar(duration=1.5):
    chars = " ▂▃▄▅▆▇█"
    width = 30
    start = time.time()
    while time.time() - start < duration:
        progress = (time.time() - start) / duration
        filled = int(width * progress)
        bar = f"{PINK}{'█' * filled}{GREY}{'░' * (width - filled)}{RESET}"
        idx = int(progress * len(chars)) % len(chars)
        spinner = chars[idx]
        sys.stdout.write(f"\r  {CYAN}⚙{RESET} {bar} {PURPLE}{spinner}{RESET} Calculating...")
        sys.stdout.flush()
        time.sleep(0.05)
    print("\r" + " " * 60 + "\r", end="")

# --- ＬＯＧＩＣ ---

def generate_noise_image(width, height, filename="noise.jpg"):
    """Generates a random RGB noise image."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(filename)
    return filename

def run_dream(input_path, output_path, model, weights_path, layer, steps, scale, octaves):
    cmd = [
        "python", "dream.py",
        "--input", str(input_path),
        "--output", str(output_path),
        "--model", model,
        "--weights", str(weights_path),
        "--layers", layer,
        "--steps", str(steps),
        "--scale", str(scale),
        "--octaves", str(octaves),
        "--width", "800"
    ]
    
    # Hide raw output, we show our own progress
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{RED}✖ FATAL ERROR:{RESET}")
        print(e.stderr.decode())
        return False

def main():
    clear_screen()
    print(BANNER)
    
    # State
    state = {
        "input": "love.jpg",
        "model": "vgg16",
        "weights": "exports/eva_deep.npz",
        "layer": "relu4_3",
        "steps": "60",
        "scale": "1.4",
        "octaves": "4"
    }
    
    out_dir = Path("interactive_outs")
    out_dir.mkdir(exist_ok=True)

    while True:
        clear_screen()
        print(BANNER)
        # Status Display
        status_text = (
            f"{GREY}INPUT   ::{RESET} {state['input']}\n"
            f"{GREY}MODEL   ::{RESET} {state['model']} @ {state['weights']}\n"
            f"{GREY}TARGET  ::{RESET} {state['layer']}\n"
            f"{GREY}CONFIG  ::{RESET} {state['steps']} steps | x{state['scale']} scale"
        )
        box_print("SYSTEM STATUS", status_text)
        
        print(f"\n{BOLD}C O M M A N D S :{RESET}")
        print(f"  {GREEN}[1]{RESET} ⚡ DREAM         {GREEN}[2]{RESET} 📂 CHANGE INPUT")
        print(f"  {GREEN}[3]{RESET} 🧠 CHANGE MODEL  {GREEN}[4]{RESET} 🎯 CHANGE LAYER")
        print(f"  {GREEN}[5]{RESET} ⚙️  CONFIG        {GREEN}[6]{RESET} 🎲 GEN FROM NOISE")
        print(f"  {RED}[q]{RESET} 🚪 EXIT")
        
        choice = input(f"\n{CYAN}user@deepdream:~$ {RESET}").strip().lower()
        
        if choice == 'q':
            print(f"\n{PURPLE}See you space cowboy...{RESET}")
            break
            
        elif choice == '2':
            path = input(f"{GREY}Enter path:{RESET} ").strip()
            if Path(path).exists():
                state["input"] = path
            else:
                print(f"{RED}File not found.{RESET}")
                time.sleep(1)

        elif choice == '3':
            w = input(f"{GREY}Weights (.npz) [{state['weights']}]:{RESET} ").strip()
            if w: state["weights"] = w
            m = input(f"{GREY}Architecture [{state['model']}]:{RESET} ").strip()
            if m: state["model"] = m

        elif choice == '4':
            l = input(f"{GREY}Layer(s) [{state['layer']}]:{RESET} ").strip()
            if l: state["layer"] = l

        elif choice == '5':
            s = input(f"{GREY}Steps [{state['steps']}]:{RESET} ").strip()
            if s: state["steps"] = s
            sc = input(f"{GREY}Scale [{state['scale']}]:{RESET} ").strip()
            if sc: state["scale"] = sc

        elif choice == '1':
            ts = int(time.time())
            out_path = out_dir / f"dream_{ts}.jpg"
            
            print(f"\n{PINK}>>> INITIATING NEURAL HALLUCINATION SEQUENCE...{RESET}")
            loading_bar()
            
            success = run_dream(
                state["input"], out_path, state["model"], state["weights"], 
                state["layer"], state["steps"], state["scale"], state["octaves"]
            )
            
            if success:
                print(f"\n{GREEN}✔ SEQUENCE COMPLETE.{RESET} {GREY}({out_path}){RESET}")
                show_inline(out_path, width=800)
                
                # Feedback Loop
                print(f"\n{YELLOW}OPTIONS:{RESET} [y] Use as input  [s] Save & Continue  [n] Discard")
                fb = input(f"{CYAN}>> {RESET}").lower()
                
                if fb == 'y':
                    state["input"] = str(out_path)
                    print(f"{GREEN}↺ FEEDBACK LOOP CLOSED.{RESET}")
                elif fb == 's':
                    save_name = input(f"{GREY}Save as (e.g. result.jpg): {RESET}").strip() or f"result_{ts}.jpg"
                    shutil.copy(out_path, save_name)
                    print(f"{GREEN}Saved to {save_name}{RESET}")
            print("\n")

        elif choice == '6':
            ts = int(time.time())
            noise_file = f"noise_{ts}.jpg"
            out_path = out_dir / f"gen_noise_{ts}.jpg"
            
            print(f"\n{YELLOW}>>> GENERATING STATIC SIGNAL...{RESET}")
            generate_noise_image(800, 600, noise_file)
            
            print(f"{PINK}>>> DREAMING NEURAL PATTERNS...{RESET}")
            loading_bar()
            
            success = run_dream(
                noise_file, out_path, state["model"], state["weights"], 
                state["layer"], state["steps"], state["scale"], state["octaves"]
            )
            
            # Cleanup
            Path(noise_file).unlink(missing_ok=True)
            
            if success:
                print(f"\n{GREEN}✔ GENERATION COMPLETE.{RESET} {GREY}({out_path}){RESET}")
                show_inline(out_path, width=800)
                
                print(f"\n{YELLOW}OPTIONS:{RESET} [y] Use as input  [s] Save & Continue  [n] Discard")
                fb = input(f"{CYAN}>> {RESET}").lower()
                
                if fb == 'y':
                    state["input"] = str(out_path)
                    print(f"{GREEN}↺ FEEDBACK LOOP CLOSED.{RESET}")
                elif fb == 's':
                    save_name = input(f"{GREY}Save as (e.g. result.jpg): {RESET}").strip() or f"result_{ts}.jpg"
                    shutil.copy(out_path, save_name)
                    print(f"{GREEN}Saved to {save_name}{RESET}")
            print("\n")

if __name__ == "__main__":
    main()