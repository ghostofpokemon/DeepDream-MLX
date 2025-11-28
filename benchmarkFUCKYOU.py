from datetime import datetime
import time
import subprocess
import os

CMD_BASE = [
    "python",
    "deepdream.py", # or dream_mlx.py
    "love.jpg",
    "--model", "vgg16",
    "--steps", "10",
    "--lr", "0.09",
    "--pyramid_size", "4",
    "--pyramid_ratio", "1.8",
    "--jitter", "32",
    "--smoothing_coefficient", "0.5",
    "--layers", "relu4_3"
]

def benchmark(script_name):
    cmd = list(CMD_BASE)
    cmd[1] = script_name
    
    # Adjust arguments based on script
    image_arg = cmd[2] # "love.jpg"
    if script_name == "dream_mlx.py":
        # dream_mlx.py requires --input before the image
        cmd[2] = "--input"
        cmd.insert(3, image_arg)
    elif script_name == "dream_pt.py":
        # dream_pt.py takes positional input
        # and uses DIFFERENT flags
        # It doesn't support --model, --pyramid_size (it has --octaves flag but not size arg?), --jitter, --smoothing
        # It has --steps (we added it), --step_size (instead of --lr)
        
        # Rebuild command for dream_pt.py
        cmd = [
            "python",
            "dream_pt.py",
            image_arg,
            "--steps", "10",
            "--step_size", "0.09",
            # "--octaves" # Optional, enables multi-scale
        ]
    else:
        # deepdream.py takes image as positional arg (cmd[2] is fine)
        pass

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = f"bench_{script_name}_{timestamp}.jpg"
    
    if script_name == "dream_pt.py":
        cmd.extend(["--output_image", out_file])
    else:
        cmd.extend(["--output", out_file])
    
    print(f"Running {script_name}...")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.time()
    
    if result.returncode != 0:
        print(f"❌ {script_name} FAILED with error:")
        print(result.stderr)
    else:
        print(f"✅ {script_name} took {end - start:.4f} seconds")
        if os.path.exists(out_file):
            print(f"   Output created: {out_file}")
        else:
            print(f"   ⚠️ Warning: Command succeeded but {out_file} was not found.")

if __name__ == "__main__":
    benchmark("dream_mlx.py")
    benchmark("deepdream.py")
    benchmark("dream_pt.py")
