import time
import subprocess

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

from datetime import datetime

def benchmark(script_name):
    cmd = list(CMD_BASE)
    cmd[1] = script_name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    cmd.extend(["--output", f"bench_{script_name}_{timestamp}.jpg"])
    
    print(f"Running {script_name}...")
    start = time.time()
    subprocess.run(cmd, capture_output=True) # Capture output to avoid console I/O affecting timing? 
    # Actually, user suspects console I/O IS the issue, so maybe we should let it print?
    # But for a fair "engine" test, we should silence both or let both run.
    # Let's capture output to measure pure compute time + overhead, ignoring terminal scrolling speed.
    
    end = time.time()
    print(f"{script_name} took {end - start:.4f} seconds")

if __name__ == "__main__":
    benchmark("dream_mlx.py")
    benchmark("deepdream.py")
