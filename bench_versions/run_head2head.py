#!/usr/bin/env python3
"""
Python runner for the head-to-head DeepDream scripts.
Uses love.jpg at width 400, steps 8, GoogLeNet.
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG = os.path.join(ROOT, "love.jpg")
OUT = os.path.join(ROOT, "bench_versions", "out")
os.makedirs(OUT, exist_ok=True)

def run(name, cmd):
    print(f"---- {name} ----")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH','')}"
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print(f"FAILED: {name} (code {proc.returncode})")
    print()

def main():
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH','')}"

    runs = [
        ("9f21d03_deepdream.py", [
            sys.executable, os.path.join(ROOT, "bench_versions", "dream_9f21d03_deepdream.py"),
            IMG, "--model", "googlenet", "--width", "400", "--steps", "8",
            "--output", os.path.join(OUT, "out_9f21d03_deepdream.jpg"),
        ]),
        ("9f21d03_dream_mlx.py", [
            sys.executable, os.path.join(ROOT, "bench_versions", "dream_9f21d03_dream_mlx.py"),
            "--model", "googlenet", "--input", IMG, "--img_width", "400", "--steps", "8",
            "--output", os.path.join(OUT, "out_9f21d03_dream_mlx.jpg"),
        ]),
        ("97b4af6.py", [
            sys.executable, os.path.join(ROOT, "bench_versions", "dream_97b4af6.py"),
            "--model", "googlenet", "--input", IMG, "--width", "400", "--steps", "8",
            "--output", os.path.join(OUT, "out_97b4af6.jpg"),
        ]),
        ("48a1ec7.py", [
            sys.executable, os.path.join(ROOT, "bench_versions", "dream_48a1ec7.py"),
            "--model", "googlenet", "--input", IMG, "--width", "400", "--steps", "8",
            "--output", os.path.join(OUT, "out_48a1ec7.jpg"),
        ]),
        ("current", [
            sys.executable, os.path.join(ROOT, "bench_versions", "dream_current.py"),
            "--model", "googlenet", "--input", IMG, "--width", "400", "--steps", "8",
            "--output", os.path.join(OUT, "out_current.jpg"),
        ]),
    ]

    for name, cmd in runs:
        run(name, cmd)

if __name__ == "__main__":
    main()
