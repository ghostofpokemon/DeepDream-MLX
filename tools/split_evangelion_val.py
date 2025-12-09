#!/usr/bin/env python3
"""
Split a labeled ImageFolder dataset into train/val splits.
Moves a percentage of images from each class folder into a sibling 'val' directory.
"""

import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Root of the dataset (containing class folders)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of data to move to validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    source_path = Path(args.source).resolve()
    
    # If source ends in 'train', we assume the parent is the dataset root
    # and we want to create 'val' alongside 'train'.
    # If source is the root (containing class folders directly), we should probably structure it first.
    
    # Let's assume the user provides the folder containing class subfolders.
    # We will Create a temporary structure if it's not already nested.
    
    # Actually, usually ImageFolder expects:
    # dataset/train/classA
    # dataset/train/classB
    # dataset/val/classA
    # dataset/val/classB
    
    # If the user provides 'dataset/evangelion-ui-5class' and it has class folders directly:
    # We should move them into 'train' first? Or just split them?
    
    # Let's keep it simple:
    # Input: dataset/evangelion-ui-5class
    # Action: 
    #   1. Detect class folders.
    #   2. For each class, identify images.
    #   3. Create dataset/evangelion-ui-5class/train (if not exists) and dataset/evangelion-ui-5class/val
    #   4. Move everything into 'train' first (if not already there).
    #   5. Move random subset to 'val'.
    
    print(f"Processing {source_path}...")
    
    # Detect if we are already inside a 'train' folder or at root
    is_already_split = (source_path / "train").exists()
    
    classes = [d for d in source_path.iterdir() if d.is_dir() and d.name not in [".git", ".DS_Store", "train", "val"]]
    
    if not classes and is_already_split:
        # We are at root, and 'train' exists. Let's operate on 'train'.
        root_path = source_path
        train_path = source_path / "train"
        val_path = source_path / "val"
        print(f"Detected split structure. Splitting from {train_path} to {val_path}")
        classes = [d for d in train_path.iterdir() if d.is_dir()]
    else:
        # We are likely at root with flat classes. Move them to 'train'.
        print("Detected flat structure. Moving to 'train'/'val' layout.")
        root_path = source_path
        train_path = source_path / "train"
        val_path = source_path / "val"
        
        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)
        
        # Move current class folders into train
        for class_dir in classes:
            print(f"Moving {class_dir.name} -> train/{class_dir.name}")
            shutil.move(str(class_dir), str(train_path / class_dir.name))
        
        # Re-list classes from new train location
        classes = [d for d in train_path.iterdir() if d.is_dir()]

    val_path.mkdir(parents=True, exist_ok=True)
    
    total_moved = 0
    for class_dir in classes:
        images = sorted([p for p in class_dir.glob("*") if p.is_file()])
        random.shuffle(images)
        
        num_val = int(len(images) * args.val_ratio)
        val_images = images[:num_val]
        
        dest_class_dir = val_path / class_dir.name
        dest_class_dir.mkdir(parents=True, exist_ok=True)
        
        for img in val_images:
            shutil.move(str(img), str(dest_class_dir / img.name))
            
        print(f"  {class_dir.name}: {len(images)} total -> {len(images)-num_val} train, {num_val} val")
        total_moved += num_val

    print(f"Done. Moved {total_moved} images to {val_path}")

if __name__ == "__main__":
    main()
