#!/usr/bin/env python3
"""
Split random files from EvangelionUserInterfaces into EvangelionUserInterfaces_val.
"""

import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split validation subset")
    parser.add_argument("--source", default="EvangelionUserInterfaces ", help="Source folder")
    parser.add_argument("--dest", default="EvangelionUserInterfaces_val", help="Destination val folder")
    parser.add_argument("--count", type=int, default=54, help="Number of files to move")
    args = parser.parse_args()

    src = Path(args.source)
    dst = Path(args.dest)
    dst.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(src.iterdir()) if p.is_file()]
    if len(files) < args.count:
        raise SystemExit(f"Not enough files: {len(files)} available, need {args.count}")
    random.shuffle(files)
    for f in files[: args.count]:
        shutil.move(str(f), dst / f.name)

    print(f"Moved {args.count} files into {dst}")


if __name__ == "__main__":
    main()
