#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path


def split_validation(train_dir: Path, val_dir: Path, val_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)

    for category in ["play", "no-play"]:
        src_dir = train_dir / category
        dst_dir = val_dir / category
        dst_dir.mkdir(parents=True, exist_ok=True)

        for clip in dst_dir.glob("*.mp4"):
            shutil.move(str(clip), src_dir / clip.name)

        all_clips = list(src_dir.glob("*.mp4"))
        target_val = int(len(all_clips) * val_ratio)

        for clip in random.sample(all_clips, target_val):
            shutil.move(str(clip), dst_dir / clip.name)

        print(f"{category}: {target_val}/{len(all_clips)} in validation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train")
    parser.add_argument("--val", default="data/val")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_validation(Path(args.train), Path(args.val), args.ratio, args.seed)


if __name__ == "__main__":
    main()
