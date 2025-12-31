#!/usr/bin/env python3
import argparse
from pathlib import Path

from common import extract_clip, parse_time


def parse_timestamps(timestamps_str: str) -> list[tuple[float, float]]:
    ranges = []
    for part in timestamps_str.split(","):
        start_str, end_str = part.strip().split("-")
        start = parse_time(start_str)
        end = parse_time(end_str)
        ranges.append((start, end))
    return ranges


def main():
    parser = argparse.ArgumentParser(description="Extract no-play clips from video")
    parser.add_argument("--video", required=True, help="Source video file")
    parser.add_argument(
        "--timestamps",
        required=True,
        help="Timestamp ranges (e.g. '0:00-1:00,5:30-6:15')",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--size", type=int, default=224, help="Output size (default: 224)"
    )
    parser.add_argument(
        "--segments", type=int, default=1, help="Split each range into n segments"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output)
    video_name = video_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    ranges = parse_timestamps(args.timestamps)
    total_clips = len(ranges) * args.segments
    print(
        f"Extracting {total_clips} no-play clips ({len(ranges)} ranges x {args.segments} segments)"
    )

    count = 0
    for start, end in ranges:
        total_duration = end - start
        segment_duration = total_duration / args.segments
        for seg in range(args.segments):
            seg_start = start + seg * segment_duration
            count += 1
            output_path = output_dir / f"no-play_{count:04d}_{video_name}.mp4"
            print(
                f"Extracting clip {count}: {seg_start:.1f}s - {seg_start + segment_duration:.1f}s ({segment_duration:.1f}s)"
            )
            if not extract_clip(
                video_path, seg_start, segment_duration, output_path, args.size
            ):
                print("  Failed to extract clip")
                count -= 1

    print(f"\nDone! Extracted {count} no-play clips")


if __name__ == "__main__":
    main()
