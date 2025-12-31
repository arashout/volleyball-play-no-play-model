#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def parse_time(time_str: str) -> float:
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        hours = 0
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Invalid time format: {time_str}")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def extract_clip(
    video_path: Path,
    start: float,
    duration: float,
    output_path: Path,
    size: int = 224,
) -> bool:
    if duration <= 0:
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-vf", f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}",
        "-c:v", "libx264",
        "-an",
        "-loglevel", "error",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


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
    parser.add_argument("--timestamps", required=True, help="Timestamp ranges (e.g. '0:00-1:00,5:30-6:15')")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=224, help="Output size (default: 224)")
    parser.add_argument("--segments", type=int, default=1, help="Split each range into n segments")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output)
    video_name = video_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    ranges = parse_timestamps(args.timestamps)
    total_clips = len(ranges) * args.segments
    print(f"Extracting {total_clips} no-play clips ({len(ranges)} ranges x {args.segments} segments)")

    count = 0
    for start, end in ranges:
        total_duration = end - start
        segment_duration = total_duration / args.segments
        for seg in range(args.segments):
            seg_start = start + seg * segment_duration
            count += 1
            output_path = output_dir / f"no-play_{count:04d}_{video_name}.mp4"
            print(f"Extracting clip {count}: {seg_start:.1f}s - {seg_start + segment_duration:.1f}s ({segment_duration:.1f}s)")
            if not extract_clip(video_path, seg_start, segment_duration, output_path, args.size):
                print("  Failed to extract clip")
                count -= 1

    print(f"\nDone! Extracted {count} no-play clips")


if __name__ == "__main__":
    main()
