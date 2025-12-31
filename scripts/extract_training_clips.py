#!/usr/bin/env python3
import argparse
import csv
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


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


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


def parse_rallies(csv_path: Path) -> list[tuple[float, float]]:
    rallies = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = parse_time(row["start"])
            end = parse_time(row["end"])
            rallies.append((start, end))
    return rallies


def main():
    parser = argparse.ArgumentParser(description="Extract training clips from video")
    parser.add_argument("--video", required=True, help="Source video file")
    parser.add_argument("--csv", required=True, help="CSV with rally timestamps")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=224, help="Output size (default: 224)")
    parser.add_argument("--trim-end", type=float, default=2.0, help="Seconds to trim from rally end (default: 4)")
    args = parser.parse_args()

    video_path = Path(args.video)
    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    video_name = video_path.stem

    play_dir = output_dir / "play"
    no_play_dir = output_dir / "no-play"
    play_dir.mkdir(parents=True, exist_ok=True)
    no_play_dir.mkdir(parents=True, exist_ok=True)

    rallies = parse_rallies(csv_path)
    print(f"Found {len(rallies)} rallies")

    play_count = 0
    no_play_count = 0

    for i, (start, end) in enumerate(rallies):
        play_duration = (end - args.trim_end) - start
        if play_duration > 0:
            play_count += 1
            output_path = play_dir / f"play_{play_count:04d}_{video_name}.mp4"
            print(f"Extracting play clip {play_count}: {format_duration(start)} - {format_duration(end - args.trim_end)}")
            if not extract_clip(video_path, start, play_duration, output_path, args.size):
                print("  Failed to extract clip")
                play_count -= 1

        if i < len(rallies) - 1:
            gap_start = end
            gap_end = rallies[i + 1][0]
            gap_duration = gap_end - gap_start
            if gap_duration > 0:
                no_play_count += 1
                output_path = no_play_dir / f"no-play_{no_play_count:04d}_{video_name}.mp4"
                print(f"Extracting no_play clip {no_play_count}: {format_duration(gap_start)} - {format_duration(gap_end)}")
                if not extract_clip(video_path, gap_start, gap_duration, output_path, args.size):
                    print("  Failed to extract clip")
                    no_play_count -= 1

    print(f"\nDone! Extracted {play_count} play clips and {no_play_count} no_play clips")


if __name__ == "__main__":
    main()
