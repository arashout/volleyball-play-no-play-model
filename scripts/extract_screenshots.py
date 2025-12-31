#!/usr/bin/env python3
import subprocess
import csv
import sys
from pathlib import Path


def time_to_seconds(time_str: str) -> float:
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(time_str)


def seconds_to_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def extract_screenshots(
    video_path: str, csv_path: str, output_dir: str = "screenshots", fps: int = 15
):
    video = Path(video_path)
    if not video.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            time = row["time"]
            action = row["action"].replace(" ", "_")
            start_sec = time_to_seconds(time)

            for i in range(fps):
                frame_time = start_sec + (i / fps)
                timestamp = seconds_to_time(frame_time)
                filename = f"{time.replace(':', '_')}_{action}_f{i:02d}.jpg"
                output_path = out / filename

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    timestamp,
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    "-vf",
                    "scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2:black",
                    str(output_path),
                ]
                subprocess.run(cmd, capture_output=True)
            print(f"Extracted {fps} frames for: {time} {action}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <video_file> <csv_file> [output_dir] [fps]")
        sys.exit(1)

    video = sys.argv[1]
    csv_file = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else "screenshots"
    fps = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    extract_screenshots(video, csv_file, output, fps)
