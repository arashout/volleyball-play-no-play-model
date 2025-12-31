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
        "-ss",
        str(start),
        "-i",
        str(video_path),
        "-t",
        str(duration),
        "-vf",
        f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}",
        "-c:v",
        "libx264",
        "-an",
        "-loglevel",
        "error",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0
