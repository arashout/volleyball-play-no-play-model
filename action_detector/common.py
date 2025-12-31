import base64
import re
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_processed_log(output_dir: Path) -> set[str]:
    log_path = output_dir / "processed_log.txt"
    if not log_path.exists():
        return set()
    return set(log_path.read_text().strip().split("\n"))


def append_to_processed_log(output_dir: Path, clip_name: str):
    log_path = output_dir / "processed_log.txt"
    with open(log_path, "a") as f:
        f.write(f"{clip_name}\n")


def get_image_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".webp":
        return "image/webp"
    return "image/jpeg"


def encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def extract_clip_name(filename: str) -> str:
    match = re.match(r"(.+)_f\d+", filename)
    if match:
        return match.group(1)
    return filename


def extract_frame_number(filename: str) -> int:
    match = re.search(r"_f(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0


def group_images_by_clip(images: list[Path]) -> dict[str, list[Path]]:
    clips = defaultdict(list)
    for img in images:
        clip_name = extract_clip_name(img.stem)
        clips[clip_name].append(img)
    for clip_name in clips:
        clips[clip_name].sort(key=lambda p: extract_frame_number(p.stem))
    return dict(clips)
