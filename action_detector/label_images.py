import argparse
import base64
import glob
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import cast

import anthropic

from .models import ClipAnalysis
from .prompts import SYSTEM_PROMPT

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


def analyze_clip(client: anthropic.Anthropic, clip_name: str, frames: list[Path]) -> ClipAnalysis:
    content = []
    for i, frame_path in enumerate(frames):
        media_type = get_image_media_type(frame_path)
        image_data = encode_image(frame_path)
        content.append({
            "type": "text",
            "text": f"Frame {i}:",
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            },
        })

    content.append({
        "type": "text",
        "text": f"Clip: {clip_name}\n\nAnalyze these {len(frames)} frames and return detections for frames with ball contact.",
    })

    messages = cast(list, [{"role": "user", "content": content}])

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=[
            {
                "name": "report_detections",
                "description": "Report the detected volleyball actions across the clip frames",
                "input_schema": ClipAnalysis.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": "report_detections"},
    )

    for block in message.content:
        if block.type == "tool_use" and block.name == "report_detections":
            return ClipAnalysis.model_validate(block.input)

    return ClipAnalysis(frames=[])


def process_clip(
    client: anthropic.Anthropic,
    clip_name: str,
    frames: list[Path],
    output_dir: Path,
) -> int:
    try:
        result = analyze_clip(client, clip_name, frames)
        print(f"\n=== {clip_name} ({len(frames)} frames) ===")

        detection_count = 0
        for i, frame_path in enumerate(frames):
            detection = result.frames[i] if i < len(result.frames) else None
            label_path = output_dir / f"{frame_path.stem}.txt"

            if detection:
                label_path.write_text(detection.to_yolo_line())
                reasoning_path = output_dir / f"{frame_path.stem}_reasoning.txt"
                reasoning_path.write_text(f"{detection.action.name}: {detection.reasoning}")
                print(f"  Frame {i}: {detection.action.name}")
                detection_count += 1
            else:
                label_path.write_text("")

        print(f"  Detections: {detection_count}/{len(frames)} frames")
        return detection_count

    except anthropic.RateLimitError:
        print(f"Rate limited on {clip_name}, waiting 60s...")
        time.sleep(60)
        return process_clip(client, clip_name, frames, output_dir)
    except Exception as e:
        print(f"Error processing {clip_name}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Label volleyball clips using Claude Vision")
    parser.add_argument("pattern", help="Glob pattern for images (e.g., './screenshots/*.png')")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory for YOLO labels")
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    all_images = [
        Path(p) for p in glob.glob(args.pattern)
        if Path(p).is_file() and Path(p).suffix.lower() in IMAGE_EXTENSIONS
    ]
    all_images.sort()

    if not all_images:
        print(f"No images found matching pattern: {args.pattern}")
        sys.exit(1)

    clips = group_images_by_clip(all_images)
    processed_log = load_processed_log(output_dir)
    clips_to_process = {k: v for k, v in clips.items() if k not in processed_log}

    print(f"Found {len(clips)} clips, {len(clips_to_process)} remaining to process")

    total_detections = 0
    for clip_name, frames in clips_to_process.items():
        detections = process_clip(client, clip_name, frames, output_dir)
        append_to_processed_log(output_dir, clip_name)
        total_detections += detections

    print(f"\nDone! Processed {len(clips_to_process)} clips with {total_detections} total detections")


if __name__ == "__main__":
    main()
