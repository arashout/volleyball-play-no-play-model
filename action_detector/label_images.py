import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import cast

import anthropic

from .common import (
    IMAGE_EXTENSIONS,
    append_to_processed_log,
    encode_image,
    get_image_media_type,
    group_images_by_clip,
    load_processed_log,
)
from .models import ClipAnalysis
from .prompts import SYSTEM_PROMPT


def analyze_clip(
    client: anthropic.Anthropic, clip_name: str, frames: list[Path]
) -> ClipAnalysis:
    content = []
    for i, frame_path in enumerate(frames):
        media_type = get_image_media_type(frame_path)
        image_data = encode_image(frame_path)
        content.append(
            {
                "type": "text",
                "text": f"Frame {i}:",
            }
        )
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                },
            }
        )

    content.append(
        {
            "type": "text",
            "text": f"Clip: {clip_name}\n\nAnalyze these {len(frames)} frames and return detections for frames with ball contact.",
        }
    )

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
    skip_empty: bool = False,
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
                reasoning_path.write_text(
                    f"{detection.action.name}: {detection.reasoning}"
                )
                print(f"  Frame {i}: {detection.action.name}")
                detection_count += 1
            elif not skip_empty:
                label_path.write_text("")

        print(f"  Detections: {detection_count}/{len(frames)} frames")
        return detection_count

    except anthropic.RateLimitError:
        print(f"Rate limited on {clip_name}, waiting 60s...")
        time.sleep(60)
        return process_clip(client, clip_name, frames, output_dir, skip_empty)
    except Exception as e:
        print(f"Error processing {clip_name}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Label volleyball clips using Claude Vision"
    )
    parser.add_argument(
        "pattern", help="Glob pattern for images (e.g., './screenshots/*.png')"
    )
    parser.add_argument("--output", "-o", type=Path, default=Path("labels/vision"))
    parser.add_argument(
        "--skip-empty", action="store_true", help="Don't create empty label files"
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    all_images = [
        Path(p)
        for p in glob.glob(args.pattern)
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
        detections = process_clip(
            client, clip_name, frames, output_dir, args.skip_empty
        )
        append_to_processed_log(output_dir, clip_name)
        total_detections += detections

    print(
        f"\nDone! Processed {len(clips_to_process)} clips with {total_detections} total detections"
    )


if __name__ == "__main__":
    main()
