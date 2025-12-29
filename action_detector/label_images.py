import argparse
import base64
import glob
import os
import sys
import time
from pathlib import Path
from typing import cast

import anthropic

from .models import ImageAnalysis
from .prompts import SYSTEM_PROMPT

BATCH_SIZE = 10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_processed_log(output_dir: Path) -> set[str]:
    log_path = output_dir / "processed_log.txt"
    if not log_path.exists():
        return set()
    return set(log_path.read_text().strip().split("\n"))


def append_to_processed_log(output_dir: Path, filenames: list[str]):
    log_path = output_dir / "processed_log.txt"
    with open(log_path, "a") as f:
        for name in filenames:
            f.write(f"{name}\n")


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


def analyze_image(client: anthropic.Anthropic, image_path: Path) -> ImageAnalysis:
    media_type = get_image_media_type(image_path)
    image_data = encode_image(image_path)

    messages = cast(
        list,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Filename: {image_path.name}\n\nAnalyze this volleyball image and return the detections.",
                    },
                ],
            }
        ],
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=[
            {
                "name": "report_detections",
                "description": "Report the detected volleyball actions in the image",
                "input_schema": ImageAnalysis.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": "report_detections"},
    )

    for block in message.content:
        if block.type == "tool_use" and block.name == "report_detections":
            return ImageAnalysis.model_validate(block.input)

    return ImageAnalysis(detections=[])


def process_batch(
    client: anthropic.Anthropic,
    images: list[Path],
    output_dir: Path,
) -> tuple[list[str], int]:
    processed = []
    total_detections = 0

    for image_path in images:
        try:
            result = analyze_image(client, image_path)
            label_path = output_dir / f"{image_path.stem}.txt"
            label_path.write_text(result.to_yolo_format())
            if result.detections:
                reasoning_path = output_dir / f"{image_path.stem}_reasoning.txt"
                reasoning_path.write_text(result.to_reasoning_format())
            print(f"\n--- {image_path.name} ---")
            if result.detections:
                print(f"Labels:\n{result.to_yolo_format()}")
                print(f"Reasoning:\n{result.to_reasoning_format()}")
            else:
                print("No detections")
            processed.append(image_path.name)
            total_detections += len(result.detections)
        except anthropic.RateLimitError:
            print("Rate limited, waiting 60s...")
            time.sleep(60)
            try:
                result = analyze_image(client, image_path)
                label_path = output_dir / f"{image_path.stem}.txt"
                label_path.write_text(result.to_yolo_format())
                if result.detections:
                    reasoning_path = output_dir / f"{image_path.stem}_reasoning.txt"
                    reasoning_path.write_text(result.to_reasoning_format())
                print(f"\n--- {image_path.name} ---")
                if result.detections:
                    print(f"Labels:\n{result.to_yolo_format()}")
                    print(f"Reasoning:\n{result.to_reasoning_format()}")
                else:
                    print("No detections")
                processed.append(image_path.name)
                total_detections += len(result.detections)
            except Exception as e:
                print(f"Error processing {image_path.name} after retry: {e}")
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")

    return processed, total_detections


def main():
    parser = argparse.ArgumentParser(description="Label volleyball images using Claude Vision")
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

    processed_log = load_processed_log(output_dir)
    images_to_process = [p for p in all_images if p.name not in processed_log]

    print(f"Found {len(all_images)} images matching pattern, {len(images_to_process)} remaining to process")

    total_processed = 0
    total_detections = 0

    for i in range(0, len(images_to_process), BATCH_SIZE):
        batch = images_to_process[i : i + BATCH_SIZE]
        processed, detections = process_batch(client, batch, output_dir)
        append_to_processed_log(output_dir, processed)
        total_processed += len(processed)
        total_detections += detections
        print(f"Processed {total_processed}/{len(images_to_process)} images, {total_detections} detections so far")

    print(f"\nDone! Processed {total_processed} images with {total_detections} total detections")


if __name__ == "__main__":
    main()
