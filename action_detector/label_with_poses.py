import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import cast

import anthropic

from pose_detector import PoseDetector, FramePoses
from .common import (
    IMAGE_EXTENSIONS,
    append_to_processed_log,
    encode_image,
    get_image_media_type,
    group_images_by_clip,
    load_processed_log,
)
from .models import PoseGuidedClipAnalysis, ActionType
from .prompts import POSE_GUIDED_PROMPT


def format_player_boxes(frame_poses: FramePoses) -> str:
    if not frame_poses.poses:
        return "No players detected"

    lines = []
    for i, pose in enumerate(frame_poses.poses):
        x_center = (pose.x1 + pose.x2) / 2 / frame_poses.width
        y_center = (pose.y1 + pose.y2) / 2 / frame_poses.height
        width = (pose.x2 - pose.x1) / frame_poses.width
        height = (pose.y2 - pose.y1) / frame_poses.height
        lines.append(
            f"  Player {i}: center=({x_center:.2f}, {y_center:.2f}), "
            f"size=({width:.2f}, {height:.2f}), conf={pose.confidence:.2f}"
        )
    return "\n".join(lines)


def analyze_clip_with_poses(
    client: anthropic.Anthropic,
    clip_name: str,
    frames: list[Path],
    pose_results: list[FramePoses],
) -> PoseGuidedClipAnalysis:
    content = []

    for i, (frame_path, frame_poses) in enumerate(zip(frames, pose_results)):
        media_type = get_image_media_type(frame_path)
        image_data = encode_image(frame_path)

        player_info = format_player_boxes(frame_poses)
        content.append(
            {
                "type": "text",
                "text": f"Frame {i}:\nDetected players:\n{player_info}",
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
            "text": f"Clip: {clip_name}\n\nAnalyze these {len(frames)} frames and identify which player (by index) is performing an action, if any.",
        }
    )

    messages = cast(list, [{"role": "user", "content": content}])

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=POSE_GUIDED_PROMPT,
        messages=messages,
        tools=[
            {
                "name": "report_detections",
                "description": "Report the detected volleyball actions with player indices",
                "input_schema": PoseGuidedClipAnalysis.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": "report_detections"},
    )

    for block in message.content:
        if block.type == "tool_use" and block.name == "report_detections":
            return PoseGuidedClipAnalysis.model_validate(block.input)

    return PoseGuidedClipAnalysis(frames=[])


def process_clip(
    client: anthropic.Anthropic,
    detector: PoseDetector,
    clip_name: str,
    frames: list[Path],
    output_dir: Path,
    skip_empty: bool = False,
) -> int:
    try:
        pose_results = [detector.detect_file(str(f)) for f in frames]
        result = analyze_clip_with_poses(client, clip_name, frames, pose_results)

        print(f"\n=== {clip_name} ({len(frames)} frames) ===")

        detection_count = 0
        for frame_result in result.frames:
            frame_idx = frame_result.frame_index
            if frame_idx >= len(frames):
                continue

            frame_path = frames[frame_idx]
            frame_poses = pose_results[frame_idx]
            label_path = output_dir / f"{frame_path.stem}.txt"

            if frame_result.detection:
                det = frame_result.detection
                player_idx = det.player_index

                if player_idx < len(frame_poses.poses):
                    pose = frame_poses.poses[player_idx]
                    x_center = (pose.x1 + pose.x2) / 2 / frame_poses.width
                    y_center = (pose.y1 + pose.y2) / 2 / frame_poses.height
                    width = (pose.x2 - pose.x1) / frame_poses.width
                    height = (pose.y2 - pose.y1) / frame_poses.height

                    yolo_line = f"{det.action.value} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    label_path.write_text(yolo_line)

                    reasoning_path = output_dir / f"{frame_path.stem}_reasoning.txt"
                    reasoning_path.write_text(
                        f"{ActionType(det.action).name}: {det.reasoning}"
                    )

                    print(
                        f"  Frame {frame_idx}: {ActionType(det.action).name} (player {player_idx})"
                    )
                    detection_count += 1
                elif not skip_empty:
                    label_path.write_text("")
            elif not skip_empty:
                label_path.write_text("")

        print(f"  Detections: {detection_count}/{len(frames)} frames")
        return detection_count

    except anthropic.RateLimitError:
        print(f"Rate limited on {clip_name}, waiting 60s...")
        time.sleep(60)
        return process_clip(client, detector, clip_name, frames, output_dir, skip_empty)
    except Exception as e:
        print(f"Error processing {clip_name}: {e}")
        import traceback

        traceback.print_exc()
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", help="Glob pattern for images")
    parser.add_argument("--output", "-o", type=Path, default=Path("labels/pose_guided"))
    parser.add_argument("--pose-model", default="output/yolo11n-pose.onnx")
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
    detector = PoseDetector(args.pose_model)

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
            client, detector, clip_name, frames, output_dir, args.skip_empty
        )
        append_to_processed_log(output_dir, clip_name)
        total_detections += detections

    print(
        f"\nDone! Processed {len(clips_to_process)} clips with {total_detections} total detections"
    )


if __name__ == "__main__":
    main()
