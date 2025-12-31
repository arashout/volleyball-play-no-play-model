#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import normalize_video, NUM_FRAMES


def extract_frames_at_time(video_path, start_sec, end_sec, num_frames=NUM_FRAMES):
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if len(frames) < num_frames:
        return None

    indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    sampled = np.stack([frames[i] for i in indices])
    return sampled


def resize_frames(frames, size=224):
    import cv2

    resized = []
    for frame in frames:
        h, w = frame.shape[:2]
        scale = max(size / h, size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame = cv2.resize(frame, (new_w, new_h))
        start_h = (new_h - size) // 2
        start_w = (new_w - size) // 2
        frame = frame[start_h : start_h + size, start_w : start_w + size]
        resized.append(frame)
    return np.stack(resized)


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def process_video(video_path, onnx_path, window_seconds=3.0, stride_seconds=1.0):
    import cv2

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    results = []
    current_time = 0.0

    print(f"Video: {duration:.1f}s @ {fps:.1f}fps")
    print(f"Window: {window_seconds}s, Stride: {stride_seconds}s")
    print()

    while current_time + window_seconds <= duration:
        start_time = current_time
        end_time = current_time + window_seconds

        frames = extract_frames_at_time(video_path, start_time, end_time)
        if frames is None:
            current_time += stride_seconds
            continue

        frames = resize_frames(frames)
        frames = normalize_video(frames)

        input_tensor = frames.transpose(0, 3, 1, 2)[np.newaxis].astype(np.float32)

        logits = session.run(None, {"pixel_values": input_tensor})[0]
        pred = int(np.argmax(logits, axis=-1)[0])
        confidence = float(np.exp(logits[0][pred]) / np.sum(np.exp(logits[0])))

        label = "play" if pred == 1 else "no-play"
        results.append(
            {
                "start": start_time,
                "end": end_time,
                "label": label,
                "confidence": confidence,
            }
        )

        print(
            f"{format_time(start_time)} - {format_time(end_time)}: {label} ({confidence:.2f})"
        )
        current_time += stride_seconds

    return results


def merge_segments(results, min_duration=1.0):
    if not results:
        return []

    merged = []
    current = {
        "start": results[0]["start"],
        "end": results[0]["end"],
        "label": results[0]["label"],
    }

    for r in results[1:]:
        if r["label"] == current["label"]:
            current["end"] = r["end"]
        else:
            if current["end"] - current["start"] >= min_duration:
                merged.append(current)
            current = {"start": r["start"], "end": r["end"], "label": r["label"]}

    if current["end"] - current["start"] >= min_duration:
        merged.append(current)

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Input video path")
    parser.add_argument(
        "--onnx", default="best_model/model.onnx", help="ONNX model path"
    )
    parser.add_argument(
        "--window", type=float, default=3.0, help="Window size in seconds"
    )
    parser.add_argument("--stride", type=float, default=1.0, help="Stride in seconds")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument(
        "--merge", action="store_true", help="Merge consecutive same-label segments"
    )
    args = parser.parse_args()

    results = process_video(args.video, args.onnx, args.window, args.stride)

    if args.merge:
        results = merge_segments(results)
        print("\nMerged segments:")
        for r in results:
            print(f"{format_time(r['start'])} - {format_time(r['end'])}: {r['label']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
