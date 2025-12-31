#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import normalize_video, NUM_FRAMES


def resize_frame(frame, size=224):
    h, w = frame.shape[:2]
    scale = max(size / h, size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    frame = cv2.resize(frame, (new_w, new_h))
    start_h = (new_h - size) // 2
    start_w = (new_w - size) // 2
    return frame[start_h : start_h + size, start_w : start_w + size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Input video path")
    parser.add_argument(
        "--onnx", default="best_model/model.onnx", help="ONNX model path"
    )
    parser.add_argument(
        "--interval", type=int, default=15, help="Frames between predictions"
    )
    args = parser.parse_args()

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_buffer = deque(maxlen=NUM_FRAMES * 3)
    frame_count = 0
    current_label = "..."
    current_conf = 0.0

    print(f"Playing video @ {fps:.1f}fps. Press 'q' to quit, space to pause.")

    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1

            if frame_count % args.interval == 0 and len(frame_buffer) >= NUM_FRAMES:
                indices = np.linspace(0, len(frame_buffer) - 1, NUM_FRAMES).astype(int)
                sampled = [resize_frame(frame_buffer[i]) for i in indices]
                sampled = np.stack(sampled)
                sampled = normalize_video(sampled)

                input_tensor = sampled.transpose(0, 3, 1, 2)[np.newaxis].astype(
                    np.float32
                )
                logits = session.run(None, {"pixel_values": input_tensor})[0]
                pred = int(np.argmax(logits, axis=-1)[0])
                current_conf = float(
                    np.exp(logits[0][pred]) / np.sum(np.exp(logits[0]))
                )
                current_label = "PLAY" if pred == 1 else "NO PLAY"

        display = frame.copy()
        h, w = display.shape[:2]

        color = (0, 255, 0) if current_label == "PLAY" else (0, 0, 255)
        cv2.rectangle(display, (10, 10), (250, 80), (0, 0, 0), -1)
        cv2.rectangle(display, (10, 10), (250, 80), color, 3)
        cv2.putText(
            display, current_label, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3
        )
        cv2.putText(
            display,
            f"{current_conf:.0%}",
            (180, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        time_sec = frame_count / fps
        time_str = f"{int(time_sec // 60):02d}:{time_sec % 60:05.2f}"
        cv2.putText(
            display,
            time_str,
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        progress = int((frame_count / total_frames) * (w - 20))
        cv2.rectangle(display, (10, h - 10), (10 + progress, h - 5), (0, 255, 0), -1)

        cv2.imshow("Volleyball Play Detection", display)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
