import argparse
import json
import cv2
import numpy as np
import onnxruntime as ort
import threading
import queue
from pathlib import Path
from tqdm import tqdm
from infer_onnx import preprocess_frame, LABELS
from utils import NUM_FRAMES
from smooth_predictions import smooth_predictions


def frame_reader(cap, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_frame(frame_rgb)
        frame_queue.put(preprocessed)


def generate_predictions(model_path: str, video_path: str, output_path: str, smooth: bool = False):
    session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    window_frames = NUM_FRAMES
    # we overlap windows by half
    stride_frames = NUM_FRAMES // 2
    window_duration = window_frames / fps

    predictions = []
    frame_buffer = []
    frame_idx = 0
    window_start_frame = 0

    frame_queue = queue.Queue(maxsize=64)
    stop_event = threading.Event()
    reader_thread = threading.Thread(
        target=frame_reader, args=(cap, frame_queue, stop_event), daemon=True
    )
    reader_thread.start()

    pbar = tqdm(total=total_frames, desc="Analyzing")

    while True:
        preprocessed = frame_queue.get()
        if preprocessed is None:
            break

        frame_buffer.append(preprocessed)
        frame_idx += 1

        if len(frame_buffer) >= NUM_FRAMES:
            indices = np.linspace(0, len(frame_buffer) - 1, NUM_FRAMES, dtype=int)
            sampled = [frame_buffer[i] for i in indices]
            pixel_values = np.stack(sampled)[np.newaxis, ...].astype(np.float32)
            outputs = session.run(None, {"pixel_values": pixel_values})
            logits = np.asarray(outputs[0])
            pred_idx = int(np.argmax(logits, axis=-1)[0])
            confidence = float(np.exp(logits[0, pred_idx]) / np.exp(logits[0]).sum())

            start_time = window_start_frame / fps
            end_time = start_time + window_duration

            label = LABELS[pred_idx]
            predictions.append({
                "startTime": round(start_time, 3),
                "endTime": round(end_time, 3),
                "label": "no-play" if label == "no_play" else "play",
                "confidence": round(confidence, 4)
            })

            frame_buffer = frame_buffer[stride_frames:]
            window_start_frame += stride_frames

        pbar.update(1)

    stop_event.set()
    pbar.close()
    cap.release()

    if smooth:
        smoothed = smooth_predictions(predictions)
        output = {"playNoPlayPredictions": smoothed}
        print(f"Generated {len(predictions)} raw -> {len(smoothed)} smoothed predictions")
    else:
        output = {"playNoPlayPredictions": predictions}
        print(f"Generated {len(predictions)} predictions")

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--model-path", default="best_model/model.onnx")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--smooth", action="store_true", help="Apply smoothing pipeline")
    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        p = Path(args.video_path)
        output_path = str(p.parent / f"{p.stem}_predictions.json")

    generate_predictions(args.model_path, args.video_path, output_path, smooth=args.smooth)
