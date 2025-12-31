import argparse
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from utils import NUM_FRAMES

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
LABELS = ["no_play", "play"]


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    img = Image.fromarray(frame)
    w, h = img.size
    short_side = min(w, h)
    scale = 224 / short_side
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)


def run_inference(session, frames):
    processed = np.stack([preprocess_frame(f) for f in frames])
    pixel_values = processed[np.newaxis, ...].astype(np.float32)
    outputs = session.run(None, {"pixel_values": pixel_values})
    logits = outputs[0]
    pred_idx = int(np.argmax(logits, axis=-1)[0])
    confidence = float(np.exp(logits[0, pred_idx]) / np.exp(logits[0]).sum())
    return LABELS[pred_idx], confidence


def play_video_with_labels(model_path: str, video_path: str):
    session = ort.InferenceSession(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)

    frame_buffer = []
    current_label = "..."
    current_conf = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(frame_rgb)

        if len(frame_buffer) >= NUM_FRAMES:
            indices = np.linspace(0, len(frame_buffer) - 1, NUM_FRAMES, dtype=int)
            sampled = [frame_buffer[i] for i in indices]
            current_label, current_conf = run_inference(session, sampled)
            frame_buffer = frame_buffer[NUM_FRAMES // 2 :]

        color = (0, 255, 0) if current_label == "play" else (0, 0, 255)
        text = f"{current_label} ({current_conf:.1%})"
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.imshow("Volleyball Play Detection", frame)
        key = cv2.waitKey(delay)
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--model-path", default="output/model.onnx")
    args = parser.parse_args()
    play_video_with_labels(args.model_path, args.video_path)
