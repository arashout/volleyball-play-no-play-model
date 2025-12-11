import av
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from utils import NUM_FRAMES, read_video_pyav, sample_frame_indices


def predict(video_path: str, model_path: str):
    processor = VideoMAEImageProcessor.from_pretrained(model_path)
    model = VideoMAEForVideoClassification.from_pretrained(model_path)
    model.eval()

    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    if total_frames == 0:
        total_frames = sum(1 for _ in container.decode(video=0))
        container.seek(0)

    indices = sample_frame_indices(NUM_FRAMES, 1, total_frames)
    video = read_video_pyav(container, indices)
    container.close()

    inputs = processor(list(video), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()

    id_map = model.config.id2label
    if id_map is None:
        raise ValueError("Model does not have id2label mapping.")
    return id_map[predicted_class]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict play/no-play from a volleyball video")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument(
        "--model-path",
        default="reddbeann/finetuned-volleyball-classification",
        help="Path to the model (default: reddbeann/finetuned-volleyball-classification)",
    )
    args = parser.parse_args()

    result = predict(args.video_path, args.model_path)
    print(f"Prediction: {result}")
