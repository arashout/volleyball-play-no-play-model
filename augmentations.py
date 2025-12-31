import random
import numpy as np
import torch
from torchvision.transforms import v2


def sample_temporal_jitter(
    num_frames: int, total_frames: int, max_stride: int = 2
) -> list[int]:
    stride = random.randint(1, max_stride)
    span = num_frames * stride
    if span > total_frames:
        stride = max(1, total_frames // num_frames)
        span = num_frames * stride
    max_start = max(0, total_frames - span)
    start = random.randint(0, max_start) if max_start > 0 else 0
    return [start + i * stride for i in range(num_frames)]


def get_augmentation_pipeline(p=0.5):
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=p),
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)],
                p=p,
            ),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))], p=p),
            v2.RandomApply(
                [v2.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0))], p=p
            ),
            v2.RandomApply([v2.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=p),
            v2.RandomApply([v2.RandomAffine(degrees=0, shear=(-10, 10, -10, 10))], p=p),
        ]
    )


def augment_video(video: np.ndarray, pipeline=None) -> np.ndarray:
    if pipeline is None:
        pipeline = get_augmentation_pipeline()
    # (T, H, W, C) -> (T, C, H, W) for torchvision
    tensor = torch.from_numpy(video).permute(0, 3, 1, 2)
    augmented = pipeline(tensor)
    # (T, C, H, W) -> (T, H, W, C)
    return augmented.permute(0, 2, 3, 1).numpy()


if __name__ == "__main__":
    import argparse
    import av
    from pathlib import Path
    from utils import read_video_pyav, sample_frame_indices, NUM_FRAMES

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="augmented_preview.mp4", help="Output path")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of augmented versions to generate",
    )
    args = parser.parse_args()

    container = av.open(args.video)
    total_frames = container.streams.video[0].frames
    if total_frames == 0:
        total_frames = sum(1 for _ in container.decode(video=0))
        container.seek(0)

    indices = sample_frame_indices(NUM_FRAMES, 1, total_frames)
    video = read_video_pyav(container, indices)
    container.close()

    fps = 8
    output_path = Path(args.output)
    pipeline = get_augmentation_pipeline(p=0.7)

    sample_aug = augment_video(video, pipeline)
    height, width = sample_aug.shape[1], sample_aug.shape[2]

    output = av.open(str(output_path), mode="w")
    stream = output.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for i in range(args.iterations):
        frames_to_write = augment_video(video, pipeline)

        for frame_data in frames_to_write:
            frame = av.VideoFrame.from_ndarray(
                frame_data.astype(np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                output.mux(packet)

        for _ in range(fps // 2):
            black = np.zeros((height, width, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(black, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)

    for packet in stream.encode():
        output.mux(packet)
    output.close()

    print(f"Preview saved to {output_path}")
    print(f"Contains {args.iterations} augmented versions at {width}x{height}")
