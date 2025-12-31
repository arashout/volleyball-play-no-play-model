import numpy as np

NUM_FRAMES = 16

# ImageNet normalization values used by VideoMAE
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def normalize_video(video):
    """
    Normalize video frames using ImageNet mean and std.
    """
    # Convert to float and scale to [0, 1]
    video = video.astype(np.float32) / 255.0

    # Normalize using ImageNet mean and std
    video = (video - IMAGENET_MEAN) / IMAGENET_STD

    return video


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if seg_len < converted_len:
        converted_len = seg_len
        frame_sample_rate = seg_len // clip_len
    end_idx = seg_len
    start_idx = max(0, end_idx - converted_len)
    indices = np.linspace(start_idx, end_idx - 1, num=clip_len)
    indices = np.clip(indices, 0, seg_len - 1).astype(np.int64)
    return indices
