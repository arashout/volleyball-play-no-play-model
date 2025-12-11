import numpy as np

NUM_FRAMES = 16


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
