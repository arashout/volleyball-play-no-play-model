import numpy as np
from typing import List, Dict, Tuple


def predictions_to_signal(
    predictions: List[Dict], sample_rate: float = 30.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert overlapping window predictions to a continuous confidence signal.
    Returns (timestamps, play_confidence) arrays.
    """
    if not predictions:
        return np.array([]), np.array([])

    end_time = max(p["endTime"] for p in predictions)
    num_samples = int(end_time * sample_rate) + 1
    timestamps = np.linspace(0, end_time, num_samples)

    weights = np.zeros(num_samples)
    weighted_conf = np.zeros(num_samples)

    for pred in predictions:
        start_idx = int(pred["startTime"] * sample_rate)
        end_idx = int(pred["endTime"] * sample_rate)
        conf = pred["confidence"] if pred["label"] == "play" else 1 - pred["confidence"]

        for i in range(start_idx, min(end_idx + 1, num_samples)):
            weights[i] += 1
            weighted_conf[i] += conf

    mask = weights > 0
    play_confidence = np.zeros(num_samples)
    play_confidence[mask] = weighted_conf[mask] / weights[mask]

    return timestamps, play_confidence


def apply_smoothing(
    confidence: np.ndarray, window_seconds: float = 4.0, sample_rate: float = 30.0
) -> np.ndarray:
    """
    Apply rolling average smoothing to the confidence signal.
    """
    window_size = int(window_seconds * sample_rate)
    if window_size < 1:
        return confidence.copy()

    kernel = np.ones(window_size) / window_size
    padded = np.pad(confidence, window_size // 2, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")

    return smoothed[window_size // 2 : window_size // 2 + len(confidence)]


def apply_hysteresis(
    confidence: np.ndarray, high_thresh: float = 0.6, low_thresh: float = 0.4
) -> np.ndarray:
    """
    Apply hysteresis thresholding to prevent oscillation.
    Returns binary array (1 = play, 0 = no-play).
    """
    labels = np.zeros(len(confidence), dtype=np.int32)
    if len(confidence) == 0:
        return labels

    state = 1 if confidence[0] >= (high_thresh + low_thresh) / 2 else 0
    labels[0] = state

    for i in range(1, len(confidence)):
        if state == 0 and confidence[i] >= high_thresh:
            state = 1
        elif state == 1 and confidence[i] <= low_thresh:
            state = 0
        labels[i] = state

    return labels


def enforce_min_duration(
    labels: np.ndarray, min_seconds: float = 8.0, sample_rate: float = 30.0
) -> np.ndarray:
    """
    Merge segments shorter than min_duration into surrounding segments.
    """
    if len(labels) == 0:
        return labels.copy()

    min_samples = int(min_seconds * sample_rate)
    result = labels.copy()

    segments = []
    start = 0
    for i in range(1, len(result)):
        if result[i] != result[i - 1]:
            segments.append((start, i, result[start]))
            start = i
    segments.append((start, len(result), result[start]))

    for start, end, label in segments:
        duration = end - start
        if duration < min_samples:
            if start == 0:
                new_label = segments[1][2] if len(segments) > 1 else label
            elif end == len(result):
                new_label = segments[-2][2] if len(segments) > 1 else label
            else:
                new_label = 1 - label
            result[start:end] = new_label

    return result


def labels_to_segments(
    labels: np.ndarray, timestamps: np.ndarray
) -> List[Dict]:
    """
    Convert binary label array back to segment list.
    """
    if len(labels) == 0:
        return []

    segments = []
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segments.append({
                "startTime": round(float(timestamps[start_idx]), 3),
                "endTime": round(float(timestamps[i]), 3),
                "label": "play" if labels[start_idx] == 1 else "no-play"
            })
            start_idx = i

    segments.append({
        "startTime": round(float(timestamps[start_idx]), 3),
        "endTime": round(float(timestamps[-1]), 3),
        "label": "play" if labels[start_idx] == 1 else "no-play"
    })

    return segments


def smooth_predictions(
    predictions: List[Dict],
    sample_rate: float = 30.0,
    smoothing_window: float = 4.0,
    high_thresh: float = 0.6,
    low_thresh: float = 0.4,
    min_segment_duration: float = 8.0
) -> List[Dict]:
    """
    Full pipeline: raw predictions -> smoothed segments.
    """
    timestamps, confidence = predictions_to_signal(predictions, sample_rate)
    smoothed = apply_smoothing(confidence, smoothing_window, sample_rate)
    labels = apply_hysteresis(smoothed, high_thresh, low_thresh)
    labels = enforce_min_duration(labels, min_segment_duration, sample_rate)
    segments = labels_to_segments(labels, timestamps)
    return segments
