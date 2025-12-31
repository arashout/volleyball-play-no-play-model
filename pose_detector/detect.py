import glob
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .models import FramePoses, Keypoint, KeypointName, PoseDetection


class PoseDetector:
    def __init__(self, model_path: str = "output/yolo11n-pose.onnx"):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.img_size = self.input_shape[2]

    def preprocess(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        h, w = image.shape[:2]
        scale = self.img_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (self.img_size - new_w) // 2, (self.img_size - new_h) // 2
        padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        return blob, scale, (pad_x, pad_y)

    def postprocess(
        self,
        output: np.ndarray,
        scale: float,
        padding: tuple[int, int],
        orig_shape: tuple[int, int],
        conf_threshold: float = 0.25,
    ) -> list[PoseDetection]:
        # output shape: (1, 56, 8400) -> 4 box + 1 conf + 51 keypoints (17*3)
        predictions = output[0].T

        detections = []
        for pred in predictions:
            conf = pred[4]
            if conf < conf_threshold:
                continue

            cx, cy, w, h = pred[:4]
            pad_x, pad_y = padding

            x1 = (cx - w / 2 - pad_x) / scale
            y1 = (cy - h / 2 - pad_y) / scale
            x2 = (cx + w / 2 - pad_x) / scale
            y2 = (cy + h / 2 - pad_y) / scale

            x1 = max(0, min(x1, orig_shape[1]))
            y1 = max(0, min(y1, orig_shape[0]))
            x2 = max(0, min(x2, orig_shape[1]))
            y2 = max(0, min(y2, orig_shape[0]))

            keypoints = []
            kp_data = pred[5:].reshape(17, 3)
            for i, (kx, ky, kc) in enumerate(kp_data):
                kx = (kx - pad_x) / scale
                ky = (ky - pad_y) / scale
                keypoints.append(
                    Keypoint(name=KeypointName(i), x=kx, y=ky, confidence=kc)
                )

            detections.append(
                PoseDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    keypoints=keypoints,
                )
            )

        # NMS
        if len(detections) > 1:
            detections = self._nms(detections, iou_threshold=0.45)

        return detections

    def _nms(
        self, detections: list[PoseDetection], iou_threshold: float
    ) -> list[PoseDetection]:
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [d for d in detections if self._iou(best, d) < iou_threshold]
        return keep

    def _iou(self, a: PoseDetection, b: PoseDetection) -> float:
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = a.width * a.height
        area_b = b.width * b.height
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0

    def detect(self, image: np.ndarray) -> list[PoseDetection]:
        blob, scale, padding = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})
        return self.postprocess(outputs[0], scale, padding, image.shape[:2])

    def detect_file(self, image_path: str) -> FramePoses:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        poses = self.detect(image)
        return FramePoses(
            frame_path=image_path,
            width=image.shape[1],
            height=image.shape[0],
            poses=poses,
        )

    def detect_batch(self, pattern: str) -> list[FramePoses]:
        results = []
        for path in sorted(glob.glob(pattern)):
            if Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                try:
                    results.append(self.detect_file(path))
                except Exception as e:
                    print(f"Error processing {path}: {e}")
        return results
