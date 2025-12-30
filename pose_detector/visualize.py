import argparse
from pathlib import Path

import cv2
import numpy as np

from .detect import PoseDetector
from .models import FramePoses, KeypointName

SKELETON = [
    (KeypointName.NOSE, KeypointName.LEFT_EYE),
    (KeypointName.NOSE, KeypointName.RIGHT_EYE),
    (KeypointName.LEFT_EYE, KeypointName.LEFT_EAR),
    (KeypointName.RIGHT_EYE, KeypointName.RIGHT_EAR),
    (KeypointName.LEFT_SHOULDER, KeypointName.RIGHT_SHOULDER),
    (KeypointName.LEFT_SHOULDER, KeypointName.LEFT_ELBOW),
    (KeypointName.RIGHT_SHOULDER, KeypointName.RIGHT_ELBOW),
    (KeypointName.LEFT_ELBOW, KeypointName.LEFT_WRIST),
    (KeypointName.RIGHT_ELBOW, KeypointName.RIGHT_WRIST),
    (KeypointName.LEFT_SHOULDER, KeypointName.LEFT_HIP),
    (KeypointName.RIGHT_SHOULDER, KeypointName.RIGHT_HIP),
    (KeypointName.LEFT_HIP, KeypointName.RIGHT_HIP),
    (KeypointName.LEFT_HIP, KeypointName.LEFT_KNEE),
    (KeypointName.RIGHT_HIP, KeypointName.RIGHT_KNEE),
    (KeypointName.LEFT_KNEE, KeypointName.LEFT_ANKLE),
    (KeypointName.RIGHT_KNEE, KeypointName.RIGHT_ANKLE),
]

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def draw_poses(image: np.ndarray, frame_poses: FramePoses) -> np.ndarray:
    img = image.copy()
    for i, pose in enumerate(frame_poses.poses):
        color = COLORS[i % len(COLORS)]

        cv2.rectangle(
            img,
            (int(pose.x1), int(pose.y1)),
            (int(pose.x2), int(pose.y2)),
            color,
            2,
        )

        cv2.putText(
            img,
            f"{pose.confidence:.2f}",
            (int(pose.x1), int(pose.y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        kp_dict = {kp.name: kp for kp in pose.keypoints}
        for start, end in SKELETON:
            kp1, kp2 = kp_dict.get(start), kp_dict.get(end)
            if kp1 and kp2 and kp1.confidence > 0.5 and kp2.confidence > 0.5:
                cv2.line(
                    img,
                    (int(kp1.x), int(kp1.y)),
                    (int(kp2.x), int(kp2.y)),
                    color,
                    2,
                )

        for kp in pose.keypoints:
            if kp.confidence > 0.5:
                cv2.circle(img, (int(kp.x), int(kp.y)), 4, color, -1)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", help="Glob pattern for images")
    parser.add_argument("--model", default="output/yolo11n-pose.onnx")
    parser.add_argument("--output", "-o", type=Path, help="Output dir for annotated images")
    args = parser.parse_args()

    detector = PoseDetector(args.model)
    results = detector.detect_batch(args.pattern)

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for frame_poses in results:
            image = cv2.imread(frame_poses.frame_path)
            annotated = draw_poses(image, frame_poses)
            out_path = args.output / Path(frame_poses.frame_path).name
            cv2.imwrite(str(out_path), annotated)
            print(f"Saved: {out_path}")
    else:
        for i, frame_poses in enumerate(results):
            image = cv2.imread(frame_poses.frame_path)
            annotated = draw_poses(image, frame_poses)

            cv2.imshow("Pose Detection", annotated)
            print(f"[{i+1}/{len(results)}] {frame_poses.frame_path}: {len(frame_poses.poses)} poses")

            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
