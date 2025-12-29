import argparse
import glob
import sys
from pathlib import Path

import cv2

from .models import ActionType

COLORS = {
    ActionType.SERVE: (255, 0, 0),
    ActionType.RECEIVE: (0, 255, 0),
    ActionType.SET: (0, 255, 255),
    ActionType.SPIKE: (0, 0, 255),
    ActionType.BLOCK: (255, 0, 255),
    ActionType.DIG: (255, 255, 0),
}


def load_yolo_labels(label_path: Path) -> list[tuple[ActionType, float, float, float, float]]:
    if not label_path.exists():
        return []
    labels = []
    for line in label_path.read_text().strip().split("\n"):
        if not line:
            continue
        parts = line.split()
        action = ActionType(int(parts[0]))
        x_center, y_center, width, height = map(float, parts[1:5])
        labels.append((action, x_center, y_center, width, height))
    return labels


def draw_labels(image, labels: list[tuple[ActionType, float, float, float, float]]):
    h, w = image.shape[:2]
    for action, x_center, y_center, width, height in labels:
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        color = COLORS.get(action, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = action.name
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image


def main():
    parser = argparse.ArgumentParser(description="View images with YOLO labels")
    parser.add_argument("pattern", help="Glob pattern for images")
    parser.add_argument("--labels", "-l", type=Path, default=Path("labels/pose_guided"))
    args = parser.parse_args()

    images = sorted(glob.glob(args.pattern))
    if not images:
        print(f"No images found matching: {args.pattern}")
        sys.exit(1)

    print(f"Found {len(images)} images. Controls: [n]ext, [p]rev, [q]uit")

    idx = 0
    while True:
        image_path = Path(images[idx])
        label_path = args.labels / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load: {image_path}")
            idx = (idx + 1) % len(images)
            continue

        labels = load_yolo_labels(label_path)
        image = draw_labels(image, labels)

        window_title = f"{image_path.name} ({idx + 1}/{len(images)}) - {len(labels)} detections"
        cv2.imshow(window_title, image)

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord("q"):
            break
        elif key == ord("n") or key == ord(" "):
            idx = (idx + 1) % len(images)
        elif key == ord("p"):
            idx = (idx - 1) % len(images)


if __name__ == "__main__":
    main()
