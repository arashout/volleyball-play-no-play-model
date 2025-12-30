from pathlib import Path
from ultralytics import YOLO


def export_pose_model(model_path: str = "models/yolo11n-pose.pt", output_dir: str = "output"):
    model = YOLO(model_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.export(
        format="onnx",
        opset=18,
        simplify=True,
        dynamic=False,
        imgsz=640,
    )

    exported_path = Path(model_path).with_suffix(".onnx")
    target_path = output_path / "yolo11n-pose.onnx"

    if exported_path.exists():
        exported_path.rename(target_path)
        print(f"Exported to: {target_path}")

    return target_path


if __name__ == "__main__":
    export_pose_model()
