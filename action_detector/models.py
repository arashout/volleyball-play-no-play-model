from enum import IntEnum
from pydantic import BaseModel


class ActionType(IntEnum):
    SERVE = 0
    RECEIVE = 1
    SET = 2
    SPIKE = 3
    BLOCK = 4
    DIG = 5


class Detection(BaseModel):
    action: ActionType
    x_center: float
    y_center: float
    width: float
    height: float
    reasoning: str

    def to_yolo_line(self) -> str:
        return f"{self.action.value} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


class ImageAnalysis(BaseModel):
    detections: list[Detection]

    def to_yolo_format(self) -> str:
        return "\n".join(d.to_yolo_line() for d in self.detections)

    def to_reasoning_format(self) -> str:
        lines = []
        for i, d in enumerate(self.detections):
            lines.append(f"[{i}] {d.action.name}: {d.reasoning}")
        return "\n".join(lines)
