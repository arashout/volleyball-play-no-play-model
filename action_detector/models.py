from enum import IntEnum
from pydantic import BaseModel


class ActionType(IntEnum):
    SERVE = 0
    RECEIVE = 1
    SET = 2
    SPIKE = 3
    BLOCK = 4
    DIG = 5


class FrameDetection(BaseModel):
    action: ActionType
    x_center: float
    y_center: float
    width: float
    height: float
    reasoning: str

    def to_yolo_line(self) -> str:
        return f"{self.action.value} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


class ClipAnalysis(BaseModel):
    frames: list[FrameDetection | None]
