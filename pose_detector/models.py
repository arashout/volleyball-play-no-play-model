from enum import IntEnum
from pydantic import BaseModel


class KeypointName(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class Keypoint(BaseModel):
    name: KeypointName
    x: float
    y: float
    confidence: float


class PoseDetection(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    keypoints: list[Keypoint]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class FramePoses(BaseModel):
    frame_path: str
    width: int
    height: int
    poses: list[PoseDetection]
