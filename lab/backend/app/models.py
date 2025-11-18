from pydantic import BaseModel
from typing import Literal, Optional


class StartMessage(BaseModel):
    action: Literal["start"]
    mode: Literal["single", "double"] = "single"
    controller: Literal["pid", "nn"] = "pid"
    dt: float = 0.02
    target: float = 0.0
    # Optional track length (total length of cart travel). If provided,
    # the simulator will clamp cart x to [-track_length/2, track_length/2].
    track_length: Optional[float] = None
    # Optional initial PID gains to apply during start
    kp: Optional[float] = None
    ki: Optional[float] = None
    kd: Optional[float] = None


class ControlMessage(BaseModel):
    action: Literal["control"]
    type: Literal["pid", "nn"]
    params: Optional[dict] = None


class TrackMessage(BaseModel):
    action: Literal["set_track"]
    track_length: float
