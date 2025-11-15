from pydantic import BaseModel
from typing import Literal, Optional


class StartMessage(BaseModel):
    action: Literal["start"]
    mode: Literal["single", "double"] = "single"
    controller: Literal["pid", "nn"] = "pid"
    dt: float = 0.02
    target: float = 0.0
    # Optional initial PID gains to apply during start
    kp: Optional[float] = None
    ki: Optional[float] = None
    kd: Optional[float] = None


class ControlMessage(BaseModel):
    action: Literal["control"]
    type: Literal["pid", "nn"]
    params: Optional[dict] = None
