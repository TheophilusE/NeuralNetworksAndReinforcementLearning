from pydantic import BaseModel
from typing import Literal, Optional


class StartMessage(BaseModel):
    action: Literal["start"]
    mode: Literal["single", "double"] = "single"
    controller: Literal["pid", "nn"] = "pid"
    dt: float = 0.02
    target: float = 0.0


class ControlMessage(BaseModel):
    action: Literal["control"]
    type: Literal["pid", "nn"]
    params: Optional[dict] = None
