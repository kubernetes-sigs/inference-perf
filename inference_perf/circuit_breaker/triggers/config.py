from pydantic import BaseModel, Field
from typing import Literal, Union


class TriggerConsecutive(BaseModel):
    type: Literal["consecutive"]
    threshold: int = Field(..., ge=1)


class TriggerRateOverWindow(BaseModel):
    type: Literal["rate_over_window"]
    window_sec: float = Field(..., gt=0.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    min_samples: int = Field(0, ge=0)


TriggerSpec = Union[
    TriggerConsecutive,
    TriggerRateOverWindow,
]
