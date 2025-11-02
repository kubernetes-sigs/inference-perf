from pydantic import BaseModel, Field
from typing import List
from .triggers.config import TriggerSpec


class MetricsSpec(BaseModel):
    """
    Manage matches and rules to select target metrics.
    """

    matches: List[str] = Field(..., description="Determine data is target metrics or not", min_length=1)
    rules: List[str] = Field(default=[], description="Determine data is hit or not")


class CircuitBreakerConfig(BaseModel):
    """
    Declarative breaker configuration.
    """

    name: str
    metrics: MetricsSpec
    triggers: List[TriggerSpec]
