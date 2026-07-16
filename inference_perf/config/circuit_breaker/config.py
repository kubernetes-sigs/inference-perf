# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Literal, Union

from pydantic import BaseModel, Field


class TriggerConsecutive(BaseModel):
    type: Literal["consecutive"] = Field(description="Trip after a number of consecutive hits.")
    threshold: int = Field(..., ge=1, description="Number of consecutive hits that trips the breaker.")


class TriggerRateOverWindow(BaseModel):
    type: Literal["rate_over_window"] = Field(description="Trip when the hit rate over a time window exceeds a threshold.")
    window_sec: float = Field(..., gt=0.0, description="Length of the sliding window in seconds.")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Hit rate (0.0 to 1.0) over the window that trips the breaker.")
    min_samples: int = Field(0, ge=0, description="Minimum samples in the window before the trigger can trip.")


TriggerSpec = Union[
    TriggerConsecutive,
    TriggerRateOverWindow,
]


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

    name: str = Field(description="Unique name for this circuit breaker, referenced from load.circuit_breakers.")
    metrics: MetricsSpec = Field(description="Which metrics the breaker watches and when a data point counts as a hit.")
    triggers: List[TriggerSpec] = Field(description="Conditions on the watched metrics that trip the breaker.")
