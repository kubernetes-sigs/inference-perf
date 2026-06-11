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
import time
from enum import Enum
from os import cpu_count
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inference_perf.config.common import Distribution, DistributionType
from inference_perf.config.datagen.replay import TraceConfig


class LoadType(Enum):
    CONSTANT = "constant"
    POISSON = "poisson"
    TRACE_REPLAY = "trace_replay"
    CONCURRENT = "concurrent"
    TRACE_SESSION_REPLAY = "trace_session_replay"


class LoadStage(BaseModel):
    """Base class for load stages. Use specific subclasses for different load types."""

    pass


class StandardLoadStage(LoadStage):
    """Load stage for CONSTANT and POISSON load types.

    Exactly one of ``rate`` or ``interval`` must be set: ``rate`` dispatches
    requests at a fixed QPS, ``interval`` draws the delay between consecutive
    requests from a Distribution (CONSTANT load type only).
    """

    rate: Optional[float] = Field(None, gt=0, description="Request rate (QPS). Specify exactly one of 'rate' or 'interval'.")
    interval: Optional[Distribution] = Field(
        None,
        description=(
            "Distribution of the delay in seconds between consecutive requests, "
            "sampled once per request (e.g. type: uniform, min: 1, max: 10). "
            "Only supported with the CONSTANT load type. Specify exactly one of "
            "'rate' or 'interval'. Set min/max explicitly: samples are clamped to "
            "[min, max] and the Distribution defaults are tuned for token counts."
        ),
    )
    duration: int = Field(..., gt=0, description="Duration in seconds")

    # These fields should not be set for standard load types
    num_requests: Optional[int] = Field(default=None, description="Not used for standard load types")
    concurrency_level: Optional[int] = Field(default=None, description="Not used for standard load types")

    @model_validator(mode="after")
    def validate_standard_fields(self) -> "StandardLoadStage":
        if (self.rate is None) == (self.interval is None):
            raise ValueError("Specify exactly one of 'rate' or 'interval' for CONSTANT/POISSON load stages")
        if self.interval is not None:
            if self.interval.min < 0:
                raise ValueError(
                    f"interval.min ({self.interval.min}) must be >= 0; intervals are delays in seconds between requests."
                )
            if self.interval.type == DistributionType.FIXED and self.interval.mean <= 0:
                raise ValueError(f"interval.mean ({self.interval.mean}) must be > 0 for a fixed interval")
        if self.num_requests is not None:
            raise ValueError("num_requests should not be set for CONSTANT/POISSON load types")
        if self.concurrency_level is not None:
            raise ValueError("concurrency_level should not be set for CONSTANT/POISSON load types")
        return self


class ConcurrentLoadStage(LoadStage):
    """Load stage for CONCURRENT load type."""

    num_requests: int = Field(..., gt=0, description="Number of requests to send")
    concurrency_level: int = Field(..., gt=0, description="Concurrency level")

    # These fields are set at runtime for load generation but should not be configured
    rate: Optional[float] = Field(None, description="Set at runtime for load generation")
    duration: Optional[int] = Field(None, description="Set at runtime for load generation")

    @model_validator(mode="after")
    def validate_concurrent_fields(self) -> "ConcurrentLoadStage":
        # Allow rate and duration to be set at runtime, but they should start as None
        # No validation needed here since they're set dynamically
        return self


class TraceSessionReplayLoadStage(LoadStage):
    """Load stage for TRACE_SESSION_REPLAY load type.

    A stage runs exactly ``num_sessions`` sessions (a slice of the corpus) at
    ``concurrent_sessions`` concurrency.  A session cursor on ``LoadGenerator``
    advances across stages so each stage draws the next N sessions — mirroring
    how ``get_data()`` advances through data across Standard/Concurrent stages.

    Modes:
    1. Simple concurrency control: set concurrent_sessions (and optionally num_sessions)
    2. Rate-based with concurrency: set concurrent_sessions + session_rate (+ num_sessions)
    3. Random-interval dispatch: set concurrent_sessions + session_interval (+ num_sessions)
    """

    # Session concurrency control (REQUIRED)
    concurrent_sessions: int = Field(
        ...,  # Required field
        ge=0,
        description=(
            "Maximum number of sessions active simultaneously. "
            "0 = all sessions active at once (stress test mode). "
            "N > 0 = at most N sessions active; when one completes, next is activated."
        ),
    )

    # Optional rate limiting
    session_rate: Optional[float] = Field(
        None,
        gt=0,
        description="Sessions to start per second (optional, omit for no rate limit)",
    )
    session_interval: Optional[Distribution] = Field(
        None,
        description=(
            "Distribution of the delay in seconds between session starts, sampled "
            "once per dispatch (e.g. type: uniform, min: 1, max: 10 starts each "
            "session 1-10s after the previous one). Mutually exclusive with "
            "session_rate. Set min/max explicitly: samples are clamped to "
            "[min, max] and the Distribution defaults are tuned for token counts."
        ),
    )
    num_sessions: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "Number of sessions to run in this stage. "
            "Draws the next N sessions from the corpus. "
            "None = all remaining sessions."
        ),
    )
    timeout: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Wall-clock safety limit in seconds. If exceeded, in-flight sessions are "
            "cancelled and stage exits as FAILED. Optional."
        ),
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_trace_session_fields(self) -> "TraceSessionReplayLoadStage":
        if self.session_rate is not None and self.session_interval is not None:
            raise ValueError(
                "Specify either 'session_rate' or 'session_interval', not both. "
                "session_interval is equivalent to a randomized session_rate."
            )

        if self.session_interval is not None:
            if self.session_interval.min < 0:
                raise ValueError(
                    f"session_interval.min ({self.session_interval.min}) must be >= 0; "
                    f"intervals are delays in seconds between session starts."
                )
            if self.session_interval.type == DistributionType.FIXED and self.session_interval.mean <= 0:
                raise ValueError(f"session_interval.mean ({self.session_interval.mean}) must be > 0 for a fixed interval")

        # Validate session_rate vs concurrent_sessions
        if self.session_rate is not None and self.concurrent_sessions > 0:
            if self.session_rate > self.concurrent_sessions:
                raise ValueError(
                    f"session_rate ({self.session_rate}) cannot exceed "
                    f"concurrent_sessions ({self.concurrent_sessions}). "
                    f"You can't start sessions faster than the concurrency limit allows."
                )

        return self


class StageGenType(Enum):
    GEOM = "geometric"
    LINEAR = "linear"


class SweepConfig(BaseModel):
    type: StageGenType
    num_requests: int = 2000
    timeout: float = 60
    num_stages: int = 5
    stage_duration: int = 180
    saturation_percentile: float = 95


class MultiLoRAConfig(BaseModel):
    name: str
    split: float


class LoadConfig(BaseModel):
    type: LoadType = LoadType.CONSTANT
    interval: float = 1.0
    stages: Union[List[StandardLoadStage], List[ConcurrentLoadStage], List[TraceSessionReplayLoadStage]] = []
    sweep: Optional[SweepConfig] = None
    num_workers: int = max(1, cpu_count())  # type: ignore
    worker_max_concurrency: int = 100
    worker_max_tcp_connections: int = 2500
    trace: Optional[TraceConfig] = None
    circuit_breakers: List[str] = []
    request_timeout: Optional[float] = None
    lora_traffic_split: Optional[List[MultiLoRAConfig]] = None
    base_seed: int = Field(default_factory=lambda: int(time.time() * 1000))

    @model_validator(mode="after")
    def validate_load_config(self) -> "LoadConfig":
        # Validate that sweep is not used with concurrent or trace session replay load types
        if self.type in (LoadType.CONCURRENT, LoadType.TRACE_SESSION_REPLAY) and self.sweep is not None:
            raise ValueError(f"Cannot have sweep config with {self.type.value.upper()} load type")

        # Validate stage types match load type
        if self.type == LoadType.CONCURRENT:
            for i, stage in enumerate(self.stages):
                if not isinstance(stage, ConcurrentLoadStage):
                    raise ValueError(
                        f"Stage {i}: CONCURRENT load type requires ConcurrentLoadStage, got {type(stage).__name__}"
                    )
        elif self.type == LoadType.TRACE_SESSION_REPLAY:
            for i, stage in enumerate(self.stages):
                if not isinstance(stage, TraceSessionReplayLoadStage):
                    raise ValueError(
                        f"Stage {i}: TRACE_SESSION_REPLAY load type requires TraceSessionReplayLoadStage, got {type(stage).__name__}"
                    )
        else:  # CONSTANT, POISSON, or TRACE_REPLAY
            for i, stage in enumerate(self.stages):
                if not isinstance(stage, StandardLoadStage):
                    raise ValueError(
                        f"Stage {i}: {self.type.value.upper()} load type requires StandardLoadStage, got {type(stage).__name__}"
                    )
                if stage.interval is not None and self.type != LoadType.CONSTANT:
                    raise ValueError(
                        f"Stage {i}: 'interval' is only supported with the CONSTANT load type; "
                        f"{self.type.value.upper()} defines its own arrival process, use 'rate' instead"
                    )

        # Validate multilora traffic split adds up to 1.0 if present
        if self.lora_traffic_split is not None:
            total = sum(config.split for config in self.lora_traffic_split)
            if total != 1.0:
                raise ValueError("MultiLoRA traffic split in load config does not add up to 1.0")

        return self
