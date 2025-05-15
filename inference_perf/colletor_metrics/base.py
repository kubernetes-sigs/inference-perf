from abc import abstractmethod
from typing import Any, Optional
from pydantic import BaseModel

from inference_perf.prompts.base import LlmPrompt


class Metric(BaseModel):
    """Abstract type to track reportable (but not neccesarily summarizable) metrics"""

    stage_id: Optional[int] = None

    @abstractmethod
    async def to_report(self) -> dict[str, Any]:
        """Create the report for this metric"""
        raise NotImplementedError


class FailedResponseData(BaseModel):
    error_type: str
    error_msg: str


class ResponseData(BaseModel):
    info: dict[str, Any]
    error: Optional[FailedResponseData]


class PromptLifecycleMetric(Metric):
    """Tracks data for a request across its lifecycle"""

    start_time: float
    end_time: float
    request: LlmPrompt
    response: ResponseData

    async def to_report(self) -> dict[str, Any]:
        return self.model_dump()
