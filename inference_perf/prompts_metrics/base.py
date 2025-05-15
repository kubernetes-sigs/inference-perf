from typing import Any, Optional

from pydantic import BaseModel
from inference_perf.colletor_metric.base import Metric
from inference_perf.prompts.base import LlmPrompt


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
