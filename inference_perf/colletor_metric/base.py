from abc import abstractmethod
from typing import Any, Optional
from pydantic import BaseModel


class Metric(BaseModel):
    """Abstract type to track reportable (but not neccesarily summarizable) metrics"""

    stage_id: Optional[int] = None

    @abstractmethod
    async def to_report(self) -> dict[str, Any]:
        """Create the report for this metric"""
        raise NotImplementedError
