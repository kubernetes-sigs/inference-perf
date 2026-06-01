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
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class APIType(Enum):
    Completion = "completion"
    Chat = "chat"


class ResponseFormatType(Enum):
    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"


class ResponseFormat(BaseModel):
    """Configuration for structured output via response_format parameter.

    See vLLM docs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    """

    type: ResponseFormatType = ResponseFormatType.JSON_SCHEMA
    name: str = "structured_output"  # Name for the json_schema
    json_schema: Optional[dict[str, Any]] = None

    def to_api_format(self) -> dict[str, Any]:
        """Convert to the format expected by vLLM/OpenAI API."""
        if self.type == ResponseFormatType.JSON_OBJECT:
            return {"type": "json_object"}
        # json_schema type
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "schema": self.json_schema,
            },
        }


class APIConfig(BaseModel):
    type: APIType = APIType.Completion
    streaming: bool = False
    headers: Optional[dict[str, str]] = None
    slo_unit: Optional[str] = None
    slo_tpot_header: Optional[str] = None
    slo_ttft_header: Optional[str] = None
    response_format: Optional[ResponseFormat] = None
