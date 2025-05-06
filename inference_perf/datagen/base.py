# Copyright 2025 The Kubernetes Authors.
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
from pydantic import BaseModel
from inference_perf.config import APIType
from abc import ABC, abstractmethod
from typing import Any, Generator, List


class PromptData(BaseModel, ABC):
    @abstractmethod
    def to_payload(self, model_name: str, max_tokens: int) -> dict[str, Any]:
        raise NotImplementedError


class CompletionData(PromptData):
    prompt: str

    def to_payload(self, model_name: str, max_tokens: int) -> dict[str, Any]:
        return {
            "model": model_name,
            "prompt": self.prompt,
            "max_tokens": max_tokens,
        }


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionData(PromptData):
    messages: List[ChatMessage]

    def to_payload(self, model_name: str, max_tokens: int) -> dict[str, Any]:
        return {
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "max_tokens": max_tokens,
        }


class DataGenerator(ABC):
    """Abstract base class for data generators."""

    apiType: APIType

    def __init__(self, apiType: APIType) -> None:
        if apiType not in self.get_supported_apis():
            raise Exception(f"Unsupported API type {apiType}")
        self.apiType = apiType

    @abstractmethod
    def get_supported_apis(self) -> List[APIType]:
        raise NotImplementedError

    @abstractmethod
    def get_data(self) -> Generator[PromptData, None, None]:
        raise NotImplementedError
