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
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from inference_perf.config import APIConfig, APIType
from inference_perf.apis import InferenceAPIData
from .metrics import BaseMetrics


class ModelServerClient(ABC):
    @abstractmethod
    def __init__(self, api_config: APIConfig, timeout: Optional[float] = None, *args: Tuple[int, ...]) -> None:
        if api_config.type not in self.get_supported_apis():
            raise Exception(f"Unsupported API type {api_config}")

        self.api_config = api_config
        self.timeout = timeout

    def new_session(self) -> "ModelServerClientSession":
        return ModelServerClientSession(self)

    @abstractmethod
    def get_supported_apis(self) -> List[APIType]:
        raise NotImplementedError

    @abstractmethod
    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_prometheus_metric_metadata(self) -> BaseMetrics:
        # assumption: all metrics clients have metrics exported in Prometheus format
        raise NotImplementedError


class ModelServerClientSession:
    def __init__(self, client: ModelServerClient):
        self.client = client

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        await self.client.process_request(data, stage_id, scheduled_time, lora_adapter)

    async def close(self) -> None:  # noqa - subclasses optionally override this
        pass
