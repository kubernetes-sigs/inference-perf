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
from inference_perf.datagen import InferenceData
from inference_perf.reportgen import ReportGenerator, RequestMetric
from .base import ModelServerClient
from typing import Any
import aiohttp
import json
import time


class vLLMModelServerClient(ModelServerClient):
    def __init__(self, uri: str, model_name: str) -> None:
        self.model_name = model_name
        self.uri = uri + "/v1/completions"
        self.max_completion_tokens = 30

    def set_report_generator(self, reportgen: ReportGenerator) -> None:
        self.reportgen = reportgen

    def _createPayload(self, data: InferenceData) -> dict[str, Any]:
        return {"model": self.model_name, "prompt": data.system_prompt, "max_tokens": self.max_completion_tokens}

    async def process_request(self, data: InferenceData) -> None:
        payload = self._createPayload(data)
        headers = {"Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            start = time.monotonic()
            async with session.post(self.uri, headers=headers, data=json.dumps(payload)) as response:
                content = await response.json()
                end = time.monotonic()
                usage = content["usage"]
                self.reportgen.collect_request_metrics(
                    RequestMetric(
                        prompt_tokens=usage["prompt_tokens"],
                        completion_tokens=usage["completion_tokens"],
                        time_taken=end - start,
                    )
                )
