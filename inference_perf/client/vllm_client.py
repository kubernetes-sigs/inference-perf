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
from inference_perf.config import APIType
from inference_perf.utils import CustomTokenizer
from .base import ModelServerClient, ModelServerPrometheusMetric, PrometheusMetricMetadata, RequestMetric
from typing import Any, List
import aiohttp
import json
import time


class vLLMModelServerClient(ModelServerClient):
    def __init__(self, uri: str, model_name: str, tokenizer: CustomTokenizer, api_type: APIType) -> None:
        self.model_name = model_name
        self.uri = uri + ("/v1/chat/completions" if api_type == APIType.Chat else "/v1/completions")
        self.max_completion_tokens = 30
        self.tokenizer = tokenizer

    def set_report_generator(self, reportgen: ReportGenerator) -> None:
        self.reportgen = reportgen

    def _create_payload(self, payload: InferenceData) -> dict[str, Any]:
        if payload.type == APIType.Completion:
            return {
                "model": self.model_name,
                "prompt": payload.data.prompt if payload.data else "",
                "max_tokens": self.max_completion_tokens,
            }
        if payload.type == APIType.Chat:
            return {
                "model": self.model_name,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in (payload.chat.messages if payload.chat else [])
                ],
                "max_tokens": self.max_completion_tokens,
            }
        raise Exception("api type not supported - has to be completions or chat completions")

    async def process_request(self, data: InferenceData, stage_id: int) -> None:
        payload = self._create_payload(data)
        headers = {"Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            start = time.monotonic()
            try:
                async with session.post(self.uri, headers=headers, data=json.dumps(payload)) as response:
                    if response.status == 200:
                        content = await response.json()
                        end = time.monotonic()
                        choices = content.get("choices", [])

                        if data.type == APIType.Completion:
                            prompt = data.data.prompt if data.data else ""
                            output_text = choices[0].get("text", "")
                        elif data.type == APIType.Chat:
                            prompt = " ".join([msg.content for msg in data.chat.messages]) if data.chat else ""
                            output_text = choices[0].get("message", {}).get("content", "")
                        else:
                            raise Exception("Unsupported API type")

                    
                        prompt_tokens = self.tokenizer.count_tokens(prompt)
                        output_tokens = self.tokenizer.count_tokens(output_text)

                        self.request_metrics.append(
                            RequestMetric(
                                stage_id=stage_id,
                                prompt_tokens=prompt_tokens,
                                output_tokens=output_tokens,
                                time_per_request=end - start,
                            )
                        )
                    else:
                        print(await response.text())
            except aiohttp.ClientConnectorError as e:
                print("vLLM Server connection error:\n", str(e))

    def get_request_metrics(self) -> List[RequestMetric]:
        return self.request_metrics

    def get_prometheus_metric_metadata(self) -> PrometheusMetricMetadata:
        return self.prometheus_metric_metadata
