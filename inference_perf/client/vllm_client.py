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
from inference_perf.datagen import PromptData
from inference_perf.reportgen import ReportGenerator
from inference_perf.config import APIType, CustomTokenizerConfig, FailedResponse, RequestMetric, SuccessfulResponse
from inference_perf.utils import CustomTokenizer
from .base import ModelServerClient
from typing import Optional
import aiohttp
import json
import time


class vLLMModelServerClient(ModelServerClient):
    def __init__(self, uri: str, model_name: str, tokenizer: Optional[CustomTokenizerConfig], api_type: APIType) -> None:
        self.model_name = model_name
        self.uri = uri + ("/v1/chat/completions" if api_type == APIType.Chat else "/v1/completions")
        self.max_completion_tokens = 30
        self.tokenizer_available = False

        if tokenizer and tokenizer.pretrained_model_name_or_path:
            try:
                self.custom_tokenizer = CustomTokenizer(
                    tokenizer.pretrained_model_name_or_path,
                    tokenizer.token,
                    tokenizer.trust_remote_code,
                )
                self.tokenizer_available = True
            except Exception as e:
                print(f"Tokenizer initialization failed: {e}")
                print("Falling back to usage metrics.")
        else:
            print("Tokenizer path is empty. Falling back to usage metrics.")

    def set_report_generator(self, reportgen: ReportGenerator) -> None:
        self.reportgen = reportgen

    async def process_request(self, data: PromptData, stage_id: int) -> None:
        payload = data.to_payload(model_name=self.model_name, max_tokens=self.max_completion_tokens)
        headers = {"Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            start = time.monotonic()
            try:
                async with session.post(self.uri, headers=headers, data=json.dumps(payload)) as response:
                    if response.status == 200:
                        content = await response.json()
                        end = time.monotonic()
                        usage = content.get("usage", {})
                        choices = content.get("choices", [])

                        if data.type == APIType.Completion:
                            prompt = data.data.prompt if data.data else ""
                            output_text = choices[0].get("text", "")
                        elif data.type == APIType.Chat:
                            prompt = " ".join([msg.content for msg in data.chat.messages]) if data.chat else ""
                            output_text = choices[0].get("message", {}).get("content", "")
                        else:
                            raise Exception("Unsupported API type")

                        if self.tokenizer_available:
                            prompt_len = self.custom_tokenizer.count_tokens(prompt)
                            output_len = self.custom_tokenizer.count_tokens(output_text)
                        else:
                            prompt_len = usage.get("prompt_tokens", 0)
                            output_len = usage.get("completion_tokens", 0)

                        self.reportgen.collect_request_metric(
                            RequestMetric(
                                stage_id=stage_id,
                                prompt_len=prompt_len,
                                prompt=prompt,
                                result=SuccessfulResponse(output_len=output_len, output=output_text),
                                start_time=start,
                                end_time=end,
                            )
                        )
                    else:
                        print(await response.text())
            except Exception as e:
                RequestMetric(
                    stage_id=stage_id,
                    prompt_len=prompt_len,
                    prompt=prompt,
                    result=FailedResponse(exception=e),
                    start_time=start,
                    end_time=time.monotonic(),
                )
