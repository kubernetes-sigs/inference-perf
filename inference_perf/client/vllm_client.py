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
from inference_perf.reportgen import ReportGenerator
from inference_perf.config import APIType, CustomTokenizerConfig, RequestMetric
from inference_perf.utils import CustomTokenizer
from .base import FailedResponseData, ModelServerClient, PromptData, ResponseData, SuccessfulResponseData
from typing import Any, List, Optional
from aiohttp import ClientSession, ClientResponse
import json
import time

class VllmCompletionPromptData(PromptData):
    prompt: str

    def to_payload(self, model_name: str, max_tokens: int) -> dict[str, Any]:
        return {
            "model": model_name,
            "prompt": self.prompt,
            "max_tokens": max_tokens,
        }

    async def get_response_data(self, res: ClientResponse, tokenizer: CustomTokenizer) -> ResponseData:
        content = await res.json()
        choices = content.get("choices", [])
        output_text = choices[0].get("text", "")
        output_len = tokenizer.count_tokens(output_text)
        return SuccessfulResponseData(
            info={
                "output_text": output_text,
                "output_len": output_len,
            }
        )


class ChatMessage(BaseModel):
    role: str
    content: str


class VllmChatCompletionPromptData(PromptData):
    messages: List[ChatMessage]

    def to_payload(self, model_name: str, max_tokens: int) -> dict[str, Any]:
        return {
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "max_tokens": max_tokens,
        }

    async def get_response_data(self, res: ClientResponse, tokenizer: CustomTokenizer) -> ResponseData:
        content = await res.json()
        choices = content.get("choices", [])
        output_text = choices[0].get("message", {}).get("content", "")
        output_len = tokenizer.count_tokens(output_text)
        return SuccessfulResponseData(
            info={
                "output_text": output_text,
                "output_len": output_len,
            }
        )


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

    async def process_request(self, prompt_data: PromptData, stage_id: int) -> None:
        payload = prompt_data.to_payload(model_name=self.model_name, max_tokens=self.max_completion_tokens)
        headers = {"Content-Type": "application/json"}
        async with ClientSession() as session:
            start = time.monotonic()
            try:
                async with session.post(self.uri, headers=headers, data=json.dumps(payload)) as response:
                    if response.status == 200:
                        response_data = await prompt_data.get_response_data(res=response, tokenizer=self.custom_tokenizer)
                        end = time.monotonic()

                        self.reportgen.collect_request_metric(
                            RequestMetric(
                                stage_id=stage_id,
                                request=prompt_data,
                                response=response_data,
                                start_time=start,
                                end_time=end,
                            )
                        )
                    else:
                        RequestMetric(
                            stage_id=stage_id,
                            request=prompt_data,
                            response=FailedResponseData(error_msg=(await response.text()), error_type="Non 200 reponse"),
                            start_time=start,
                            end_time=time.monotonic(),
                        )
            except Exception as e:
                RequestMetric(
                    stage_id=stage_id,
                    request=prompt_data,
                    response=FailedResponseData(error_msg=str(e), error_type=type(e).__name__),
                    start_time=start,
                    end_time=time.monotonic(),
                )
