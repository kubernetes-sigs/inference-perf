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


from typing import Any, List
import aiohttp
from inference_perf.prompts.base import LlmPrompt, PromptLifecycleMetric, ResponseData, ResponsesSummary, safe_float, summarize
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class LlmCompletionPrompt(LlmPrompt):
    prompt: str

    def get_route() -> str:
        return "/v1/completions"

    def to_payload(self, model_name: str, max_tokens: int) -> dict[str, Any]:
        return {
            "model": model_name,
            "prompt": self.prompt,
            "max_tokens": max_tokens,
        }

    async def process_response(self, res: aiohttp.ClientResponse, tokenizer: CustomTokenizer) -> ResponseData:
        content = await res.json()
        choices = content.get("choices", [])
        prompt_len = tokenizer.count_tokens(self.prompt)
        output_text = choices[0].get("text", "")
        output_len = tokenizer.count_tokens(output_text)
        return ResponseData(
            info={
                "prompt": self.prompt,
                "prompt_len": prompt_len,
                "output_text": output_text,
                "output_len": output_len,
            },
            error=None,
        )

    def summarize_requests(self, metrics: List[PromptLifecycleMetric]) -> ResponsesSummary:
        all_successful: List[PromptLifecycleMetric] = [x for x in metrics if x.response.error is None]
        all_failed: List[PromptLifecycleMetric] = [x for x in metrics if x.response.error is not None]

        return ResponsesSummary(
            load_summary={
                "count": len(metrics),
            },
            successes={
                "count": len(all_successful),
                "time_per_request": summarize([(metric.end_time - metric.start_time) for metric in metrics]).model_dump(),
                "prompt_len": summarize(
                    [safe_float(success.response.info.get("prompt_len")) for success in all_successful]
                ).model_dump(),
                "output_len": summarize(
                    [float(v) for success in all_successful if (v := success.response.info.get("output_len")) is not None]
                ).model_dump(),
                "per_token_latency": summarize(
                    [
                        ((metric.end_time - metric.start_time) / output_len) if output_len and output_len != 0 else 0
                        for metric in all_successful
                        for output_len in [safe_float(metric.response.info.get("output_len"))]
                    ]
                ).model_dump(),
            },
            failures={
                "count": len(all_failed),
                # need to filter to only the failures, currently dont do that, same for successes
                "time_per_request": summarize([(failed.end_time - failed.start_time) for failed in all_failed]).model_dump()
                if len(all_failed) != 0
                else None,
            },
        )


