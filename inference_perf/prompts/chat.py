from typing import Any, List
import aiohttp
from pydantic import BaseModel
from inference_perf.prompts.base import LlmPrompt, PromptLifecycleMetric, ResponseData, ResponsesSummary, safe_float, summarize
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class ChatMessage(BaseModel):
    role: str
    content: str


class LlmChatCompletionPrompt(LlmPrompt):
    messages: List[ChatMessage]

    def get_route() -> str:
        return "/v1/chat/completions"

    def to_payload(self, model_name: str, max_tokens: int) -> dict[str, Any]:
        return {
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "max_tokens": max_tokens,
        }

    async def process_response(self, res: aiohttp.ClientResponse, tokenizer: CustomTokenizer) -> ResponseData:
        content = await res.json()
        choices = content.get("choices", [])
        output_text = choices[0].get("message", {}).get("content", "")
        output_len = tokenizer.count_tokens(output_text)
        return ResponseData(
            info={
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
                "time_per_request": summarize(
                    [(successful.end_time - successful.start_time) for successful in all_successful]
                ).model_dump(),
                "output_len": summarize(
                    [
                        float(v)
                        for success in all_successful
                        if (v := safe_float(success.response.info.get("output_len"))) is not None
                    ]
                ).model_dump(),
                "per_token_latency": summarize(
                    [
                        (success.end_time - success.start_time) / success.response.output_len
                        if success.response.output_len != 0
                        else 0
                        for success in all_successful
                    ]
                ).model_dump(),
            },
            failures={
                "count": len(all_failed),
                "time_per_request": summarize([(failed.end_time - failed.start_time) for failed in all_failed]).model_dump(),
            },
        )
