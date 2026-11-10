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
"""SGLang's native /generate route (#986).

The request is ``text`` plus ``sampling_params``, outside the OpenAI envelope,
and token counts come back in ``meta_info`` rather than ``usage``.

/generate streams cumulative output by default: every chunk carries all the
text and output_ids so far. A server started with
--incremental-streaming-output sends only the new text and ids. parse_stream
handles both with per-request state from stream_state, the stream contract
proposed in #909.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

from aiohttp import ClientResponse

from inference_perf.apis import InferenceAPIData, InferenceInfo, StreamedResponseMetrics, UnaryResponseMetrics
from inference_perf.apis.response_errors import EmptyResponseError, InBandError, in_band_error
from inference_perf.apis.streaming_parser import parse_sse_stream
from inference_perf.config import APIConfig, APIType
from inference_perf.payloads import RequestBody, RequestMetrics, Text
from inference_perf.utils.custom_tokenizer import CustomTokenizer


@dataclass
class GenerateStreamState:
    # Characters of output text returned so far for this request.
    text_len: int = 0
    # The latest meta_info: SGLang's token counts for the request so far.
    meta_info: Optional[Dict[str, object]] = None


def stream_state() -> GenerateStreamState:
    return GenerateStreamState()


def parse_stream(chunk: Mapping[str, object], state: GenerateStreamState) -> Optional[str]:
    """The new output text in one streamed chunk, or None when it adds none.

    completion_tokens counts every token generated so far in both modes, so a
    chunk carrying fewer output_ids than that is a delta. The first chunk
    reads the same either way. Without both fields the chunk is read as the
    server's default, cumulative.
    """
    meta_info = chunk.get("meta_info")
    if isinstance(meta_info, dict):
        state.meta_info = meta_info
    text = chunk.get("text")
    if not isinstance(text, str):
        return None
    output_ids = chunk.get("output_ids")
    completion_tokens = meta_info.get("completion_tokens") if isinstance(meta_info, dict) else None
    if isinstance(output_ids, list) and isinstance(completion_tokens, int) and len(output_ids) < completion_tokens:
        new = text
    else:
        new = text[state.text_len :]
    state.text_len += len(new)
    return new or None


def server_usage(meta_info: object) -> Optional[Dict[str, object]]:
    """meta_info's token counts under the OpenAI usage keys the reports read, or None without any."""
    if not isinstance(meta_info, dict):
        return None
    usage: Dict[str, object] = {
        key: meta_info[key] for key in ("prompt_tokens", "completion_tokens") if isinstance(meta_info.get(key), int)
    }
    if isinstance(meta_info.get("cached_tokens"), int):
        usage["prompt_tokens_details"] = {"cached_tokens": meta_info["cached_tokens"]}
    return usage or None


class SGLangGenerateAPIData(InferenceAPIData):
    prompt: str
    max_tokens: int = 0
    model_response: str = ""

    def get_api_type(self) -> APIType:
        return APIType.SGLangGenerate

    def get_route(self) -> str:
        return "/generate"

    def _resolve_prompt_tokens(self, usage: Optional[Dict[str, object]], tokenizer: CustomTokenizer) -> int:
        # The server's count when it reports one, as for completions.
        prompt_tokens = usage.get("prompt_tokens") if usage else None
        if isinstance(prompt_tokens, int):
            return prompt_tokens
        return tokenizer.count_tokens(self.prompt)

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> RequestBody:
        # /generate serves the model the server was started with and takes no model field.
        if self.max_tokens == 0:
            self.max_tokens = max_tokens
        return {
            "text": self.prompt,
            "sampling_params": {"max_new_tokens": self.max_tokens, "ignore_eos": ignore_eos},
            "stream": streaming,
        }

    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str] = None
    ) -> InferenceInfo:
        if config.streaming:
            state = stream_state()
            new_texts: List[str] = []

            def extract(chunk: Mapping[str, object]) -> Optional[str]:
                new = parse_stream(chunk, state)
                if new:
                    new_texts.append(new)
                return new

            parsed = await parse_sse_stream(response, extract_content=extract)
            usage = server_usage(state.meta_info)
            # A stream with neither text nor token counts is not a generation.
            if not parsed.output_text and usage is None:
                raise EmptyResponseError(parsed.raw_content)
            prompt_len = self._resolve_prompt_tokens(usage, tokenizer)
            output_len = tokenizer.count_tokens(parsed.output_text, add_special_tokens=False)
            self.model_response = parsed.output_text
            return InferenceInfo(
                request_metrics=RequestMetrics(text=Text(input_tokens=prompt_len)),
                response_metrics=StreamedResponseMetrics(
                    # Each chunk's new text, one per entry of chunk_times. A cumulative chunk repeats all the
                    # text so far, which the report's token timeline would count again; the raw body stays in
                    # the request's response.
                    response_chunks=[json.dumps({"text": text}) for text in new_texts],
                    chunk_times=parsed.chunk_times,
                    output_tokens=output_len,
                    output_token_times=parsed.chunk_times,
                    server_usage=usage,
                ),
                lora_adapter=lora_adapter,
                extra_info={"raw_response": parsed.raw_content},
            )

        data = await response.json()
        # A 200 whose body is an error payload is that error, not a success.
        if (error_payload := in_band_error(data)) is not None:
            raise InBandError(error_payload)
        usage = server_usage(data.get("meta_info"))
        output_text = data.get("text")
        if not isinstance(output_text, str):
            output_text = ""
        if not output_text and usage is None:
            raise EmptyResponseError()
        prompt_len = self._resolve_prompt_tokens(usage, tokenizer)
        output_len = tokenizer.count_tokens(output_text, add_special_tokens=False)
        self.model_response = output_text
        return InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=prompt_len)),
            response_metrics=UnaryResponseMetrics(output_tokens=output_len, server_usage=usage),
            lora_adapter=lora_adapter,
        )
