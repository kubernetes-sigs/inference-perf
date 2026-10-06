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

import logging
from typing import Any, Callable, Dict, List, Optional

import jmespath
from aiohttp import ClientResponse
from inference_perf.apis import InferenceAPIData, InferenceInfo, StreamedResponseMetrics, UnaryResponseMetrics
from inference_perf.apis.streaming_parser import parse_ndjson_stream, parse_sse_stream
from inference_perf.payloads import RequestBody, RequestMetrics, Text
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import (
    APIConfig,
    APIType,
    TemplateConfig,
    TemplateResponseConfig,
    TemplateStreamChunks,
    TemplateStreamConfig,
    TemplateStreamFraming,
)

logger = logging.getLogger(__name__)

# Count paths that have already logged a warning in this process. A wrong path
# would otherwise log on every request.
_warned_count_paths: set[str] = set()


class _ChunkReader:
    """Reads the new text and the token counts from each chunk of a stream."""

    def __init__(self, response: TemplateResponseConfig, cumulative: bool) -> None:
        self.text_path = response.text_path
        self.count_paths = [path for path in (response.input_tokens_path, response.output_tokens_path) if path]
        self.cumulative = cumulative
        self.chunk_texts: List[str] = []
        # The last value that each count path selected.
        self.counts: Dict[str, Any] = {}
        # Whether text_path selected a string in any chunk.
        self.found_text = False
        # The text so far of a cumulative stream.
        self.text = ""

    def __call__(self, chunk: Any) -> Optional[str]:
        for path in self.count_paths:
            value = jmespath.search(path, chunk)
            if value is not None:
                self.counts[path] = value
        text = jmespath.search(self.text_path, chunk)
        if not isinstance(text, str):
            return None
        self.found_text = True
        new_text = text
        if self.cumulative:
            # An empty text is a chunk without text, not an output that was cleared.
            if not text:
                return None
            # A cumulative chunk repeats the text so far, so only what is past it is new.
            new_text = text[len(self.text) :]
            self.text = text
        if new_text:
            self.chunk_texts.append(new_text)
        return new_text


class TemplateAPIData(InferenceAPIData):
    prompt: str
    max_tokens: int = 0
    # Only used to count the prompt on the client, as in CompletionAPIData. The
    # template decides what the server receives.
    add_special_tokens: Optional[bool] = None
    template: TemplateConfig
    # The route with ${model} filled in. It is set with the request body, because
    # the model name is only known then.
    route: Optional[str] = None

    def get_api_type(self) -> APIType:
        return APIType.Template

    def get_route(self) -> str:
        if self.route is None:
            raise RuntimeError("the template route is filled in by to_request_body(), which has not run yet")
        return self.route

    def _count_prompt_tokens(self, tokenizer: CustomTokenizer) -> int:
        return tokenizer.count_tokens(
            self.prompt, add_special_tokens=self.add_special_tokens if self.add_special_tokens is not None else True
        )

    def _resolve_prompt_tokens(self, server_usage: Optional[Dict[str, Any]], tokenizer: CustomTokenizer) -> int:
        """Input tokens as reported by the server, falling back to client-side tokenization."""
        prompt_tokens = server_usage.get("prompt_tokens") if server_usage else None
        if prompt_tokens is not None:
            return int(prompt_tokens)
        return self._count_prompt_tokens(tokenizer)

    def _server_usage(self, select: Callable[[str], Any]) -> Optional[Dict[str, Any]]:
        """Token counts read from the response, under the keys of an OpenAI usage object.

        select returns the value that a count path selected.
        """
        usage: Dict[str, Any] = {}
        for key, name, path in (
            ("prompt_tokens", "input_tokens_path", self.template.response.input_tokens_path),
            ("completion_tokens", "output_tokens_path", self.template.response.output_tokens_path),
        ):
            if path is None:
                continue
            value = select(path)
            # bool is a subclass of int, but it is never a token count.
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] = value
            elif path not in _warned_count_paths:
                _warned_count_paths.add(path)
                found = "nothing" if value is None else f"a {type(value).__name__}"
                logger.warning(
                    "Template response %s '%s' selected %s, not an integer. The tokenizer count is used instead. "
                    "This is logged once per path.",
                    name,
                    path,
                    found,
                )
        return usage or None

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> RequestBody:
        if self.max_tokens == 0:
            self.max_tokens = max_tokens
        self.route = self.template.render_route(effective_model_name)
        # The ignore_eos and streaming arguments are not sent. The template holds
        # every option the server takes.
        return self.template.render_body({"prompt": self.prompt, "max_tokens": self.max_tokens, "model": effective_model_name})

    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str] = None
    ) -> InferenceInfo:
        stream = self.template.response.stream
        # Config load makes sure that stream is set exactly when streaming is on.
        if config.streaming and stream is not None:
            return await self._process_stream(response, stream, tokenizer, lora_adapter)
        # A custom server does not always set an application/json content type.
        body = await response.json(content_type=None)
        text_path = self.template.response.text_path
        text = jmespath.search(text_path, body)
        if not isinstance(text, str):
            raise ValueError(f"text_path '{text_path}' did not select a string in the response")
        server_usage = self._server_usage(lambda path: jmespath.search(path, body))
        return InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=self._resolve_prompt_tokens(server_usage, tokenizer))),
            # Generated text is a continuation, so it is counted without special tokens.
            response_metrics=UnaryResponseMetrics(
                output_tokens=tokenizer.count_tokens(text, add_special_tokens=False), server_usage=server_usage
            ),
            lora_adapter=lora_adapter,
        )

    async def _process_stream(
        self, response: ClientResponse, stream: TemplateStreamConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str]
    ) -> InferenceInfo:
        reader = _ChunkReader(self.template.response, cumulative=stream.chunks == TemplateStreamChunks.CUMULATIVE)
        parse = parse_ndjson_stream if stream.framing == TemplateStreamFraming.NDJSON else parse_sse_stream
        output_text, chunk_times, raw_content, _, _ = await parse(response, reader)
        if not reader.found_text:
            raise ValueError(f"text_path '{reader.text_path}' did not select a string in any chunk of the response")
        # In a cumulative stream the last text is the whole output, even if the server changed earlier text.
        text = reader.text if reader.cumulative else output_text
        server_usage = self._server_usage(reader.counts.get)
        return InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=self._resolve_prompt_tokens(server_usage, tokenizer))),
            # The raw stream is kept once, as raw_response. Keeping the chunks again
            # would double the memory of a cumulative stream, so the report gets
            # chunk_texts instead.
            response_metrics=StreamedResponseMetrics(
                chunk_times=chunk_times,
                chunk_texts=reader.chunk_texts,
                output_tokens=tokenizer.count_tokens(text, add_special_tokens=False),
                output_token_times=chunk_times,
                server_usage=server_usage,
            ),
            lora_adapter=lora_adapter,
            extra_info={"raw_response": raw_content},
        )
