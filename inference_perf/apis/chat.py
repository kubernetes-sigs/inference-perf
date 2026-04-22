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

from typing import Any, List, Optional, Union
from aiohttp import ClientResponse
from pydantic import BaseModel
from inference_perf.apis import InferenceAPIData, InferenceInfo, UnaryInferenceResponseInfo, StreamedInferenceResponseInfo
from inference_perf.payloads import (
    Payload,
    Text,
    Images,
    Videos,
    Audios,
    Image,
    Video,
    Audio,
)
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig, APIType
from inference_perf.apis.streaming_parser import parse_sse_stream


class ChatMessage(BaseModel):
    role: str
    content: Union[str, list[dict[str, Any]]]


class ChatCompletionAPIData(InferenceAPIData):
    messages: List[ChatMessage]
    modalities: Optional[List[str]] = None
    audio: Optional[dict[str, Any]] = None
    max_tokens: int = 0
    multimodal_metrics: Optional[dict[str, Any]] = None

    def _get_payload(self, prompt_len: int, output_len: int) -> Payload:
        image = None
        video = None
        audio = None
        if self.multimodal_metrics:
            if "images" in self.multimodal_metrics:
                image_instances = [Image(**inst) for inst in self.multimodal_metrics.get("image_instances", [])]
                image = Images(count=self.multimodal_metrics["images"], instances=image_instances)
            if "videos" in self.multimodal_metrics or "video_instances" in self.multimodal_metrics:
                video_instances = [Video(**inst) for inst in self.multimodal_metrics.get("video_instances", [])]
                video = Videos(
                    count=self.multimodal_metrics.get("videos", 0),
                    instances=video_instances,
                )
            if "audio_clips" in self.multimodal_metrics or "audio_instances" in self.multimodal_metrics:
                audio_instances = [Audio(**inst) for inst in self.multimodal_metrics.get("audio_instances", [])]
                audio = Audios(
                    count=self.multimodal_metrics.get("audio_clips", 0),
                    instances=audio_instances,
                )
        return Payload(text=Text(input_tokens=prompt_len, output_tokens=output_len), image=image, video=video, audio=audio)

    def get_api_type(self) -> APIType:
        return APIType.Chat

    def get_route(self) -> str:
        return "/v1/chat/completions"

    async def to_payload(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> dict[str, Any]:
        if self.max_tokens == 0:
            self.max_tokens = max_tokens
        payload: dict[str, Any] = {
            "model": effective_model_name,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "max_tokens": self.max_tokens,
            "ignore_eos": ignore_eos,
            "stream": streaming,
            **({"stream_options": {"include_usage": True}} if streaming else {}),
        }
        if self.modalities:
            payload["modalities"] = self.modalities
        if self.audio:
            payload["audio"] = self.audio
        return payload

    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str] = None
    ) -> InferenceInfo:
        if config.streaming:
            # Use shared streaming parser with chat-specific content extraction
            output_text, chunk_times, raw_content, response_chunks, server_usage = await parse_sse_stream(
                response, extract_content=lambda data: data.get("choices", [{}])[0].get("delta", {}).get("content")
            )

            prompt_len = sum(
                tokenizer.count_tokens(
                    msg.content
                    if isinstance(msg.content, str)
                    else "".join(
                        part.get("text", "")
                        for part in msg.content
                        if isinstance(part, dict) and part.get("type") in ["text", "input_text"]
                    )
                )
                for msg in self.messages
                if msg.content
            )
            output_len = tokenizer.count_tokens(output_text)
            return InferenceInfo(
                payload=self._get_payload(prompt_len, output_len),
                input_tokens=prompt_len,
                response_info=StreamedInferenceResponseInfo(
                    response_chunks=response_chunks,
                    chunk_times=chunk_times,
                    output_tokens=output_len,
                    output_token_times=chunk_times,
                    server_usage=server_usage,
                ),
                lora_adapter=lora_adapter,
                extra_info={"raw_response": raw_content},
            )
        else:
            data = await response.json()
            prompt_len = sum(
                tokenizer.count_tokens(
                    msg.content
                    if isinstance(msg.content, str)
                    else "".join(
                        part.get("text", "")
                        for part in msg.content
                        if isinstance(part, dict) and part.get("type") in ["text", "input_text"]
                    )
                )
                for msg in self.messages
                if msg.content
            )
            choices = data.get("choices", [])
            if len(choices) == 0:
                return InferenceInfo(
                    payload=self._get_payload(prompt_len, 0),
                    input_tokens=prompt_len,
                    lora_adapter=lora_adapter,
                )
            output_text = "".join([choice.get("message", {}).get("content", "") for choice in choices])
            output_len = tokenizer.count_tokens(output_text)
            return InferenceInfo(
                payload=self._get_payload(prompt_len, output_len),
                input_tokens=prompt_len,
                response_info=UnaryInferenceResponseInfo(output_tokens=output_len),
                lora_adapter=lora_adapter,
            )
