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

import base64
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from aiohttp import ClientResponse
from pydantic import BaseModel

from inference_perf.apis import InferenceAPIData, InferenceInfo, UnaryResponseMetrics, StreamedResponseMetrics
from inference_perf.payloads.multimodal_spec import ImageRepresentation, MultimodalSpec, VideoRepresentation
from inference_perf.apis.streaming_parser import parse_sse_stream
from inference_perf.config import APIConfig, APIType
from inference_perf.mediagen.pool import get_video_pool
from inference_perf.mediagen.synthesis import generate_jpeg_bytes, generate_mp4_bytes, generate_png_bytes, generate_wav_bytes
from inference_perf.payloads import (
    Audio,
    Audios,
    Image,
    Images,
    RequestBody,
    RequestMetrics,
    Text,
    Video,
    Videos,
)
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class ChatMessage(BaseModel):
    role: str
    content: Union[str, list[dict[str, Any]]]


def _encode_image(width: int, height: int, representation: ImageRepresentation, rng: np.random.Generator) -> Tuple[bytes, str]:
    """Generate image bytes in the requested representation and return ``(bytes, data_url)``."""
    if representation == ImageRepresentation.JPEG:
        raw = generate_jpeg_bytes(width, height, rng)
        return raw, f"data:image/jpeg;base64,{base64.b64encode(raw).decode('ascii')}"
    raw = generate_png_bytes(width, height, rng)
    return raw, f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"


def assemble_content(text_str: str, media_items: list[Tuple[dict[str, Any], float]]) -> list[dict[str, Any]]:
    """Weave media blocks into a text prompt at sampled insertion points."""
    if not media_items:
        return [{"type": "text", "text": text_str}]

    media_items.sort(key=lambda x: x[1])
    content: list[dict[str, Any]] = []
    last_idx = 0
    text_len = len(text_str)

    for block, point in media_items:
        point = max(0.0, min(1.0, point))
        idx = int(point * text_len)

        if idx > last_idx:
            content.append({"type": "text", "text": text_str[last_idx:idx]})
            last_idx = idx

        content.append(block)

    if last_idx < text_len:
        content.append({"type": "text", "text": text_str[last_idx:]})

    return content


class ChatCompletionAPIData(InferenceAPIData):
    messages: List[ChatMessage]
    max_tokens: int = 0

    # Payload-side multimodal spec (sampled per-request). Materialized into the
    # first user message at ``to_request_body`` time using a fresh RNG.
    multimodal_spec: Optional[MultimodalSpec] = None

    # Optional shared prefix carried separately from the user message text.
    # When ``prefix_text`` and/or ``prefix_multimodal_spec`` are set, the wire
    # user message is built as: prefix_content + " " + payload_content.
    # Prefix-side bytes are materialized with deterministic per-instance seeding
    # so that the same spec produces identical wire bytes across requests in a
    # group — required for prefix-cache benchmarks.
    prefix_text: Optional[str] = None
    prefix_multimodal_spec: Optional[MultimodalSpec] = None
    # Folded into the deterministic seed when materializing
    # ``prefix_multimodal_spec`` so that prefix specs with identical
    # dimensions but different ``prefix_cache_key`` produce distinct bytes.
    # Datagens that sample one prefix spec per shared-prefix group set this
    # to the group id to keep within-group bytes stable while making across-
    # group bytes diverge.
    prefix_cache_key: Optional[int] = None

    # Realized per-instance metrics populated by ``to_request_body`` and read
    # by ``process_response`` to build ``RequestMetrics``. Aggregates across
    # prefix-side and payload-side specs.
    realized_images: Optional[Images] = None
    realized_videos: Optional[Videos] = None
    realized_audios: Optional[Audios] = None

    @staticmethod
    def _materialize_multimodal_content(
        text_str: str,
        spec: MultimodalSpec,
        deterministic: bool,
        cache_key: Optional[int] = None,
    ) -> Tuple[list[dict[str, Any]], list[Image], list[Video], list[Audio]]:
        """Materialize bytes from ``spec`` and weave into ``text_str``.

        Returns ``(content_list, images, videos, audios)``. Pure (no mutation
        of class state). ``deterministic=True`` seeds RNGs per-instance from
        spec content + ``cache_key``, yielding identical bytes for identical
        (spec, cache_key) pairs across calls — required for prefix-cache
        benchmarks. ``cache_key`` is ignored when ``deterministic=False``.
        """
        media_items: list[Tuple[dict[str, Any], float]] = []
        image_instances: list[Image] = []
        video_instances: list[Video] = []
        audio_instances: list[Audio] = []
        fresh_rng = np.random.default_rng()

        def _rng_for(*key_parts: object) -> np.random.Generator:
            if not deterministic:
                return fresh_rng
            return np.random.default_rng(hash((cache_key,) + key_parts) & 0xFFFFFFFF)

        for i, img in enumerate(spec.images):
            raw_bytes, data_url = _encode_image(
                img.width, img.height, img.representation, _rng_for("img", i, img.width, img.height)
            )
            media_items.append(({"type": "image_url", "image_url": {"url": data_url}}, img.insertion_point))
            image_instances.append(
                Image(
                    pixels=img.width * img.height,
                    bytes=len(raw_bytes),
                    aspect_ratio=img.width / img.height if img.height > 0 else 0.0,
                )
            )

        for vi, vid in enumerate(spec.videos):
            aspect = vid.width / vid.height if vid.height > 0 else 0.0
            if vid.representation == VideoRepresentation.MP4:
                if deterministic:
                    # Deterministic mode bypasses the pool — pool sampling is
                    # non-deterministic, and this path is rare (prefix-side
                    # videos in shared-prefix benchmarks) so per-init encode
                    # cost is acceptable.
                    mp4_bytes = generate_mp4_bytes(
                        vid.width, vid.height, vid.frames, _rng_for("vid_mp4", vi, vid.width, vid.height, vid.frames)
                    )
                else:
                    mp4_bytes = get_video_pool().get(vid.width, vid.height, vid.frames)
                data_url = f"data:video/mp4;base64,{base64.b64encode(mp4_bytes).decode('ascii')}"
                media_items.append(({"type": "video_url", "video_url": {"url": data_url}}, vid.insertion_point))
                total_bytes = len(mp4_bytes)
            else:
                # PNG_FRAMES or JPEG_FRAMES — emit one image_url block per frame.
                frame_fmt = (
                    ImageRepresentation.JPEG
                    if vid.representation == VideoRepresentation.JPEG_FRAMES
                    else ImageRepresentation.PNG
                )
                total_bytes = 0
                for f in range(vid.frames):
                    raw_bytes, data_url = _encode_image(
                        vid.width, vid.height, frame_fmt, _rng_for("vid_frame", vi, f, vid.width, vid.height)
                    )
                    media_items.append(({"type": "image_url", "image_url": {"url": data_url}}, vid.insertion_point))
                    total_bytes += len(raw_bytes)
            video_instances.append(
                Video(pixels=vid.width * vid.height, bytes=total_bytes, aspect_ratio=aspect, frames=vid.frames)
            )

        for aud in spec.audios:
            # WAV generation is deterministic by duration (silent samples), so no
            # RNG is involved.
            wav_bytes = generate_wav_bytes(aud.duration)
            b64_audio = base64.b64encode(wav_bytes).decode("ascii")
            media_items.append(
                ({"type": "input_audio", "input_audio": {"data": b64_audio, "format": "wav"}}, aud.insertion_point)
            )
            audio_instances.append(Audio(bytes=len(wav_bytes), seconds=aud.duration))

        return assemble_content(text_str, media_items), image_instances, video_instances, audio_instances

    def _build_user_content(self) -> Optional[list[dict[str, Any]]]:
        """Build the wire user-message content from prefix + payload specs.

        Aggregates realized metrics onto ``self`` for ``process_response``.
        Returns ``None`` when no prefix/payload multimodal data is set, in
        which case ``to_request_body`` falls through to the plain text path.
        """
        if self.prefix_multimodal_spec is None and self.multimodal_spec is None and self.prefix_text is None:
            return None

        # Find the first user message text (if any) for payload-side weaving.
        user_text = ""
        for m in self.messages:
            if m.role == "user" and isinstance(m.content, str):
                user_text = m.content
                break

        all_images: list[Image] = []
        all_videos: list[Video] = []
        all_audios: list[Audio] = []

        if self.prefix_multimodal_spec is not None or self.prefix_text is not None:
            prefix_text = self.prefix_text or ""
            if self.prefix_multimodal_spec is not None:
                prefix_content, p_imgs, p_vids, p_auds = self._materialize_multimodal_content(
                    prefix_text, self.prefix_multimodal_spec, deterministic=True, cache_key=self.prefix_cache_key
                )
                all_images.extend(p_imgs)
                all_videos.extend(p_vids)
                all_audios.extend(p_auds)
            else:
                prefix_content = [{"type": "text", "text": prefix_text}] if prefix_text else []
        else:
            prefix_content = []

        if self.multimodal_spec is not None:
            payload_content, q_imgs, q_vids, q_auds = self._materialize_multimodal_content(
                user_text, self.multimodal_spec, deterministic=False
            )
            all_images.extend(q_imgs)
            all_videos.extend(q_vids)
            all_audios.extend(q_auds)
        else:
            payload_content = [{"type": "text", "text": user_text}] if user_text else []

        # Concatenate prefix + payload, inserting a single space between them
        # if both sides have content. Adjacent text blocks are merged.
        if prefix_content and payload_content:
            if prefix_content[-1].get("type") == "text" and payload_content[0].get("type") == "text":
                merged = dict(prefix_content[-1])
                merged["text"] = merged["text"] + " " + payload_content[0]["text"]
                combined = list(prefix_content[:-1]) + [merged] + list(payload_content[1:])
            else:
                combined = list(prefix_content) + [{"type": "text", "text": " "}] + list(payload_content)
        else:
            combined = list(prefix_content) + list(payload_content)

        self.realized_images = Images(count=len(all_images), instances=all_images) if all_images else None
        self.realized_videos = Videos(count=len(all_videos), instances=all_videos) if all_videos else None
        self.realized_audios = Audios(count=len(all_audios), instances=all_audios) if all_audios else None

        return combined

    def _build_request_metrics(self, prompt_len: int, output_len: int) -> RequestMetrics:
        return RequestMetrics(
            text=Text(input_tokens=prompt_len),
            image=self.realized_images,
            video=self.realized_videos,
            audio=self.realized_audios,
        )

    def get_api_type(self) -> APIType:
        return APIType.Chat

    def get_route(self) -> str:
        return "/v1/chat/completions"

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> RequestBody:
        if self.max_tokens == 0:
            self.max_tokens = max_tokens

        # When any multimodal/prefix data is present, build a single combined
        # user message via the orchestrator. Otherwise pass messages through.
        user_content = self._build_user_content()
        if user_content is not None:
            messages: list[dict[str, Any]] = []
            replaced = False
            for m in self.messages:
                if not replaced and m.role == "user":
                    messages.append({"role": "user", "content": user_content})
                    replaced = True
                else:
                    messages.append({"role": m.role, "content": m.content})
            if not replaced:
                messages.append({"role": "user", "content": user_content})
        else:
            messages = [{"role": m.role, "content": m.content} for m in self.messages]

        return {
            "model": effective_model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "ignore_eos": ignore_eos,
            "stream": streaming,
            **({"stream_options": {"include_usage": True}} if streaming else {}),
        }

    def _count_prompt_tokens(self, tokenizer: CustomTokenizer) -> int:
        # Count text in shared prefix (if any) plus text in each message. Only
        # text parts of structured content count — image/video/audio bytes
        # don't contribute to text-token totals here.
        total = tokenizer.count_tokens(self.prefix_text) if self.prefix_text else 0
        total += sum(
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
        return total

    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str] = None
    ) -> InferenceInfo:
        if config.streaming:
            output_text, chunk_times, raw_content, response_chunks, server_usage = await parse_sse_stream(
                response, extract_content=lambda data: data.get("choices", [{}])[0].get("delta", {}).get("content")
            )
            prompt_len = self._count_prompt_tokens(tokenizer)
            output_len = tokenizer.count_tokens(output_text)
            return InferenceInfo(
                request_metrics=self._build_request_metrics(prompt_len, output_len),
                response_metrics=StreamedResponseMetrics(
                    response_chunks=response_chunks,
                    chunk_times=chunk_times,
                    output_tokens=output_len,
                    output_token_times=chunk_times,
                    server_usage=server_usage,
                ),
                lora_adapter=lora_adapter,
                extra_info={"raw_response": raw_content},
            )

        data = await response.json()
        prompt_len = self._count_prompt_tokens(tokenizer)
        choices = data.get("choices", [])
        if len(choices) == 0:
            return InferenceInfo(
                request_metrics=self._build_request_metrics(prompt_len, 0),
                lora_adapter=lora_adapter,
            )
        output_text = "".join([choice.get("message", {}).get("content", "") for choice in choices])
        output_len = tokenizer.count_tokens(output_text)
        return InferenceInfo(
            request_metrics=self._build_request_metrics(prompt_len, output_len),
            response_metrics=UnaryResponseMetrics(output_tokens=output_len),
            lora_adapter=lora_adapter,
        )
