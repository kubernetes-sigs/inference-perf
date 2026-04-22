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
import json
import logging
from typing import Any, Generator, List, Optional, Union

import numpy as np

from inference_perf.apis.base import InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.apis.user_session import LocalUserSession, UserSessionCompletionAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, Distribution
from inference_perf.datagen.multimodal_assets import (
    generate_png_bytes,
    resolution_to_wh,
    sample_image_resolution,
    sample_video_profile,
)
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.utils.distribution import sample_from_distribution
from .base import DataGenerator, LazyLoadDataMixin
from .datagen_utils import generate_random_exact_length_text, init_vocab_sampling, random_token_ids

logger = logging.getLogger(__name__)


# Shared Prefix Generator generates shared prefix in the prompts that are sent.
# This can be used to benchmark prefix caching cases.
class SharedPrefixDataGenerator(DataGenerator, LazyLoadDataMixin):
    @staticmethod
    def _resolve_distribution(
        param: Union[int, Distribution],
        legacy_dist: Optional[Distribution] = None,
    ) -> Distribution:
        """Resolve a Union[int, Distribution] + optional legacy Distribution into a Distribution."""
        if isinstance(param, Distribution):
            return param
        # param is an int
        if legacy_dist is not None:
            return legacy_dist
        # Fixed value: min=max=mean, std_dev=0
        return Distribution(mean=float(param), min=param, max=param, std_dev=0.0)

    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        super().__init__(api_config, config, tokenizer)

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for SharedPrefixDataGenerator but was not initialized.")

        self.vocab_size, self.special_token_ids, self.valid_token_ids = init_vocab_sampling(self.tokenizer)

        if self.shared_prefix is None:
            raise ValueError("Shared Prefix config is required for SharedPrefixDataGenerator")

        self.num_groups: int = self.shared_prefix.num_groups
        self.num_prompts_per_group: int = self.shared_prefix.num_prompts_per_group
        self.enable_multi_turn_chat: bool = self.shared_prefix.enable_multi_turn_chat

        # Deterministic seeded RNG
        self.rng: np.random.Generator = np.random.default_rng(self.shared_prefix.seed)

        self.prefix_multimodal = self.shared_prefix.multimodal
        self.payload_multimodal = config.multimodal

        # Resolve all parameters to Distribution
        system_prompt_dist = self._resolve_distribution(self.shared_prefix.system_prompt_len)
        question_dist = self._resolve_distribution(self.shared_prefix.question_len, self.shared_prefix.question_distribution)
        output_dist = self._resolve_distribution(self.shared_prefix.output_len, self.shared_prefix.output_distribution)

        # Generate per-group system prompt lengths
        self.system_prompt_lens_per_group: List[int] = sample_from_distribution(
            system_prompt_dist, self.num_groups, self.rng
        ).tolist()

        # Generate separate distributions for each group
        self.question_len_list_per_group: List[List[int]] = []
        self.output_len_list_per_group: List[List[int]] = []

        for _ in range(self.num_groups):
            question_lens = sample_from_distribution(question_dist, self.num_prompts_per_group, self.rng)
            self.question_len_list_per_group.append(question_lens.tolist())

            output_lens = sample_from_distribution(output_dist, self.num_prompts_per_group, self.rng)
            self.output_len_list_per_group.append(output_lens.tolist())

        self.prompts: list[dict[str, Any]] = []
        self.user_sessions: List[LocalUserSession] = []
        self.flat_output_lens: List[int] = []
        self._generate_prompts()

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return True

    def is_preferred_worker_requested(self) -> bool:
        return True if self.enable_multi_turn_chat else False

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        i = data.data_index % len(self.prompts)
        output_len = self.flat_output_lens[i]

        prompt_data = self.prompts[i]
        prompt_content = prompt_data["content"]
        prefix_metrics = prompt_data["metrics"]

        if self.prefix_multimodal or self.payload_multimodal:
            if self.enable_multi_turn_chat:
                raise NotImplementedError("Multimodal multi-turn shared prefix is not supported yet.")

            # Single turn multimodal
            # If it's just a string (fallback from _generate_prompts when no prefix media was generated)
            if isinstance(prompt_content, str):
                prompt_content = [{"type": "text", "text": prompt_content}]

            image_count = prefix_metrics["images"]
            video_count = prefix_metrics["videos"]
            frame_count = prefix_metrics["frames"]
            image_instances = list(prefix_metrics["image_instances"])
            video_instances = list(prefix_metrics["video_instances"])

            # Generate payload media if configured
            if self.payload_multimodal:
                # Assume the last part is the question text
                question_part = prompt_content[-1]
                if isinstance(question_part, dict) and question_part.get("type") == "text":
                    question_text = question_part["text"]
                    media_items: list[tuple[dict[str, Any], float]] = []

                    if self.payload_multimodal.image and self.payload_multimodal.image.count:
                        count = int(sample_from_distribution(self.payload_multimodal.image.count, 1, self.rng)[0])
                        image_count += count
                        for _ in range(count):
                            w, h = sample_image_resolution(self.payload_multimodal.image, self.rng)
                            png_bytes = generate_png_bytes(w, h, self.rng)
                            point = self._get_insertion_point(self.payload_multimodal.image.insertion_point)
                            media_items.append((self._build_image_block(self.payload_multimodal.image), point))
                            image_instances.append(
                                {"pixels": w * h, "bytes": len(png_bytes), "aspect_ratio": w / h if h > 0 else 0.0}
                            )

                    if self.payload_multimodal.video and self.payload_multimodal.video.count:
                        count = int(sample_from_distribution(self.payload_multimodal.video.count, 1, self.rng)[0])
                        video_count += count
                        for _ in range(count):
                            point = self._get_insertion_point(self.payload_multimodal.video.insertion_point)
                            profile = sample_video_profile(self.payload_multimodal.video, self.rng)
                            w, h = resolution_to_wh(profile.resolution)
                            dummy_png = generate_png_bytes(w, h, self.rng)
                            video_instances.append(
                                {
                                    "pixels": w * h,
                                    "bytes": len(dummy_png) * profile.frames,
                                    "aspect_ratio": w / h if h > 0 else 0.0,
                                    "frames": profile.frames,
                                }
                            )
                            for _ in range(profile.frames):
                                media_items.append((self._build_image_block_wh(w, h), point))

                    assembled_question = self._assemble_content(question_text, media_items)
                    prompt_content = prompt_content[:-1] + assembled_question

            multimodal_metrics = {
                "images": image_count,
                "videos": video_count,
                "frames": frame_count,
                "image_instances": image_instances,
                "video_instances": video_instances,
                "audio_clips": 0,
                "audio_seconds": 0.0,
                "audio_instances": [],
            }

            messages = [ChatMessage(role="user", content=prompt_content)]
            return ChatCompletionAPIData(messages=messages, max_tokens=output_len, multimodal_metrics=multimodal_metrics)

        if self.enable_multi_turn_chat:
            user_id = data.data_index % len(self.user_sessions)
            round = data.data_index // len(self.user_sessions)
            return UserSessionCompletionAPIData(
                prompt=prompt_content,
                max_tokens=output_len,
                user_session_id=self.user_sessions[user_id].user_session_id,
                target_round=round,
            )
        else:
            return CompletionAPIData(prompt=prompt_content, max_tokens=output_len)

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if not self.prompts:
            return

        i = 0
        while True:
            preferred_worker_id = i % self.num_groups if self.enable_multi_turn_chat else -1
            yield LazyLoadInferenceAPIData(data_index=i, preferred_worker_id=preferred_worker_id)
            i += 1

    def _generate_random_token_ids(self, length: int) -> List[int]:
        """Generates a list of random token IDs of a specified length."""
        return random_token_ids(self.rng, self.valid_token_ids, length)

    def _generate_exact_length_text(self, target_len: int, prefix_text: str = "") -> str:
        """Generates a string that tokenizes to exactly target_len, optionally prefixed.

        If prefix_text is provided, the TOTAL length including prefix will be target_len.
        Returns the full combined text if prefix_text is provided, else just the generated text.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generating exact length prompts.")
        return generate_random_exact_length_text(self.rng, self.valid_token_ids, self.tokenizer, target_len, prefix_text)

    def _build_image_block(self, img_cfg: Any) -> dict[str, Any]:
        w, h = sample_image_resolution(img_cfg, self.rng)
        return self._build_image_block_wh(w, h)

    def _build_image_block_wh(self, width: int, height: int) -> dict[str, Any]:
        png_bytes = generate_png_bytes(width, height, self.rng)
        data_url = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"
        return {"type": "image_url", "image_url": {"url": data_url}}

    def _get_insertion_point(self, config_insertion_point: Optional[Union[float, Distribution]]) -> float:
        if config_insertion_point is None:
            return float(self.rng.uniform(0.0, 1.0))
        if isinstance(config_insertion_point, float):
            return config_insertion_point
        return float(sample_from_distribution(config_insertion_point, 1, self.rng)[0])

    def _assemble_content(self, text_str: str, media_items: list[tuple[dict[str, Any], float]]) -> list[dict[str, Any]]:
        if not media_items:
            return [{"type": "text", "text": text_str}]

        media_items.sort(key=lambda x: x[1])
        content = []
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

    def _generate_prompts(self) -> None:
        """Pre-generates all prompts based on the configuration."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not available for generating prompts.")

        if self.shared_prefix is None:
            raise ValueError("Shared prefix is not available for generating prompts.")

        for group_id in range(self.num_groups):
            # Generate a shared prefix (system prompt) with per-group length
            sys_prompt_len = self.system_prompt_lens_per_group[group_id]
            shared_prefix_text = self._generate_exact_length_text(sys_prompt_len)

            # Generate prefix-side media payloads (image/video) and stitch them
            # into the shared prefix at the configured insertion points. Media
            # items add wire bytes only — they don't contribute to sys_prompt_len
            # (which is a text-token target).
            media_items: list[tuple[dict[str, Any], float]] = []
            image_count = 0
            video_count = 0
            frame_count = 0
            image_instances = []
            video_instances = []

            if self.prefix_multimodal:
                if self.prefix_multimodal.image and self.prefix_multimodal.image.count:
                    count = int(sample_from_distribution(self.prefix_multimodal.image.count, 1, self.rng)[0])
                    image_count = count
                    for _ in range(count):
                        w, h = sample_image_resolution(self.prefix_multimodal.image, self.rng)
                        png_bytes = generate_png_bytes(w, h, self.rng)
                        point = self._get_insertion_point(self.prefix_multimodal.image.insertion_point)
                        media_items.append((self._build_image_block(self.prefix_multimodal.image), point))
                        image_instances.append(
                            {"pixels": w * h, "bytes": len(png_bytes), "aspect_ratio": w / h if h > 0 else 0.0}
                        )

                # Each video expands to the configured number of frames at the
                # configured resolution, emitted as image_url parts (this
                # datagen's established video representation).
                if self.prefix_multimodal.video and self.prefix_multimodal.video.count:
                    count = int(sample_from_distribution(self.prefix_multimodal.video.count, 1, self.rng)[0])
                    video_count = count
                    for _ in range(count):
                        point = self._get_insertion_point(self.prefix_multimodal.video.insertion_point)
                        profile = sample_video_profile(self.prefix_multimodal.video, self.rng)
                        w, h = resolution_to_wh(profile.resolution)
                        frame_count += profile.frames
                        dummy_png = generate_png_bytes(w, h, self.rng)
                        video_instances.append(
                            {
                                "pixels": w * h,
                                "bytes": len(dummy_png) * profile.frames,
                                "aspect_ratio": w / h if h > 0 else 0.0,
                                "frames": profile.frames,
                            }
                        )
                        for _ in range(profile.frames):
                            media_items.append((self._build_image_block_wh(w, h), point))

            prefix_metrics = {
                "images": image_count,
                "videos": video_count,
                "frames": frame_count,
                "image_instances": image_instances,
                "video_instances": video_instances,
            }

            shared_prefix_content = self._assemble_content(shared_prefix_text, media_items)
            prefix_is_structured = len(shared_prefix_content) > 1 or shared_prefix_content[0]["type"] != "text"

            for prompt_id in range(self.num_prompts_per_group):
                q_len = self.question_len_list_per_group[group_id][prompt_id]
                target_total_len = sys_prompt_len + q_len

                # #383's exact-length helper: returns full prefix+question text
                # whose tokenized length equals target_total_len.
                full_text = self._generate_exact_length_text(target_total_len, prefix_text=shared_prefix_text)
                # Extract question part (skip prefix and space)
                question_text = full_text[len(shared_prefix_text) + 1 :]

                if self.enable_multi_turn_chat:
                    # If the prefix is structured (i.e. has media blocks) we JSON serialize
                    # it so it round-trips through the user-session context. Currently
                    # unreachable because load_lazy_data blocks multimodal+multi-turn —
                    # kept defensively for the day that combination is supported.
                    context_str = json.dumps(shared_prefix_content) if prefix_is_structured else shared_prefix_text

                    self.user_sessions.append(
                        LocalUserSession(
                            user_session_id=f"user_session_{self.num_prompts_per_group * group_id + prompt_id}",
                            context=context_str,
                        )
                    )
                    self.prompts.append({"content": question_text, "metrics": prefix_metrics})
                elif prefix_is_structured:
                    # Single-turn with multimodal prefix: append question as a text part.
                    full_content = shared_prefix_content + [{"type": "text", "text": question_text}]
                    self.prompts.append({"content": full_content, "metrics": prefix_metrics})
                else:
                    self.prompts.append({"content": full_text, "metrics": prefix_metrics})

        # Flatten output lengths to match prompts ordering
        self.flat_output_lens = [
            self.output_len_list_per_group[g][p] for g in range(self.num_groups) for p in range(self.num_prompts_per_group)
        ]

        # Shuffle using seeded RNG for reproducibility
        indices = self.rng.permutation(len(self.prompts))
        self.prompts = [self.prompts[i] for i in indices]
        self.flat_output_lens = [self.flat_output_lens[i] for i in indices]
        if self.enable_multi_turn_chat:
            self.user_sessions = [self.user_sessions[i] for i in indices]
