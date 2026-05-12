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
from typing import Dict, Generator, List, Optional, Union

import numpy as np

from inference_perf.apis.base import InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.apis.user_session import LocalUserSession, UserSessionCompletionAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, Distribution, SyntheticMultimodalDatagenConfig
from inference_perf.datagen.multimodal_sampling import (
    resolution_to_wh,
    sample_audio_duration,
    sample_image_resolution,
    sample_insertion_point,
    sample_video_profile,
)
from inference_perf.payloads import (
    ImageRepresentation,
    MultimodalSpec,
    SyntheticAudioSpec,
    SyntheticFramesVideoSpec,
    SyntheticImageSpec,
    SyntheticMp4VideoSpec,
    VideoRepresentation,
    VideoSpecUnion,
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

        self.prefix_multimodal: Optional[SyntheticMultimodalDatagenConfig] = self.shared_prefix.multimodal
        self.payload_multimodal: Optional[SyntheticMultimodalDatagenConfig] = config.multimodal

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

        # Per-prompt storage, all parallel (same length after _generate_prompts).
        # ``prompts[i]`` is the full prompt text (prefix + " " + question);
        # ``prefix_texts[i]`` is just the shared system prompt portion (empty
        # string when no prefix is configured).  ``prompt_groups[i]`` is the
        # group_id used to look up per-group prefix multimodal specs.
        self.prompts: List[str] = []
        self.prefix_texts: List[str] = []
        self.prompt_groups: List[int] = []

        # Prefix-side multimodal spec per group, sampled once at init. Reused
        # across all requests in the group; combined with deterministic byte
        # materialization in chat.py, this gives identical wire bytes across
        # requests in a group → server prefix-cache hits.
        self.prefix_specs_by_group: Dict[int, MultimodalSpec] = {}

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

    def _sample_payload_spec(self) -> Optional[MultimodalSpec]:
        """Sample a fresh payload-side multimodal spec for one request."""
        if not self.payload_multimodal:
            return None
        return _sample_spec(self.payload_multimodal, self.rng)

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        i = data.data_index % len(self.prompts)
        output_len = self.flat_output_lens[i]

        if self.prefix_multimodal or self.payload_multimodal:
            if self.enable_multi_turn_chat:
                raise NotImplementedError("Multimodal multi-turn shared prefix is not supported yet.")

            group_id = self.prompt_groups[i]
            prefix_text = self.prefix_texts[i]
            # Question text is everything after the prefix + separator space.
            question_text = self.prompts[i][len(prefix_text) + 1 :] if prefix_text else self.prompts[i]
            return ChatCompletionAPIData(
                messages=[ChatMessage(role="user", content=question_text)],
                prefix_text=prefix_text or None,
                prefix_multimodal_spec=self.prefix_specs_by_group.get(group_id),
                multimodal_spec=self._sample_payload_spec(),
                max_tokens=output_len,
            )

        if self.enable_multi_turn_chat:
            user_id = data.data_index % len(self.user_sessions)
            round = data.data_index // len(self.user_sessions)
            return UserSessionCompletionAPIData(
                prompt=self.prompts[i],
                max_tokens=output_len,
                user_session_id=self.user_sessions[user_id].user_session_id,
                target_round=round,
            )
        return CompletionAPIData(prompt=self.prompts[i], max_tokens=output_len)

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

    def _generate_prompts(self) -> None:
        """Pre-generates all per-group prefix texts/specs and per-prompt question texts."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not available for generating prompts.")

        if self.shared_prefix is None:
            raise ValueError("Shared prefix is not available for generating prompts.")

        for group_id in range(self.num_groups):
            sys_prompt_len = self.system_prompt_lens_per_group[group_id]
            shared_prefix_text = self._generate_exact_length_text(sys_prompt_len)

            if self.prefix_multimodal:
                # Sample the prefix-side spec once per group. Bytes are
                # materialized at request time in chat.py with deterministic
                # per-instance seeding so requests in this group produce
                # identical wire bytes (server prefix-cache hits).
                self.prefix_specs_by_group[group_id] = _sample_spec(self.prefix_multimodal, self.rng)

            for prompt_id in range(self.num_prompts_per_group):
                q_len = self.question_len_list_per_group[group_id][prompt_id]
                target_total_len = sys_prompt_len + q_len

                # #383's exact-length helper: returns full prefix+question text
                # whose tokenized length equals target_total_len.
                full_text = self._generate_exact_length_text(target_total_len, prefix_text=shared_prefix_text)

                self.prompts.append(full_text)
                self.prefix_texts.append(shared_prefix_text)
                self.prompt_groups.append(group_id)

                if self.enable_multi_turn_chat:
                    # Multi-turn is text-only (multimodal+multi-turn raises in
                    # load_lazy_data). The shared prefix serves as the session's
                    # initial context.
                    self.user_sessions.append(
                        LocalUserSession(
                            user_session_id=f"user_session_{self.num_prompts_per_group * group_id + prompt_id}",
                            context=shared_prefix_text,
                        )
                    )

        # Flatten output lengths to match prompts ordering
        self.flat_output_lens = [
            self.output_len_list_per_group[g][p] for g in range(self.num_groups) for p in range(self.num_prompts_per_group)
        ]

        # Shuffle using seeded RNG for reproducibility. Group ids and prefix
        # texts shuffle alongside so per-prompt lookups remain consistent.
        indices = self.rng.permutation(len(self.prompts))
        self.prompts = [self.prompts[i] for i in indices]
        self.prefix_texts = [self.prefix_texts[i] for i in indices]
        self.prompt_groups = [self.prompt_groups[i] for i in indices]
        self.flat_output_lens = [self.flat_output_lens[i] for i in indices]
        if self.enable_multi_turn_chat:
            self.user_sessions = [self.user_sessions[i] for i in indices]


def _sample_spec(cfg: SyntheticMultimodalDatagenConfig, rng: np.random.Generator) -> MultimodalSpec:
    """Sample a ``MultimodalSpec`` from a multimodal config block.

    Shared between prefix-side (sampled once per group at init) and
    payload-side (sampled fresh per request).
    """
    spec = MultimodalSpec()

    img_cfg = cfg.image
    if img_cfg and img_cfg.count:
        count = int(sample_from_distribution(img_cfg.count, 1, rng)[0])
        for _ in range(count):
            w, h = sample_image_resolution(img_cfg, rng)
            spec.images.append(
                SyntheticImageSpec(
                    width=w,
                    height=h,
                    insertion_point=sample_insertion_point(img_cfg.insertion_point, rng),
                    representation=img_cfg.representation,
                )
            )

    vid_cfg = cfg.video
    if vid_cfg and vid_cfg.count:
        count = int(sample_from_distribution(vid_cfg.count, 1, rng)[0])
        for _ in range(count):
            profile = sample_video_profile(vid_cfg, rng)
            w, h = resolution_to_wh(profile.resolution)
            insertion_point = sample_insertion_point(vid_cfg.insertion_point, rng)
            video_spec: VideoSpecUnion
            if vid_cfg.representation == VideoRepresentation.MP4:
                video_spec = SyntheticMp4VideoSpec(width=w, height=h, frames=profile.frames, insertion_point=insertion_point)
            else:
                video_spec = SyntheticFramesVideoSpec(
                    width=w,
                    height=h,
                    frames=profile.frames,
                    insertion_point=insertion_point,
                    frame_representation=(
                        ImageRepresentation.JPEG
                        if vid_cfg.representation == VideoRepresentation.JPEG_FRAMES
                        else ImageRepresentation.PNG
                    ),
                )
            spec.videos.append(video_spec)

    aud_cfg = cfg.audio
    if aud_cfg and aud_cfg.count:
        count = int(sample_from_distribution(aud_cfg.count, 1, rng)[0])
        for _ in range(count):
            spec.audios.append(
                SyntheticAudioSpec(
                    duration=sample_audio_duration(aud_cfg, rng),
                    insertion_point=sample_insertion_point(aud_cfg.insertion_point, rng),
                )
            )

    return spec
