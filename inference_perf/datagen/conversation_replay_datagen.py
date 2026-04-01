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
"""Conversation replay data generator for agentic workload benchmarking.

Closed-loop continuous replenishment model
------------------------------------------
Each slot (preferred_worker_id) maps to one concurrent conversation. When a
conversation completes all its turns, the slot immediately resets to a fresh
LocalUserSession and starts a new conversation from turn 0. This models
steady-state production traffic where a new conversation begins as soon as
the previous one ends.

At steady state, the C active conversations are uniformly distributed across
turn 0..N, so the mean KV-cache context across all active slots matches the
production mean across all active slots.

Usage
-----
Set ``num_conversations = C`` (the concurrency level) so each slot owns
exactly one conversation. Set ``num_requests = C × turns × num_rounds`` to
run ``num_rounds`` complete conversations per slot. The first round is
warmup; report throughput from round 2 onward.

Run each concurrency level as a separate benchmark (fresh state, not stages
of one run) to avoid context accumulating across concurrency levels.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Generator, List, Optional

import numpy as np

from aiohttp import ClientResponse
from inference_perf.apis.base import InferenceAPIData, InferenceInfo, LazyLoadInferenceAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.apis.user_session import LocalUserSession, UserSessionCompletionAPIData
from inference_perf.config import (
    APIConfig,
    APIType,
    ConversationReplayConfig,
    DataConfig,
    Distribution,
)
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.utils.distribution import sample_from_distribution

from .base import DataGenerator, LazyLoadDataMixin

logger = logging.getLogger(__name__)


class _ConversationReplayAPIData(UserSessionCompletionAPIData):
    """UserSessionCompletionAPIData with tool call latency simulation.

    After the model generates a response, sleeps for ``tool_call_latency_sec``
    seconds *before* releasing the session lock. This correctly serialises:

        model inference → sleep(tool_call_latency) → next model inference

    while allowing the GPU to serve other concurrent conversations during the
    sleep (the asyncio event loop is not blocked; other slots' requests run).

    If ``tool_call_latency_sec == 0`` the behaviour is identical to the
    parent class.
    """

    tool_call_latency_sec: float = 0.0

    async def process_response(
        self,
        response: ClientResponse,
        config: APIConfig,
        tokenizer: CustomTokenizer,
        lora_adapter: Optional[str] = None,
    ) -> InferenceInfo:
        # Run the base completion response handler (sets self.model_response,
        # records timing metrics) WITHOUT yet releasing the session lock.
        # We call CompletionAPIData directly to skip UserSessionCompletionAPIData's
        # update_context call so we can inject the sleep in between.
        inference_info = await CompletionAPIData.process_response(self, response, config, tokenizer, lora_adapter)
        self.update_inference_info(inference_info)

        # Simulate tool execution latency while holding the session lock.
        # The next turn of this conversation cannot start until the sleep
        # completes; other conversations' turns run freely during the wait.
        if self.tool_call_latency_sec > 0:
            await asyncio.sleep(self.tool_call_latency_sec)

        # Release the session lock by updating context (allows next turn).
        self.user_session.update_context(self.prompt + " " + self.model_response)
        return inference_info

    async def process_failure(
        self,
        response: Optional[ClientResponse],
        config: APIConfig,
        tokenizer: CustomTokenizer,
        exception: Exception,
        lora_adapter: Optional[str] = None,
    ) -> Optional[InferenceInfo]:
        # On failure, release the lock without sleeping (no tool was called).
        inference_info = InferenceInfo()
        self.update_inference_info(inference_info)
        self.user_session.update_context(self._session_context)
        return inference_info


@dataclass
class ConversationBlueprint:
    """Pre-computed plan for a single conversation."""

    conversation_id: int
    num_turns: int
    system_prompt: str
    turn_prompts: List[str] = field(default_factory=list)
    turn_output_lens: List[int] = field(default_factory=list)
    turn_tool_call_latencies: List[float] = field(default_factory=list)


class ConversationReplayDataGenerator(DataGenerator, LazyLoadDataMixin):
    """Generates synthetic multi-turn conversations from distribution configs.

    Each conversation has:
    - A two-part system prompt (shared prefix + dynamic per-conversation suffix)
    - N turns with independently sampled input/output token lengths
    - Sequential turn enforcement via LocalUserSession

    Conversations are dispatched round-robin across workers using
    preferred_worker_id for affinity. When all turns of a conversation are
    exhausted, the session resets and replays from the beginning (recycling).
    """

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
    ) -> None:
        super().__init__(api_config, config, tokenizer)

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for ConversationReplayDataGenerator.")

        cr_config = config.conversation_replay
        if cr_config is None:
            raise ValueError("conversation_replay config is required.")
        self.cr_config: ConversationReplayConfig = cr_config

        # Tokenizer vocab size for random token generation
        hf_tokenizer = self.tokenizer.get_tokenizer()
        if hasattr(hf_tokenizer, "vocab_size") and hf_tokenizer.vocab_size is not None:
            self.vocab_size: int = hf_tokenizer.vocab_size
        elif hasattr(hf_tokenizer, "get_vocab") and callable(hf_tokenizer.get_vocab):
            self.vocab_size = len(hf_tokenizer.get_vocab())
        else:
            try:
                self.vocab_size = len(hf_tokenizer)
            except TypeError as e:
                raise ValueError("Cannot determine tokenizer vocabulary size.") from e
        if self.vocab_size <= 0:
            raise ValueError(f"Tokenizer vocabulary size must be positive, got {self.vocab_size}.")

        # Seeded RNG for deterministic generation
        self.rng = np.random.default_rng(self.cr_config.seed)

        # Build conversation blueprints
        self.blueprints: List[ConversationBlueprint] = []
        self.user_sessions: List[LocalUserSession] = []
        self._build_conversations()

        logger.info(
            "ConversationReplayDataGenerator: %d conversations, %d total turns",
            len(self.blueprints),
            sum(bp.num_turns for bp in self.blueprints),
        )

    # -- BaseGenerator interface ------------------------------------------

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return False

    def is_preferred_worker_requested(self) -> bool:
        return True

    # -- LazyLoadDataMixin interface --------------------------------------

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        conv_idx = data.data_index % len(self.blueprints)
        bp = self.blueprints[conv_idx]
        round_num = data.data_index // len(self.blueprints)
        turn_idx = round_num % bp.num_turns
        convo_num = round_num // bp.num_turns  # which conversation this slot is on

        # Closed-loop replenishment: when a conversation finishes all its turns,
        # reset the session so the slot immediately starts a fresh conversation.
        # This happens in the worker process (after fork), so each worker safely
        # manages its own copy of the session for its assigned slots.
        if turn_idx == 0 and round_num > 0:
            self.user_sessions[conv_idx] = self._new_session(
                user_session_id=f"slot_{conv_idx}_convo_{convo_num}",
                context=bp.system_prompt,
            )
            logger.debug("Slot %d starting conversation %d", conv_idx, convo_num)
        elif len(self.user_sessions[conv_idx].context) > 2_700_000:
            # Safety reset: context is approaching max_model_len.
            # Random Qwen3 tokens decode to ~12 chars/token on average, so
            # 225K tokens ≈ 2.7M chars. Reset to fresh context rather than
            # letting vLLM reject the request due to exceeding max_model_len.
            self.user_sessions[conv_idx] = self._new_session(
                user_session_id=f"slot_{conv_idx}_convo_{convo_num}_reset",
                context=bp.system_prompt,
            )
            turn_idx = 0
            logger.warning("Slot %d: safety context reset (context >2.7M chars)", conv_idx)

        latency = bp.turn_tool_call_latencies[turn_idx] if bp.turn_tool_call_latencies else 0.0
        return _ConversationReplayAPIData(
            prompt=bp.turn_prompts[turn_idx],
            max_tokens=bp.turn_output_lens[turn_idx],
            user_session_id=self.user_sessions[conv_idx].user_session_id,
            target_round=round_num,
            tool_call_latency_sec=latency,
        )

    # -- DataGenerator interface ------------------------------------------

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if not self.blueprints:
            return

        i = 0
        while True:
            conv_idx = i % len(self.blueprints)
            yield LazyLoadInferenceAPIData(
                data_index=i,
                preferred_worker_id=conv_idx,
            )
            i += 1

    # -- Internal ---------------------------------------------------------

    @staticmethod
    def _new_session(user_session_id: str, context: str) -> LocalUserSession:
        """Create a LocalUserSession and register it in the class-level registry
        so that UserSessionCompletionAPIData.user_session (which looks up via
        ``LocalUserSession.get_instance``) resolves to it.
        """
        session = LocalUserSession(user_session_id=user_session_id, context=context)
        LocalUserSession._instances[user_session_id] = session
        return session

    def _sample_distribution(self, dist: Distribution, count: int) -> List[int]:
        """Sample ``count`` values from a Distribution."""
        arr = sample_from_distribution(dist, count, rng=self.rng)
        return [int(v) for v in arr]

    def _generate_random_token_text(self, num_tokens: int) -> str:
        """Generate random text that is approximately ``num_tokens`` long."""
        if num_tokens <= 0:
            return ""
        assert self.tokenizer is not None
        hf_tokenizer = self.tokenizer.get_tokenizer()
        token_ids = self.rng.integers(0, self.vocab_size, size=num_tokens).tolist()
        return str(hf_tokenizer.decode(token_ids, skip_special_tokens=True))

    def _build_conversations(self) -> None:
        """Pre-generate all conversation blueprints deterministically."""
        cfg = self.cr_config
        n = cfg.num_conversations

        # Sample per-conversation parameters
        if cfg.turns_per_conversation is not None:
            turn_counts = self._sample_distribution(cfg.turns_per_conversation, n)
        else:
            turn_counts = [10] * n  # default fallback

        if cfg.dynamic_system_prompt_len is not None:
            dynamic_lens = self._sample_distribution(cfg.dynamic_system_prompt_len, n)
        else:
            dynamic_lens = [0] * n

        # Generate shared system prompt once
        shared_prompt_text = self._generate_random_token_text(cfg.shared_system_prompt_len)

        total_turns = sum(turn_counts)
        logger.info(
            "Building %d conversations (%d total turns, shared prompt %d tokens)",
            n,
            total_turns,
            cfg.shared_system_prompt_len,
        )

        # Sample all turn-level parameters at once for efficiency
        if cfg.input_tokens_per_turn is not None:
            all_input_lens = self._sample_distribution(cfg.input_tokens_per_turn, total_turns)
        else:
            all_input_lens = [512] * total_turns

        if cfg.output_tokens_per_turn is not None:
            all_output_lens = self._sample_distribution(cfg.output_tokens_per_turn, total_turns)
        else:
            all_output_lens = [256] * total_turns

        if cfg.tool_call_latency_sec is not None:
            # Sample latencies as floats (seconds); re-use the same distribution
            # machinery but convert from the integer output to float seconds.
            all_tool_latencies: List[float] = [
                float(v) for v in self._sample_distribution(cfg.tool_call_latency_sec, total_turns)
            ]
        else:
            all_tool_latencies = []

        # Build each conversation
        turn_offset = 0
        for conv_id in range(n):
            num_turns = turn_counts[conv_id]

            # Two-part system prompt: shared prefix + dynamic suffix
            dynamic_text = self._generate_random_token_text(dynamic_lens[conv_id])
            system_prompt = shared_prompt_text + " " + dynamic_text if dynamic_text else shared_prompt_text

            # Generate turn prompts via batch decode
            turn_input_lens = all_input_lens[turn_offset : turn_offset + num_turns]
            turn_output_lens_list = all_output_lens[turn_offset : turn_offset + num_turns]
            turn_tool_latencies = all_tool_latencies[turn_offset : turn_offset + num_turns] if all_tool_latencies else []
            turn_offset += num_turns

            # Batch generate token IDs then decode
            turn_prompts: List[str] = []
            for tlen in turn_input_lens:
                turn_prompts.append(self._generate_random_token_text(tlen))

            bp = ConversationBlueprint(
                conversation_id=conv_id,
                num_turns=num_turns,
                system_prompt=system_prompt,
                turn_prompts=turn_prompts,
                turn_output_lens=turn_output_lens_list,
                turn_tool_call_latencies=turn_tool_latencies,
            )
            self.blueprints.append(bp)

            # Create a LocalUserSession with the system prompt as initial context
            self.user_sessions.append(
                self._new_session(
                    user_session_id=f"conv_{conv_id}",
                    context=system_prompt,
                )
            )
