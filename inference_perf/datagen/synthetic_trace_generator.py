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
"""Synthetic OTel trace generator for agentic workload benchmarking.

Generates deterministic OTel JSON trace files from configurable distributions.
Each file represents one multi-turn conversation compatible with the
otel_trace_replay pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

if TYPE_CHECKING:
    from inference_perf.config import SyntheticTraceConfig, SyntheticTraceDistribution

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distribution sampling
# ---------------------------------------------------------------------------


def sample_distribution(
    dist_type: str, dist_min: int, dist_max: int, mean: float, std_dev: float, count: int, rng: np.random.RandomState
) -> Any:
    """Sample from a distribution, returning integer values clipped to [min, max]."""
    if dist_type == "fixed":
        values = np.full(count, mean)
    elif dist_type == "uniform":
        values = rng.uniform(dist_min, dist_max, size=count)
    elif dist_type == "lognormal":
        target_mean = mean - dist_min
        if target_mean <= 0:
            values = np.full(count, dist_min)
        else:
            sigma = np.log(1 + (std_dev / max(target_mean, 1)) ** 2) ** 0.5
            mu = np.log(target_mean) - sigma**2 / 2
            values = rng.lognormal(mean=mu, sigma=sigma, size=count) + dist_min
    elif dist_type == "normal":
        values = rng.normal(loc=mean, scale=std_dev, size=count)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

    return np.clip(np.round(values), dist_min, dist_max).astype(int)


def _sample_from_config(
    config: "SyntheticTraceDistribution | None", count: int, rng: np.random.RandomState, default_value: int = 10
) -> Any:
    """Sample from a SyntheticTraceDistribution config, or return fixed default."""
    if config is None:
        return np.full(count, default_value, dtype=int)
    return sample_distribution(config.type, config.min, config.max, config.mean, config.std_dev, count, rng)


# ---------------------------------------------------------------------------
# Conversation blueprint
# ---------------------------------------------------------------------------


@dataclass
class TurnBlueprint:
    input_tokens: int
    output_tokens: int


@dataclass
class ConversationBlueprint:
    conversation_id: str
    system_prompt: str
    turns: List[TurnBlueprint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Random text generation
# ---------------------------------------------------------------------------


def generate_random_text(rng: np.random.RandomState, num_tokens: int, vocab_size: int = 30000) -> str:
    """Generate random text approximating a target token count."""
    if num_tokens <= 0:
        return ""
    words_needed = max(1, int(num_tokens * 0.8))
    word_ids = rng.randint(0, vocab_size, size=words_needed)
    return " ".join(f"w{wid}" for wid in word_ids)


# ---------------------------------------------------------------------------
# OTel trace generation
# ---------------------------------------------------------------------------


def _generate_otel_trace(
    blueprint: ConversationBlueprint, rng: np.random.RandomState, model_name: str = "gpt-4"
) -> Dict[str, Any]:
    """Generate an OTel JSON trace from a conversation blueprint."""
    trace_id = blueprint.conversation_id
    spans: List[Dict[str, Any]] = []
    messages_so_far: List[Dict[str, str]] = [{"role": "system", "content": blueprint.system_prompt}]
    base_time = datetime(2026, 1, 1, 0, 0, 0)
    current_time = base_time

    for turn_idx, turn in enumerate(blueprint.turns):
        user_text = generate_random_text(rng, turn.input_tokens)
        messages_so_far.append({"role": "user", "content": user_text})

        assistant_text = generate_random_text(rng, turn.output_tokens)
        input_tokens = sum(len(m["content"]) // 4 for m in messages_so_far)
        output_tokens = len(assistant_text) // 4

        start_time = current_time
        end_time = current_time + timedelta(seconds=2)
        current_time = end_time + timedelta(milliseconds=100)

        spans.append(
            {
                "trace_id": trace_id,
                "span_id": f"turn_{turn_idx}",
                "parent_span_id": None,
                "name": f"chat {model_name}",
                "kind": "SPAN_KIND_INTERNAL",
                "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "attributes": {
                    "exgentic.session.id": trace_id,
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": "openai",
                    "gen_ai.request.model": model_name,
                    "gen_ai.request.max_tokens": turn.output_tokens,
                    "gen_ai.usage.input_tokens": input_tokens,
                    "gen_ai.usage.output_tokens": output_tokens,
                    "gen_ai.input.messages": json.dumps(messages_so_far),
                    "gen_ai.output.text": assistant_text,
                },
                "resource_attributes": {"service.name": "synthetic-bench"},
                "status": {"code": 1, "message": ""},
            }
        )

        messages_so_far.append({"role": "assistant", "content": assistant_text})

    return {
        "trace_id": trace_id,
        "span_count": len(spans),
        "collected_at": base_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "spans": spans,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_traces_to_dir(config: "SyntheticTraceConfig", output_dir: Path, model_name: str = "gpt-4") -> int:
    """Generate synthetic OTel traces from a SyntheticTraceConfig into output_dir.

    Returns the number of trace files generated.
    """
    rng = np.random.RandomState(config.seed)

    shared_prefix_text = generate_random_text(rng, config.shared_system_prompt_len)

    turn_counts = _sample_from_config(config.turns_per_conversation, config.num_conversations, rng, default_value=4)
    dynamic_lens = _sample_from_config(config.dynamic_system_prompt_len, config.num_conversations, rng, default_value=2000)

    total_turns = int(turn_counts.sum())
    all_input_tokens = _sample_from_config(config.input_tokens_per_turn, total_turns, rng, default_value=1400)
    all_output_tokens = _sample_from_config(config.output_tokens_per_turn, total_turns, rng, default_value=526)

    blueprints: List[ConversationBlueprint] = []
    turn_cursor = 0

    for i in range(config.num_conversations):
        n_turns = int(turn_counts[i])
        dynamic_suffix_text = generate_random_text(rng, int(dynamic_lens[i]))
        system_prompt = shared_prefix_text + "\n\n" + dynamic_suffix_text

        turns = []
        for _ in range(n_turns):
            turns.append(
                TurnBlueprint(
                    input_tokens=int(all_input_tokens[turn_cursor]), output_tokens=int(all_output_tokens[turn_cursor])
                )
            )
            turn_cursor += 1

        blueprints.append(ConversationBlueprint(conversation_id=f"conv_{i:04d}", system_prompt=system_prompt, turns=turns))

    output_dir.mkdir(parents=True, exist_ok=True)
    for bp in blueprints:
        trace = _generate_otel_trace(bp, rng, model_name=model_name)
        with open(output_dir / f"{bp.conversation_id}.json", "w") as f:
            json.dump(trace, f, indent=2)

    logger.info(
        "Generated %d synthetic traces (seed=%d, total_turns=%d) in %s",
        config.num_conversations,
        config.seed,
        total_turns,
        output_dir,
    )
    return config.num_conversations
