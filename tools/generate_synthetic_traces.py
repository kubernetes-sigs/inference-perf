#!/usr/bin/env python3
"""Generate synthetic OTel trace files for agentic workload benchmarking.

Produces OTel-format JSON trace files compatible with inference-perf's
otel_trace_replay data type (PR #372). Each file represents one multi-turn
conversation with a shared system prompt prefix for prefix caching.

Usage:
    python generate_synthetic_traces.py --config synthetic_trace_config.yaml
    python generate_synthetic_traces.py --config synthetic_trace_config.yaml --stats
"""

import argparse
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Distribution sampling
# ---------------------------------------------------------------------------

@dataclass
class DistributionConfig:
    type: str = "normal"  # normal, lognormal, uniform, fixed
    min: int = 10
    max: int = 1024
    mean: float = 512
    std_dev: float = 200


def sample_distribution(config: DistributionConfig, count: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample from a distribution, returning integer values clipped to [min, max]."""
    if config.type == "fixed":
        values = np.full(count, config.mean)
    elif config.type == "uniform":
        values = rng.uniform(config.min, config.max, size=count)
    elif config.type == "lognormal":
        # Parameterize lognormal to approximate target mean within [min, max].
        # We shift so that min maps to 0, then use lognormal for the remainder.
        target_mean = config.mean - config.min
        if target_mean <= 0:
            values = np.full(count, config.min)
        else:
            # sigma controls the skew; derive from std_dev relative to mean
            sigma = np.log(1 + (config.std_dev / max(target_mean, 1)) ** 2) ** 0.5
            mu = np.log(target_mean) - sigma**2 / 2
            values = rng.lognormal(mean=mu, sigma=sigma, size=count) + config.min
    elif config.type == "normal":
        values = rng.normal(loc=config.mean, scale=config.std_dev, size=count)
    else:
        raise ValueError(f"Unknown distribution type: {config.type}")

    return np.clip(np.round(values), config.min, config.max).astype(int)


def parse_distribution(d) -> DistributionConfig:
    """Parse a distribution config from YAML (dict or scalar)."""
    if isinstance(d, (int, float)):
        return DistributionConfig(type="fixed", min=int(d), max=int(d), mean=float(d), std_dev=0)
    if isinstance(d, dict):
        return DistributionConfig(**d)
    raise ValueError(f"Invalid distribution config: {d}")


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
    system_prompt: str  # shared_prefix + dynamic_suffix
    turns: List[TurnBlueprint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Random text generation
# ---------------------------------------------------------------------------

def generate_random_text(rng: np.random.RandomState, num_tokens: int, vocab_size: int = 30000) -> str:
    """Generate random text that approximates a target token count.

    Uses random words from a simple vocabulary to produce roughly `num_tokens`
    tokens. The exact count depends on the tokenizer, but this gives a good
    approximation (~1.3 tokens per word on average for English-like text).
    """
    if num_tokens <= 0:
        return ""
    # Approximate: 1 token ≈ 4 chars, so generate num_tokens * 4 chars worth
    # Use random token IDs decoded as hex strings for consistent tokenization
    words_needed = max(1, int(num_tokens * 0.8))  # ~1.25 tokens per word
    word_ids = rng.randint(0, vocab_size, size=words_needed)
    words = [f"w{wid}" for wid in word_ids]
    return " ".join(words)


# ---------------------------------------------------------------------------
# OTel trace generation
# ---------------------------------------------------------------------------

def generate_otel_trace(
    blueprint: ConversationBlueprint,
    rng: np.random.RandomState,
    model_name: str = "gpt-4",
) -> dict:
    """Generate an OTel JSON trace from a conversation blueprint.

    Each turn becomes one span. The input messages accumulate across turns
    (system + all previous user/assistant pairs + current user message).
    The output is a placeholder assistant response.
    """
    trace_id = blueprint.conversation_id
    spans = []
    messages_so_far = [{"role": "system", "content": blueprint.system_prompt}]
    base_time = datetime(2026, 1, 1, 0, 0, 0)
    current_time = base_time
    prev_span_id = None

    for turn_idx, turn in enumerate(blueprint.turns):
        span_id = f"turn_{turn_idx}"

        # Add user message for this turn
        user_text = generate_random_text(rng, turn.input_tokens)
        messages_so_far.append({"role": "user", "content": user_text})

        # Generate placeholder assistant response
        assistant_text = generate_random_text(rng, turn.output_tokens)

        # Compute approximate token counts
        input_tokens = sum(len(m["content"]) // 4 for m in messages_so_far)
        output_tokens = len(assistant_text) // 4

        start_time = current_time
        # Simulate ~2s per turn for timing
        end_time = current_time + timedelta(seconds=2)
        current_time = end_time + timedelta(milliseconds=100)

        span = {
            "trace_id": trace_id,
            "span_id": span_id,
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
            "resource_attributes": {
                "service.name": "synthetic-bench",
            },
            "status": {
                "code": 1,
                "message": "",
            },
        }
        spans.append(span)

        # Add assistant response to conversation history
        messages_so_far.append({"role": "assistant", "content": assistant_text})

    return {
        "trace_id": trace_id,
        "span_count": len(spans),
        "collected_at": base_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "spans": spans,
    }


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_traces(config_path: str, stats_only: bool = False) -> None:
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    seed = raw_config.get("seed", 42)
    num_conversations = raw_config.get("num_conversations", 200)
    shared_system_prompt_len = raw_config.get("shared_system_prompt_len", 6000)
    dynamic_system_prompt_len = parse_distribution(
        raw_config.get("dynamic_system_prompt_len", {"type": "fixed", "min": 2000, "max": 2000, "mean": 2000, "std_dev": 0})
    )
    turns_per_conversation = parse_distribution(
        raw_config.get("turns_per_conversation", {"type": "fixed", "min": 10, "max": 10, "mean": 10, "std_dev": 0})
    )
    input_tokens_per_turn = parse_distribution(
        raw_config.get("input_tokens_per_turn", {"type": "fixed", "min": 1400, "max": 1400, "mean": 1400, "std_dev": 0})
    )
    output_tokens_per_turn = parse_distribution(
        raw_config.get("output_tokens_per_turn", {"type": "fixed", "min": 526, "max": 526, "mean": 526, "std_dev": 0})
    )
    output_dir = Path(raw_config.get("output_dir", "./synthetic_traces"))
    model_name = raw_config.get("model_name", "gpt-4")

    rng = np.random.RandomState(seed)

    # Generate shared system prompt prefix (identical for all conversations)
    shared_prefix_text = generate_random_text(rng, shared_system_prompt_len)

    # Sample distributions for all conversations at once (deterministic)
    turn_counts = sample_distribution(turns_per_conversation, num_conversations, rng)
    dynamic_lens = sample_distribution(dynamic_system_prompt_len, num_conversations, rng)

    # Pre-sample all per-turn token counts
    total_turns = int(turn_counts.sum())
    all_input_tokens = sample_distribution(input_tokens_per_turn, total_turns, rng)
    all_output_tokens = sample_distribution(output_tokens_per_turn, total_turns, rng)

    # Build conversation blueprints
    blueprints: List[ConversationBlueprint] = []
    turn_cursor = 0

    for i in range(num_conversations):
        n_turns = int(turn_counts[i])
        dynamic_len = int(dynamic_lens[i])

        # Generate per-conversation dynamic suffix
        dynamic_suffix_text = generate_random_text(rng, dynamic_len)
        system_prompt = shared_prefix_text + "\n\n" + dynamic_suffix_text

        turns = []
        for t in range(n_turns):
            turns.append(TurnBlueprint(
                input_tokens=int(all_input_tokens[turn_cursor]),
                output_tokens=int(all_output_tokens[turn_cursor]),
            ))
            turn_cursor += 1

        blueprints.append(ConversationBlueprint(
            conversation_id=f"conv_{i:04d}",
            system_prompt=system_prompt,
            turns=turns,
        ))

    # Print stats
    turn_counts_list = [len(bp.turns) for bp in blueprints]
    print(f"Generated {num_conversations} conversation blueprints")
    print(f"  Seed: {seed}")
    print(f"  Shared system prompt: {shared_system_prompt_len} tokens")
    print(f"  Dynamic suffix: mean={np.mean(dynamic_lens):.0f}, min={np.min(dynamic_lens)}, max={np.max(dynamic_lens)}")
    print(f"  Turns/convo: mean={np.mean(turn_counts_list):.1f}, min={np.min(turn_counts_list)}, max={np.max(turn_counts_list)}, median={np.median(turn_counts_list):.0f}")
    print(f"  Total turns: {total_turns}")
    print(f"  Input tokens/turn: mean={np.mean(all_input_tokens):.0f}, min={np.min(all_input_tokens)}, max={np.max(all_input_tokens)}")
    print(f"  Output tokens/turn: mean={np.mean(all_output_tokens):.0f}, min={np.min(all_output_tokens)}, max={np.max(all_output_tokens)}")

    if stats_only:
        # Print histogram of turn counts
        print("\n  Turn count distribution:")
        hist, edges = np.histogram(turn_counts_list, bins=min(20, max(turn_counts_list) - min(turn_counts_list) + 1))
        for count, edge in zip(hist, edges):
            if count > 0:
                bar = "█" * min(count, 60)
                print(f"    {int(edge):>3}: {bar} ({count})")
        return

    # Generate trace files
    output_dir.mkdir(parents=True, exist_ok=True)
    for bp in blueprints:
        trace = generate_otel_trace(bp, rng, model_name=model_name)
        output_file = output_dir / f"{bp.conversation_id}.json"
        with open(output_file, "w") as f:
            json.dump(trace, f, indent=2)

    print(f"\n  Written {num_conversations} trace files to {output_dir}/")

    # Verify shared prefix consistency
    first_system = blueprints[0].system_prompt[:100]
    all_match = all(bp.system_prompt.startswith(shared_prefix_text) for bp in blueprints)
    print(f"  Shared prefix verified: {'YES' if all_match else 'NO'} (all {num_conversations} traces share identical {shared_system_prompt_len}-token prefix)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic OTel traces for agentic workload benchmarking")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--stats", action="store_true", help="Print distribution stats without generating files")
    args = parser.parse_args()
    generate_traces(args.config, stats_only=args.stats)
