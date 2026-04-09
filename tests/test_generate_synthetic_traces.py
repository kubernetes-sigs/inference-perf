"""Tests for the synthetic OTel trace generator."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

# Import from tools/ — add to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from generate_synthetic_traces import (
    ConversationBlueprint,
    DistributionConfig,
    TurnBlueprint,
    generate_otel_trace,
    generate_random_text,
    generate_traces,
    parse_distribution,
    sample_distribution,
)


class TestDistributionConfig:
    def test_parse_scalar(self) -> None:
        config = parse_distribution(42)
        assert config.type == "fixed"
        assert config.mean == 42.0

    def test_parse_dict(self) -> None:
        config = parse_distribution({"type": "lognormal", "min": 3, "max": 50, "mean": 4, "std_dev": 3})
        assert config.type == "lognormal"
        assert config.min == 3
        assert config.max == 50

    def test_parse_invalid(self) -> None:
        with pytest.raises(ValueError):
            parse_distribution("invalid")


class TestSampleDistribution:
    def test_fixed(self) -> None:
        config = DistributionConfig(type="fixed", min=10, max=10, mean=10, std_dev=0)
        rng = np.random.RandomState(42)
        values = sample_distribution(config, 100, rng)
        assert all(v == 10 for v in values)

    def test_uniform_within_bounds(self) -> None:
        config = DistributionConfig(type="uniform", min=5, max=15, mean=10, std_dev=0)
        rng = np.random.RandomState(42)
        values = sample_distribution(config, 1000, rng)
        assert all(5 <= v <= 15 for v in values)

    def test_normal_within_bounds(self) -> None:
        config = DistributionConfig(type="normal", min=1, max=100, mean=50, std_dev=10)
        rng = np.random.RandomState(42)
        values = sample_distribution(config, 1000, rng)
        assert all(1 <= v <= 100 for v in values)

    def test_lognormal_within_bounds(self) -> None:
        config = DistributionConfig(type="lognormal", min=3, max=50, mean=4, std_dev=3)
        rng = np.random.RandomState(42)
        values = sample_distribution(config, 1000, rng)
        assert all(3 <= v <= 50 for v in values)

    def test_lognormal_right_skewed(self) -> None:
        config = DistributionConfig(type="lognormal", min=3, max=50, mean=4, std_dev=3)
        rng = np.random.RandomState(42)
        values = sample_distribution(config, 10000, rng)
        median = float(np.median(values))
        mean = float(np.mean(values))
        # Right-skewed: mean > median
        assert mean >= median, f"Expected right-skew: mean={mean} should be >= median={median}"

    def test_deterministic(self) -> None:
        config = DistributionConfig(type="normal", min=1, max=100, mean=50, std_dev=10)
        values1 = sample_distribution(config, 100, np.random.RandomState(42))
        values2 = sample_distribution(config, 100, np.random.RandomState(42))
        assert list(values1) == list(values2)

    def test_unknown_type_raises(self) -> None:
        config = DistributionConfig(type="unknown", min=1, max=10, mean=5, std_dev=1)
        with pytest.raises(ValueError, match="Unknown distribution type"):
            sample_distribution(config, 10, np.random.RandomState(42))


class TestGenerateRandomText:
    def test_nonempty(self) -> None:
        rng = np.random.RandomState(42)
        text = generate_random_text(rng, 100)
        assert len(text) > 0

    def test_zero_tokens(self) -> None:
        rng = np.random.RandomState(42)
        text = generate_random_text(rng, 0)
        assert text == ""

    def test_deterministic(self) -> None:
        text1 = generate_random_text(np.random.RandomState(42), 50)
        text2 = generate_random_text(np.random.RandomState(42), 50)
        assert text1 == text2


class TestGenerateOtelTrace:
    def test_basic_structure(self) -> None:
        bp = ConversationBlueprint(
            conversation_id="test_001",
            system_prompt="System prompt text",
            turns=[
                TurnBlueprint(input_tokens=100, output_tokens=50),
                TurnBlueprint(input_tokens=200, output_tokens=75),
            ],
        )
        rng = np.random.RandomState(42)
        trace = generate_otel_trace(bp, rng)

        assert trace["trace_id"] == "test_001"
        assert trace["span_count"] == 2
        assert len(trace["spans"]) == 2

    def test_span_fields(self) -> None:
        bp = ConversationBlueprint(
            conversation_id="test_002",
            system_prompt="SP",
            turns=[TurnBlueprint(input_tokens=50, output_tokens=25)],
        )
        rng = np.random.RandomState(42)
        trace = generate_otel_trace(bp, rng)
        span = trace["spans"][0]

        assert span["trace_id"] == "test_002"
        assert span["span_id"] == "turn_0"
        assert "gen_ai.input.messages" in span["attributes"]
        assert "gen_ai.output.text" in span["attributes"]
        assert "gen_ai.usage.input_tokens" in span["attributes"]
        assert "gen_ai.usage.output_tokens" in span["attributes"]
        assert "gen_ai.request.max_tokens" in span["attributes"]

    def test_messages_accumulate(self) -> None:
        bp = ConversationBlueprint(
            conversation_id="test_003",
            system_prompt="SP",
            turns=[
                TurnBlueprint(input_tokens=50, output_tokens=25),
                TurnBlueprint(input_tokens=50, output_tokens=25),
                TurnBlueprint(input_tokens=50, output_tokens=25),
            ],
        )
        rng = np.random.RandomState(42)
        trace = generate_otel_trace(bp, rng)

        # Turn 0: system + user (2 messages)
        msgs0 = json.loads(trace["spans"][0]["attributes"]["gen_ai.input.messages"])
        assert len(msgs0) == 2

        # Turn 1: system + user + assistant + user (4 messages)
        msgs1 = json.loads(trace["spans"][1]["attributes"]["gen_ai.input.messages"])
        assert len(msgs1) == 4

        # Turn 2: 6 messages
        msgs2 = json.loads(trace["spans"][2]["attributes"]["gen_ai.input.messages"])
        assert len(msgs2) == 6

    def test_shared_system_prompt(self) -> None:
        shared_prompt = "This is the shared system prompt"
        bp1 = ConversationBlueprint(
            conversation_id="conv_a",
            system_prompt=shared_prompt + " suffix_a",
            turns=[TurnBlueprint(input_tokens=50, output_tokens=25)],
        )
        bp2 = ConversationBlueprint(
            conversation_id="conv_b",
            system_prompt=shared_prompt + " suffix_b",
            turns=[TurnBlueprint(input_tokens=50, output_tokens=25)],
        )
        rng = np.random.RandomState(42)
        trace1 = generate_otel_trace(bp1, rng)
        trace2 = generate_otel_trace(bp2, np.random.RandomState(43))

        msgs1 = json.loads(trace1["spans"][0]["attributes"]["gen_ai.input.messages"])
        msgs2 = json.loads(trace2["spans"][0]["attributes"]["gen_ai.input.messages"])

        # Both start with the shared prefix
        assert msgs1[0]["content"].startswith(shared_prompt)
        assert msgs2[0]["content"].startswith(shared_prompt)


class TestEndToEnd:
    def test_generate_traces_creates_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "seed": 42,
                "num_conversations": 5,
                "shared_system_prompt_len": 100,
                "dynamic_system_prompt_len": {"type": "fixed", "min": 50, "max": 50, "mean": 50, "std_dev": 0},
                "turns_per_conversation": {"type": "fixed", "min": 3, "max": 3, "mean": 3, "std_dev": 0},
                "input_tokens_per_turn": {"type": "fixed", "min": 50, "max": 50, "mean": 50, "std_dev": 0},
                "output_tokens_per_turn": {"type": "fixed", "min": 25, "max": 25, "mean": 25, "std_dev": 0},
                "output_dir": os.path.join(tmpdir, "traces"),
            }
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            generate_traces(config_path)

            trace_dir = Path(config["output_dir"])
            assert trace_dir.exists()
            files = list(trace_dir.glob("*.json"))
            assert len(files) == 5

            # Validate each trace
            for trace_file in files:
                with open(trace_file) as f:
                    trace = json.load(f)
                assert "trace_id" in trace
                assert "spans" in trace
                assert len(trace["spans"]) == 3  # fixed 3 turns

    def test_deterministic_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "seed": 123,
                "num_conversations": 3,
                "shared_system_prompt_len": 50,
                "dynamic_system_prompt_len": 25,
                "turns_per_conversation": 2,
                "input_tokens_per_turn": 30,
                "output_tokens_per_turn": 15,
                "output_dir": os.path.join(tmpdir, "run1"),
            }
            config_path = os.path.join(tmpdir, "config.yaml")

            # Run 1
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            generate_traces(config_path)

            # Run 2
            config["output_dir"] = os.path.join(tmpdir, "run2")
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            generate_traces(config_path)

            # Compare
            for i in range(3):
                with open(os.path.join(tmpdir, "run1", f"conv_{i:04d}.json")) as f:
                    t1 = json.load(f)
                with open(os.path.join(tmpdir, "run2", f"conv_{i:04d}.json")) as f:
                    t2 = json.load(f)
                assert t1 == t2, f"Trace conv_{i:04d} differs between runs"

    def test_shared_prefix_identical_across_conversations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "seed": 42,
                "num_conversations": 10,
                "shared_system_prompt_len": 200,
                "dynamic_system_prompt_len": {"type": "fixed", "min": 50, "max": 50, "mean": 50, "std_dev": 0},
                "turns_per_conversation": 2,
                "input_tokens_per_turn": 30,
                "output_tokens_per_turn": 15,
                "output_dir": os.path.join(tmpdir, "traces"),
            }
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            generate_traces(config_path)

            # Extract system prompts from all traces
            system_prompts = []
            for trace_file in sorted(Path(os.path.join(tmpdir, "traces")).glob("*.json")):
                with open(trace_file) as f:
                    trace = json.load(f)
                msgs = json.loads(trace["spans"][0]["attributes"]["gen_ai.input.messages"])
                system_prompts.append(msgs[0]["content"])

            # All should share the same prefix (the shared_system_prompt part)
            # The prefix length varies by tokenizer approximation, but the first
            # ~80% of the shortest prompt should be identical across all
            min_len = min(len(sp) for sp in system_prompts)
            prefix_check_len = int(min_len * 0.5)  # conservative: check first 50%
            reference = system_prompts[0][:prefix_check_len]
            for i, sp in enumerate(system_prompts):
                assert sp[:prefix_check_len] == reference, (
                    f"Conversation {i} does not share the expected system prompt prefix"
                )
