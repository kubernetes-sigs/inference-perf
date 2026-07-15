# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType
from inference_perf.datagen.weka_trace_replay_datagen import (
    HashIdRandomGenerator,
    WekaTraceReplayDataGenerator,
    RoleSegment,
    ConversationReconstructor,
    longest_common_prefix,
    truncate_synth_buf_at_block,
    _IdleGapTimeWarp,
)


def test_longest_common_prefix() -> None:
    assert longest_common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert longest_common_prefix([1, 2], [1, 2, 3]) == 2
    assert longest_common_prefix([1], [2]) == 0
    assert longest_common_prefix([], [1]) == 0


def test_truncate_synth_buf_at_block() -> None:
    def dummy_decode(tokens: list[int]) -> str:
        return ",".join(str(t) for t in tokens)

    segments = [
        RoleSegment(role="system", block_start=0, block_count=2, tokens=[10, 11], content="10,11"),
        RoleSegment(role="user", block_start=2, block_count=3, tokens=[20, 21, 22], content="20,21,22"),
    ]

    # Truncate at block 3
    # block 3 lies inside the second segment (since system has blocks 0,1 and user has 2,3,4)
    # Target blocks = 3, so we keep 3 blocks total: system (2) + user (1)
    # The user segment should be truncated to block_count=1, tokens=[20]
    disturbed = truncate_synth_buf_at_block(segments, target_blocks=3, block_size=1, decode_tokens_to_text=dummy_decode)

    assert disturbed == 1  # Index of disturbed segment
    assert len(segments) == 2
    assert segments[0].block_count == 2
    assert segments[1].block_count == 1
    assert segments[1].tokens == [20]
    assert segments[1].content == "20"


def test_hash_id_random_generator() -> None:
    rng1 = HashIdRandomGenerator(base_seed=42)
    rng1.set_trace_id("trace_xyz")
    rng1.reseed_for_hash_id(1001)
    val1_a = rng1.randrange(1000)
    val1_b = rng1.randrange(1000)

    # Re-instantiate with same seed, trace ID and hash ID
    rng2 = HashIdRandomGenerator(base_seed=42)
    rng2.set_trace_id("trace_xyz")
    rng2.reseed_for_hash_id(1001)
    val2_a = rng2.randrange(1000)
    val2_b = rng2.randrange(1000)

    assert val1_a == val2_a
    assert val1_b == val2_b

    # A different seed, trace ID, or hash ID should produce a different result
    rng3 = HashIdRandomGenerator(base_seed=43)
    rng3.set_trace_id("trace_xyz")
    rng3.reseed_for_hash_id(1001)
    val3_a = rng3.randrange(1000)
    assert val3_a != val1_a


def test_idle_gap_time_warp() -> None:
    # Gap cap = 10s
    warp = _IdleGapTimeWarp(request_starts=[0.0, 5.0, 35.0, 45.0], cap_seconds=10.0)

    # 0.0 maps to 0.0
    assert warp.map(0.0) == 0.0
    # 5.0 maps to 5.0 (since 5.0 - 0.0 = 5.0 <= 10.0)
    assert warp.map(5.0) == 5.0

    # 35.0 has a gap from 5.0 of 30s. Cap is 10s. Excess is 20s.
    # So 35.0 shifts left by 20s to 15.0
    assert warp.map(35.0) == 15.0

    # 45.0 has a gap from 35.0 of 10s (equal to cap). So it shifts left by same 20s to 25.0
    assert warp.map(45.0) == 25.0


def test_conversation_reconstructor() -> None:
    def decode_block_tokens(hash_ids: list[int]) -> list[int]:
        # Simple dummy block decoder: block content is [hash_id]
        return [h for h in hash_ids]

    def sample_partial_tail_tokens(n: int, seed: str) -> list[int]:
        return [99] * n

    def decode_tokens_to_text(tokens: list[int]) -> str:
        return ",".join(str(t) for t in tokens)

    recon = ConversationReconstructor(
        block_size=1,
        decode_block_tokens=decode_block_tokens,
        sample_partial_tail_tokens=sample_partial_tail_tokens,
        decode_tokens_to_text=decode_tokens_to_text,
    )

    # Turn 0: input tokens = 3, hash_ids = [1, 2, 3]
    recon.init_turn_0(
        hash_ids=[1, 2, 3],
        in_tokens=3,
        tool_tokens=0,
        system_tokens=0,
        seed="seed0",
    )

    msgs = recon.snapshot_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "1,2,3"

    # Turn 1: curr_in_tokens = 5, hash_ids = [1, 2, 10, 11, 12], prev_out_tokens = 2
    # LCP of [1, 2, 3] and [1, 2, 10, 11, 12] is [1, 2] (length 2)
    # It will truncate the user segment at block 2, leaving system/user tokens [1, 2]
    # The previous assistant output size is 2. The remaining block capacity is curr_in_tokens - lcp = 3.
    # assistant blocks = min(ceil(2/1), 3) = 2. So assistant blocks = [10, 11].
    # User blocks = remaining = 1. User block = [12].
    recon.advance_turn(
        prev_hash_ids=[1, 2, 3],
        prev_in_tokens=3,
        prev_out_tokens=2,
        curr_hash_ids=[1, 2, 10, 11, 12],
        curr_in_tokens=5,
        seed="seed1",
    )

    msgs = recon.snapshot_messages()
    assert len(msgs) == 3
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "1,2"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "10,11"
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"] == "12"


def test_weka_trace_replay_generator_mock(tmp_path: Path) -> None:
    # Create a mock Weka Trace file
    trace_data = {
        "id": "mock_trace_123",
        "models": ["claude-opus-4-8"],
        "block_size": 2,
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": [
            {
                "t": 0.1,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 4,
                "out": 2,
                "hash_ids": [10, 20],
                "api_time": 0.5,
            },
            {
                "t": 1.2,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 8,
                "out": 4,
                "hash_ids": [10, 20, 30, 40],
                "api_time": 0.8,
            },
        ],
    }

    trace_file = tmp_path / "mock_trace.json"
    trace_file.write_text(json.dumps(trace_data))

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_tokenizer().encode = lambda x: [9] * len(x)
    mock_tokenizer.get_tokenizer().decode = lambda x: "".join(str(i) for i in x)

    # API config
    api_cfg = APIConfig(type=APIType.Chat, streaming=False)

    # Datagen Config
    data_cfg = DataConfig(type=DataGenType.WekaTraceReplay)
    # Setup weka_trace_replay config mock
    from inference_perf.config.datagen.replay import WekaTraceReplayConfig

    weka_cfg = WekaTraceReplayConfig(
        trace_files=[str(trace_file)],
        default_block_size=2,
    )
    data_cfg.weka_trace_replay = weka_cfg

    # Initialize generator
    gen = WekaTraceReplayDataGenerator(
        api_config=api_cfg,
        config=data_cfg,
        tokenizer=mock_tokenizer,
        num_workers=1,
    )

    assert len(gen.sessions) == 1
    session = gen._get_session(0)
    assert session.source_id == "mock_trace_123"

    # Graph should have 2 events representing the 2 parent turns
    assert len(session.graph.events) == 2
    events = sorted(session.graph.events.values(), key=lambda e: e.t_start_ms)

    assert events[0].call.total_input_tokens == 4
    assert events[0].call.expected_output_tokens == 2

    assert events[1].call.total_input_tokens == 8
    assert events[1].call.expected_output_tokens == 4

    # Predecessor edge check
    assert events[1].predecessor_event_ids == [events[0].event_id]


def test_weka_trace_replay_generator_mock_no_warp(tmp_path: Path) -> None:
    # Create a mock Weka Trace file with a huge gap (100 seconds)
    trace_data = {
        "id": "mock_trace_no_warp",
        "models": ["claude-opus-4-8"],
        "block_size": 2,
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": [
            {
                "t": 0.1,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 4,
                "out": 2,
                "hash_ids": [10, 20],
                "api_time": 0.5,
            },
            {
                "t": 100.1,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 8,
                "out": 4,
                "hash_ids": [10, 20, 30, 40],
                "api_time": 0.8,
            },
        ],
    }

    trace_file = tmp_path / "mock_trace_no_warp.json"
    trace_file.write_text(json.dumps(trace_data))

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_tokenizer().encode = lambda x: [9] * len(x)
    mock_tokenizer.get_tokenizer().decode = lambda x: "".join(str(i) for i in x)

    # API config
    api_cfg = APIConfig(type=APIType.Chat, streaming=False)

    # Datagen Config with trace_idle_gap_cap_seconds = 0 (disabled)
    data_cfg = DataConfig(type=DataGenType.WekaTraceReplay)
    from inference_perf.config.datagen.replay import WekaTraceReplayConfig

    weka_cfg = WekaTraceReplayConfig(
        trace_files=[str(trace_file)],
        default_block_size=2,
        trace_idle_gap_cap_seconds=0,
    )
    data_cfg.weka_trace_replay = weka_cfg

    # Initialize generator
    gen = WekaTraceReplayDataGenerator(
        api_config=api_cfg,
        config=data_cfg,
        tokenizer=mock_tokenizer,
        num_workers=1,
    )

    assert len(gen.sessions) == 1
    session = gen._get_session(0)
    events = sorted(session.graph.events.values(), key=lambda e: e.t_start_ms)

    # In raw seconds: t0 = 0.1s (100ms), t1 = 100.1s (100100ms)
    # The gap is exactly 100000ms. Since gap warping is disabled (<= 0),
    # the start times should reflect original timing.
    assert events[0].t_start_ms == 100
    assert events[1].t_start_ms == 100100
    assert events[1].wait_ms == 100100 - events[0].t_end_ms


def test_weka_trace_replay_parallel_matches_serial(tmp_path: Path) -> None:
    """Parallel session building (datagen_workers > 1) must produce output
    identical to the serial path: same sessions, same order, same graphs."""
    from inference_perf.config.datagen.replay import WekaTraceReplayConfig
    from inference_perf.datagen.otel_trace_to_replay_graph import graph_to_dict

    trace_files = []
    for i in range(4):
        trace_data = {
            "id": f"mock_trace_{i}",
            "models": ["claude-opus-4-8"],
            "block_size": 2,
            "tool_tokens": 0,
            "system_tokens": 0,
            "requests": [
                {
                    "t": 0.1,
                    "type": "n",
                    "model": "claude-opus-4-8",
                    "in": 4,
                    "out": 2,
                    "hash_ids": [10 + i, 20 + i],
                    "api_time": 0.5,
                },
                {
                    "t": 1.2,
                    "type": "n",
                    "model": "claude-opus-4-8",
                    "in": 8,
                    "out": 4,
                    "hash_ids": [10 + i, 20 + i, 30 + i, 40 + i],
                    "api_time": 0.8,
                },
            ],
        }
        trace_file = tmp_path / f"mock_trace_{i}.json"
        trace_file.write_text(json.dumps(trace_data))
        trace_files.append(str(trace_file))

    def build(datagen_workers: int) -> WekaTraceReplayDataGenerator:
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_tokenizer().encode = lambda x: [9] * len(x)
        mock_tokenizer.get_tokenizer().decode = lambda x: "".join(str(i) for i in x)

        data_cfg = DataConfig(type=DataGenType.WekaTraceReplay)
        data_cfg.weka_trace_replay = WekaTraceReplayConfig(
            trace_files=trace_files,
            default_block_size=2,
            datagen_workers=datagen_workers,
        )
        return WekaTraceReplayDataGenerator(
            api_config=APIConfig(type=APIType.Chat, streaming=False),
            config=data_cfg,
            tokenizer=mock_tokenizer,
            num_workers=1,
        )

    serial = build(1)
    parallel = build(2)

    # The Weka datagen uses the eager path, so no session slot is ever None.
    serial_sessions = [s for s in serial.sessions if s is not None]
    parallel_sessions = [s for s in parallel.sessions if s is not None]
    assert len(serial.sessions) == len(serial_sessions) == 4
    assert [s.session_id for s in serial_sessions] == [s.session_id for s in parallel_sessions]
    for s_serial, s_parallel in zip(serial_sessions, parallel_sessions, strict=True):
        assert s_serial.source_id == s_parallel.source_id
        assert graph_to_dict(s_serial.graph) == graph_to_dict(s_parallel.graph)


def _write_mock_trace(tmp_path: Path, trace_id: str, hash_base: int) -> str:
    trace_data = {
        "id": trace_id,
        "models": ["m"],
        "block_size": 2,
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": [
            {"t": 0.1, "type": "n", "model": "m", "in": 4, "out": 2, "hash_ids": [hash_base, hash_base + 1], "api_time": 0.5},
        ],
    }
    trace_file = tmp_path / f"{trace_id}.json"
    trace_file.write_text(json.dumps(trace_data))
    return str(trace_file)


def _mock_tokenizer() -> MagicMock:
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_tokenizer().encode = lambda x: [9] * len(x)
    mock_tokenizer.get_tokenizer().decode = lambda x: "".join(str(i) for i in x)
    return mock_tokenizer


def _build_generator(trace_files: list[str], datagen_workers: int, skip_invalid_files: bool) -> WekaTraceReplayDataGenerator:
    from inference_perf.config.datagen.replay import WekaTraceReplayConfig

    data_cfg = DataConfig(type=DataGenType.WekaTraceReplay)
    data_cfg.weka_trace_replay = WekaTraceReplayConfig(
        trace_files=trace_files,
        default_block_size=2,
        datagen_workers=datagen_workers,
        skip_invalid_files=skip_invalid_files,
    )
    return WekaTraceReplayDataGenerator(
        api_config=APIConfig(type=APIType.Chat, streaming=False),
        config=data_cfg,
        tokenizer=_mock_tokenizer(),
        num_workers=1,
    )


def test_weka_skip_invalid_files_through_parallel_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A trace that fails session building must be skipped (skip_invalid_files=True)
    or fail the run (False), with the error marshaled across pool workers."""
    from inference_perf.datagen.otel_trace_to_replay_graph import build_graph as original_build_graph

    good_files = [_write_mock_trace(tmp_path, f"good_{i}", 10 * (i + 1)) for i in range(3)]

    # File-level parsing errors are caught in _load_weka_traces, so to exercise
    # the session-build error path we make reconstruction itself raise for one
    # trace ID via a patched build_graph.
    def failing_build_graph(calls: object, source_file: str = "", **kwargs: object) -> object:
        if "good_1" in source_file:
            raise ValueError("synthetic reconstruction failure")
        return original_build_graph(calls, source_file=source_file, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("inference_perf.datagen.weka_trace_replay_datagen.build_graph", failing_build_graph)

    # skip_invalid_files=True: the failing trace is dropped, the rest survive.
    # datagen_workers=2 routes errors through the process-pool marshaling path.
    gen = _build_generator(good_files, datagen_workers=2, skip_invalid_files=True)
    assert len(gen.sessions) == 2
    assert sorted(s.source_id for s in gen.sessions if s is not None) == ["good_0", "good_2"]

    # skip_invalid_files=False: the run must fail and name the trace.
    with pytest.raises(RuntimeError, match="good_1"):
        _build_generator(good_files, datagen_workers=2, skip_invalid_files=False)


def test_weka_resolve_datagen_workers(tmp_path: Path) -> None:
    """Explicit datagen_workers is honored and capped at the trace count."""
    trace_files = [_write_mock_trace(tmp_path, f"t_{i}", 10 * (i + 1)) for i in range(2)]
    gen = _build_generator(trace_files, datagen_workers=1, skip_invalid_files=False)

    # Explicit value passes through; both are capped by the number of traces.
    gen.weka_config.datagen_workers = 5
    assert gen._resolve_datagen_workers(num_traces=2) == 2
    assert gen._resolve_datagen_workers(num_traces=10) == 5

    gen.weka_config.datagen_workers = 1
    assert gen._resolve_datagen_workers(num_traces=10) == 1

    # Auto (None) resolves to at least 1 and never exceeds the trace count.
    gen.weka_config.datagen_workers = None
    assert 1 <= gen._resolve_datagen_workers(num_traces=3) <= 3
