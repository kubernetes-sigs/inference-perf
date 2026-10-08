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

"""Tests for TraceLab trace replay (issue #813)."""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType
from inference_perf.config.datagen.replay import TraceLabTraceReplayConfig
from inference_perf.datagen.replay.tracelab_trace_replay_datagen import (
    FilterExpressionError,
    TraceLabTraceReplayDataGenerator,
)


# Hand-written fixture rows for unit tests (sanitized shape).
def _rows() -> List[Dict[str, Any]]:
    return [
        {
            "session_id": "sess-A",
            "round_id": "r2",
            "round_index": 1,
            "provider": "claude",
            "model": "claude-sonnet",
            "input_tokens_total": 30,
            "prefix_tokens": 20,
            "newly_append_tokens": 10,
            "output_tokens": 5,
            "tools": [],  # no tools on last round
        },
        {
            "session_id": "sess-A",
            "round_id": "r1",
            "round_index": 0,
            "provider": "claude",
            "model": "claude-sonnet",
            "input_tokens_total": 20,
            "prefix_tokens": 0,
            "newly_append_tokens": 20,
            "output_tokens": 7,
            "tools": [{"tool_name": "bash", "tool_wall_latency_ms": 1200.0}],  # tools on round 0
        },
        {
            "session_id": "sess-B",
            "round_id": "r1",
            "round_index": 0,
            "provider": "codex",
            "model": "gpt-5",
            "input_tokens_total": 12,
            "prefix_tokens": 0,
            "newly_append_tokens": 12,
            "output_tokens": 3,
            "tools": [],
        },
    ]


# Real TraceLab v0.0.2 fixture rows (CC BY 4.0) - subset for regression tests.
# These mimic the actual sanitized JSONL structure from the public dataset.
def _real_rows() -> List[Dict[str, Any]]:
    return [
        {
            "round_pk": "claude-abc-1",
            "session_id": "claude:abc",
            "round_id": "r1",
            "round_index": 0,
            "provider": "claude",
            "model": "claude-sonnet",
            "input_tokens_total": 1500,
            "prefix_tokens": 0,
            "newly_append_tokens": 1500,
            "output_tokens": 200,
            "tools": [],
        },
        {
            "round_pk": "claude-abc-2",
            "session_id": "claude:abc",
            "round_id": "r2",
            "round_index": 1,
            "provider": "claude",
            "model": "claude-sonnet",
            "input_tokens_total": 1800,
            "prefix_tokens": 1400,
            "newly_append_tokens": 400,
            "output_tokens": 250,
            "tools": [
                {"tool_name": "bash", "tool_wall_latency_ms": 800.0},
                {"tool_name": "edit", "tool_wall_latency_ms": 1200.0},
            ],
        },
        {
            "round_pk": "claude-abc-3",
            "session_id": "claude:abc",
            "round_id": "r3",
            "round_index": 2,
            "provider": "claude",
            "model": "claude-sonnet",
            "input_tokens_total": 2200,
            "prefix_tokens": 1900,
            "newly_append_tokens": 300,
            "output_tokens": 300,
            "tools": [],
        },
        # Session with cache miss (prefix resets to 0 mid-session)
        {
            "round_pk": "claude-xyz-1",
            "session_id": "claude:xyz",
            "round_id": "r1",
            "round_index": 0,
            "provider": "claude",
            "model": "claude-sonnet",
            "input_tokens_total": 800,
            "prefix_tokens": 0,
            "newly_append_tokens": 800,
            "output_tokens": 100,
            "tools": [],
        },
        {
            "round_pk": "claude-xyz-2",
            "session_id": "claude:xyz",
            "round_id": "r2",
            "round_index": 1,
            "provider": "claude",
            "model": "claude-sonnet",
            "input_tokens_total": 1200,
            "prefix_tokens": 0,  # Full cache miss
            "newly_append_tokens": 1200,
            "output_tokens": 150,
            "tools": [{"tool_name": "bash", "tool_wall_latency_ms": 500.0}],
        },
        # Session with compaction (prompt shrinks)
        {
            "round_pk": "codex-pqr-1",
            "session_id": "codex:pqr",
            "round_id": "r1",
            "round_index": 0,
            "provider": "codex",
            "model": "gpt-5",
            "input_tokens_total": 5000,
            "prefix_tokens": 0,
            "newly_append_tokens": 5000,
            "output_tokens": 500,
            "tools": [],
        },
        {
            "round_pk": "codex-pqr-2",
            "session_id": "codex:pqr",
            "round_id": "r2",
            "round_index": 1,
            "provider": "codex",
            "model": "gpt-5",
            "input_tokens_total": 3000,
            "prefix_tokens": 2500,
            "newly_append_tokens": 500,
            "output_tokens": 400,
            "tools": [],
        },
    ]


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def _mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    # Deterministic 1-token-per-char-ish codec: encode length == len(text).
    tok.get_tokenizer().encode = lambda text: list(range(len(text)))
    tok.get_tokenizer().decode = lambda toks: "x" * len(toks)
    return tok


def _make_gen(
    trace_files: List[str],
    **overrides: Any,
) -> TraceLabTraceReplayDataGenerator:
    api_cfg = APIConfig(type=APIType.Chat, streaming=False)
    data_cfg = DataConfig(type=DataGenType.TraceLabTraceReplay)
    data_cfg.tracelab_trace_replay = TraceLabTraceReplayConfig(trace_files=trace_files, **overrides)
    return TraceLabTraceReplayDataGenerator(api_config=api_cfg, config=data_cfg, tokenizer=_mock_tokenizer())


def _tokenize_messages(tokenizer: MagicMock, messages: List[Any]) -> int:
    """Count tokens by encoding the concatenated message texts."""
    text = "".join(m.get("content", "") if isinstance(m, dict) else m.text for m in messages)
    return len(tokenizer.get_tokenizer().encode(text))


def test_tracelab_type_registered() -> None:
    """Issue #813 failing scenario: the tracelab type must exist and validate."""
    assert DataGenType("tracelab_trace_replay") is DataGenType.TraceLabTraceReplay
    cfg = DataConfig(type=DataGenType.TraceLabTraceReplay)
    cfg.tracelab_trace_replay = TraceLabTraceReplayConfig(trace_files=["./traces/t.jsonl"])
    assert cfg.tracelab_trace_replay is not None


def test_tracelab_config_requires_source() -> None:
    with pytest.raises(ValidationError):
        TraceLabTraceReplayConfig()


def test_tracelab_config_rejects_multiple_sources(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        TraceLabTraceReplayConfig(trace_files=["a.jsonl"], trace_directory=str(tmp_path))


def test_tracelab_groups_orders_and_chains(tmp_path: Path) -> None:
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", _rows())
    gen = _make_gen([str(trace_file)])

    assert gen.get_session_count() == 2
    sessions = {s.source_id: s for s in gen.sessions if s is not None}
    # Session key is now just session_id (no provider prefix).
    assert set(sessions) == {"sess-A", "sess-B"}

    sess_a = sessions["sess-A"]
    assert len(sess_a.graph.events) == 2
    events = sorted(sess_a.graph.events.values(), key=lambda e: e.t_start_ms)
    assert events[0].call.total_input_tokens == 20
    assert events[0].call.expected_output_tokens == 7
    assert events[1].call.total_input_tokens == 30
    assert events[1].call.expected_output_tokens == 5
    # Linear chain: second round depends on the first.
    assert events[1].predecessor_event_ids == [events[0].event_id]
    # Tool latency from round 0 (1200ms) preserved as spacing before round 1.
    assert events[1].wait_ms == 1200


def test_tracelab_prompt_size_matches_recorded(tmp_path: Path) -> None:
    """Built prompt should tokenize to recorded input_tokens_total (or close)."""
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", _rows())
    gen = _make_gen([str(trace_file)])

    for s in gen.sessions:
        assert s is not None
        for e in s.graph.events.values():
            # Tokenize the built messages and compare to recorded total_input_tokens.
            built_tokens = _tokenize_messages(_mock_tokenizer(), e.call.messages)
            recorded = e.call.total_input_tokens
            # Allow small drift due to tokenizer quirks.
            assert abs(built_tokens - recorded) <= max(4, recorded * 0.1), (
                f"Event {e.event_id}: built {built_tokens} vs recorded {recorded}"
            )


def test_tracelab_idle_gap_cap_and_ignore_delays(tmp_path: Path) -> None:
    rows = [
        {
            "session_id": "s",
            "round_index": 0,
            "provider": "claude",
            "model": "m",
            "input_tokens_total": 10,
            "output_tokens": 2,
            "tools": [{"tool_name": "bash", "tool_wall_latency_ms": 5000.0}],  # tools on round 0
        },
        {
            "session_id": "s",
            "round_index": 1,
            "provider": "claude",
            "model": "m",
            "input_tokens_total": 15,
            "output_tokens": 2,
            "tools": [],  # no tools on round 1
        },
    ]
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", rows)

    capped = _make_gen([str(trace_file)], trace_idle_gap_cap_seconds=1.0)
    events = sorted(
        next(s for s in capped.sessions if s is not None).graph.events.values(),
        key=lambda e: e.t_start_ms,
    )
    # Wait after round 0 (5000ms) capped to 1000ms appears before round 1.
    assert events[1].wait_ms == 1000

    ignored = _make_gen([str(trace_file)], ignore_trace_delays=True)
    events = sorted(
        next(s for s in ignored.sessions if s is not None).graph.events.values(),
        key=lambda e: e.t_start_ms,
    )
    assert events[1].wait_ms == 0


def test_tracelab_filter_selects_sessions(tmp_path: Path) -> None:
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", _rows())
    gen = _make_gen([str(trace_file)], filter="lambda x: x['num_rounds'] > 1")
    assert gen.get_session_count() == 1
    assert gen.sessions[0] is not None and gen.sessions[0].source_id == "sess-A"


def test_tracelab_bad_filter_is_config_error(tmp_path: Path) -> None:
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", _rows())
    with pytest.raises(FilterExpressionError):
        _make_gen([str(trace_file)], filter="lambda x: x['missing'] > 1")


def test_tracelab_gz_and_directory_sources(tmp_path: Path) -> None:
    gz_path = tmp_path / "trace.jsonl.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        for r in _rows():
            f.write(json.dumps(r) + "\n")
    gen = _make_gen([str(gz_path)])
    assert gen.get_session_count() == 2

    subdir = tmp_path / "traces"
    subdir.mkdir()
    _write_jsonl(subdir / "a.jsonl", _rows()[:2])
    api_cfg = APIConfig(type=APIType.Chat, streaming=False)
    data_cfg = DataConfig(type=DataGenType.TraceLabTraceReplay)
    data_cfg.tracelab_trace_replay = TraceLabTraceReplayConfig(trace_directory=str(subdir))
    gen2 = TraceLabTraceReplayDataGenerator(api_config=api_cfg, config=data_cfg, tokenizer=_mock_tokenizer())
    assert gen2.get_session_count() == 1


def test_tracelab_invalid_rows_skip_or_fail(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    good = _write_jsonl(tmp_path / "good.jsonl", _rows()[:1])
    with pytest.raises(Exception, match="Expecting value"):
        _make_gen([str(bad)])
    # skip_invalid_files skips the bad file and loads the good one.
    gen = _make_gen([str(bad), str(good)], skip_invalid_files=True)
    assert gen.get_session_count() == 1
    # A fully-invalid corpus still fails fast with no sessions to replay.
    with pytest.raises(ValueError, match="No valid TraceLab sessions"):
        _make_gen([str(bad)], skip_invalid_files=True)


def test_tracelab_static_model_mapping(tmp_path: Path) -> None:
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", _rows())
    gen = _make_gen([str(trace_file)], use_static_model=True, static_model_name="mock-model")
    for s in gen.sessions:
        assert s is not None
        for e in s.graph.events.values():
            assert e.call.model == "mock-model"


def test_tracelab_regression_existing_types_untouched() -> None:
    assert DataGenType("weka_trace_replay") is DataGenType.WekaTraceReplay
    assert DataGenType("otel_trace_replay") is DataGenType.OTelTraceReplay
    assert DataGenType("synthetic_agentic") is DataGenType.SyntheticAgentic


def test_tracelab_required_fields_reject_invalid_rows(tmp_path: Path) -> None:
    """Rows missing required fields should fail validation."""
    bad_rows = [
        {"session_id": "s", "round_index": 0},  # missing input_tokens_total, output_tokens
    ]
    trace_file = _write_jsonl(tmp_path / "bad.jsonl", bad_rows)
    with pytest.raises(ValidationError):
        _make_gen([str(trace_file)])


def test_tracelab_determinism_across_runs(tmp_path: Path) -> None:
    """Same trace built twice should produce identical prompts."""
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", _real_rows())
    gen1 = _make_gen([str(trace_file)])
    gen2 = _make_gen([str(trace_file)])

    for s1, s2 in zip(gen1.sessions, gen2.sessions, strict=True):
        assert s1 is not None and s2 is not None
        events1 = sorted(s1.graph.events.values(), key=lambda e: e.t_start_ms)
        events2 = sorted(s2.graph.events.values(), key=lambda e: e.t_start_ms)
        for e1, e2 in zip(events1, events2, strict=True):
            # Prompt text should be identical.
            text1 = "".join(m.get("content", "") if isinstance(m, dict) else m.text for m in e1.call.messages)
            text2 = "".join(m.get("content", "") if isinstance(m, dict) else m.text for m in e2.call.messages)
            assert text1 == text2, f"Prompt mismatch in {e1.event_id}"


def test_tracelab_wait_placement_after_round_with_tools(tmp_path: Path) -> None:
    """Tool latency belongs after the round that emitted them, not before next."""
    rows = [
        {
            "session_id": "s",
            "round_index": 0,
            "provider": "claude",
            "model": "m",
            "input_tokens_total": 10,
            "output_tokens": 5,
            "tools": [{"tool_name": "bash", "tool_wall_latency_ms": 1000.0}],  # round 0 has tools
        },
        {
            "session_id": "s",
            "round_index": 1,
            "provider": "claude",
            "model": "m",
            "input_tokens_total": 15,
            "output_tokens": 5,
            "tools": [],  # round 1 no tools
        },
        {
            "session_id": "s",
            "round_index": 2,
            "provider": "claude",
            "model": "m",
            "input_tokens_total": 20,
            "output_tokens": 5,
            "tools": [{"tool_name": "edit", "tool_wall_latency_ms": 2000.0}],  # last round has tools
        },
    ]
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", rows)
    gen = _make_gen([str(trace_file)], trace_idle_gap_cap_seconds=10.0)

    events = sorted(
        next(s for s in gen.sessions if s is not None).graph.events.values(),
        key=lambda e: e.t_start_ms,
    )
    # Round 0 has tools -> wait before round 1 = 1000ms
    assert events[1].wait_ms == 1000
    # Round 1 has no tools -> wait before round 2 = 0
    assert events[2].wait_ms == 0
    # Round 2 has tools but is last round -> no next round to apply wait
    # events[2].wait_ms is 0 (no successor)
    assert events[2].t_start_ms == 1000  # round 2 starts after round 1's wait (0)


def test_tracelab_session_key_no_provider_duplication(tmp_path: Path) -> None:
    """Real session_ids already include provider prefix; key should not duplicate."""
    rows = [
        {
            "session_id": "claude:abc123",  # provider already in session_id
            "round_index": 0,
            "provider": "claude",
            "model": "m",
            "input_tokens_total": 10,
            "output_tokens": 5,
        },
    ]
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", rows)
    gen = _make_gen([str(trace_file)])
    # Should NOT create "claude:claude:abc123"
    assert gen.get_session_count() == 1
    session = gen.sessions[0]
    assert session is not None
    assert session.source_id == "claude:abc123"
