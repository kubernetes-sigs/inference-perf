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
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType
from inference_perf.config.datagen.replay import TraceLabTraceReplayConfig
from inference_perf.datagen.replay.tracelab_trace_replay_datagen import (
    FilterExpressionError,
    TraceLabTraceReplayDataGenerator,
)


def _rows() -> list[dict]:
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
            "tools": [{"tool_name": "bash", "tool_wall_latency_ms": 1200.0}],
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
            "tools": [],
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


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def _mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    # Deterministic 1-token-per-char-ish codec: encode length == len(text).
    tok.get_tokenizer().encode = lambda text: list(range(len(text)))
    tok.get_tokenizer().decode = lambda toks: "x" * len(toks)
    return tok


def _make_gen(trace_files: list[str], **overrides) -> TraceLabTraceReplayDataGenerator:
    api_cfg = APIConfig(type=APIType.Chat, streaming=False)
    data_cfg = DataConfig(type=DataGenType.TraceLabTraceReplay)
    data_cfg.tracelab_trace_replay = TraceLabTraceReplayConfig(trace_files=trace_files, **overrides)
    return TraceLabTraceReplayDataGenerator(api_config=api_cfg, config=data_cfg, tokenizer=_mock_tokenizer())


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
    assert set(sessions) == {"claude:sess-A", "codex:sess-B"}

    sess_a = sessions["claude:sess-A"]
    assert len(sess_a.graph.events) == 2
    events = sorted(sess_a.graph.events.values(), key=lambda e: e.t_start_ms)
    assert events[0].call.total_input_tokens == 20
    assert events[0].call.expected_output_tokens == 7
    assert events[1].call.total_input_tokens == 30
    assert events[1].call.expected_output_tokens == 5
    # Linear chain: second round depends on the first.
    assert events[1].predecessor_event_ids == [events[0].event_id]
    # Tool latency (1200ms wall) preserved as spacing, under default 60s cap.
    assert events[1].wait_ms == 1200


def test_tracelab_idle_gap_cap_and_ignore_delays(tmp_path: Path) -> None:
    rows = [
        {
            "session_id": "s",
            "round_index": 0,
            "provider": "claude",
            "model": "m",
            "input_tokens_total": 10,
            "output_tokens": 2,
            "tools": [],
        },
        {
            "session_id": "s",
            "round_index": 1,
            "provider": "claude",
            "model": "m",
            "input_tokens_total": 15,
            "output_tokens": 2,
            "tools": [{"tool_name": "bash", "tool_wall_latency_ms": 5000.0}],
        },
    ]
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", rows)

    capped = _make_gen([str(trace_file)], trace_idle_gap_cap_seconds=1.0)
    events = sorted(next(s for s in capped.sessions if s is not None).graph.events.values(), key=lambda e: e.t_start_ms)
    assert events[1].wait_ms == 1000

    ignored = _make_gen([str(trace_file)], ignore_trace_delays=True)
    events = sorted(next(s for s in ignored.sessions if s is not None).graph.events.values(), key=lambda e: e.t_start_ms)
    assert events[1].wait_ms == 0


def test_tracelab_filter_selects_sessions(tmp_path: Path) -> None:
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", _rows())
    gen = _make_gen([str(trace_file)], filter="lambda x: x['num_rounds'] > 1")
    assert gen.get_session_count() == 1
    assert gen.sessions[0] is not None and gen.sessions[0].source_id == "claude:sess-A"


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
