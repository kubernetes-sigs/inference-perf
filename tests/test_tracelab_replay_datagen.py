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
import random
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig, DataConfig, DataGenType
from inference_perf.config.datagen.replay import UNCAPPED_WAIT_MS, TraceLabTraceReplayConfig
from inference_perf.datagen.replay.otel_trace_to_replay_graph import (
    RawCall,
    find_predecessors_by_text_matching,
)
from inference_perf.datagen.replay.replay_graph_types import ReplayMessage
from inference_perf.datagen.replay.tracelab_trace_replay_datagen import (
    FilterExpressionError,
    TraceLabTraceReplayDataGenerator,
    _find_tracelab_predecessors,
)
from inference_perf.utils.custom_tokenizer import CustomTokenizer


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
            "timing_events": [
                {"event_type": "user_message", "timestamp": "2026-06-01T10:00:10.000Z"},
                {"event_type": "text", "timestamp": "2026-06-01T10:00:12.000Z"},
            ],
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
            "timing_events": [
                {"event_type": "user_message", "timestamp": "2026-06-01T10:00:12.500Z"},
                {"event_type": "tool_call", "timestamp": "2026-06-01T10:00:13.000Z"},
                {"event_type": "tool_result", "timestamp": "2026-06-01T10:00:15.000Z"},
                {"event_type": "text", "timestamp": "2026-06-01T10:00:16.000Z"},
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
            "timing_events": [
                {"event_type": "user_message", "timestamp": "2026-06-01T10:00:16.700Z"},
                {"event_type": "text", "timestamp": "2026-06-01T10:00:18.000Z"},
            ],
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
    # Accepts the add_special_tokens/skip_special_tokens kwargs the generator
    # passes; the codec has no special tokens to add or skip.
    tok.get_tokenizer().encode = lambda text, **kwargs: list(range(len(text)))
    tok.get_tokenizer().decode = lambda toks, **kwargs: "x" * len(toks)
    return tok


def _make_gen(
    trace_files: List[str],
    **overrides: Any,
) -> TraceLabTraceReplayDataGenerator:
    api_cfg = APIConfig(type=APIType.Chat, streaming=False)
    data_cfg = DataConfig(type=DataGenType.TraceLabTraceReplay)
    data_cfg.tracelab_trace_replay = TraceLabTraceReplayConfig(trace_files=trace_files, **overrides)
    return TraceLabTraceReplayDataGenerator(api_config=api_cfg, config=data_cfg, tokenizer=_mock_tokenizer())


# Same as _make_gen but for the Hugging Face source, which takes either a repo
# id or a {"path": ..., **load_dataset_kwargs} dict.
def _make_hf_gen(hf_dataset_path: Any, **overrides: Any) -> TraceLabTraceReplayDataGenerator:
    api_cfg = APIConfig(type=APIType.Chat, streaming=False)
    data_cfg = DataConfig(type=DataGenType.TraceLabTraceReplay)
    data_cfg.tracelab_trace_replay = TraceLabTraceReplayConfig(hf_dataset_path=hf_dataset_path, **overrides)
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
    """Built prompt should tokenize to recorded input_tokens_total.

    Runs over growing (_rows), cache-miss, and compacting (_real_rows)
    sessions: shrinking rounds must truncate history to the shared prefix,
    not replay the full transcript.
    """
    for rows in (_rows(), _real_rows()):
        trace_file = _write_jsonl(tmp_path / "trace.jsonl", rows)
        gen = _make_gen([str(trace_file)])
        tok = _mock_tokenizer()

        for s in gen.sessions:
            assert s is not None
            for e in s.graph.events.values():
                # Tokenize the built messages and compare to recorded total_input_tokens.
                built_tokens = _tokenize_messages(tok, e.call.messages)
                recorded = e.call.total_input_tokens
                assert built_tokens == recorded, f"Event {e.event_id}: built {built_tokens} vs recorded {recorded}"


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
    """Rows missing required fields fail with the file and line named."""
    bad_rows = [
        {"session_id": "s", "round_index": 0},  # missing input_tokens_total, output_tokens
    ]
    trace_file = _write_jsonl(tmp_path / "bad.jsonl", bad_rows)
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
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


def test_tracelab_runtime_session_id_has_no_colon(tmp_path: Path) -> None:
    """Qualified event ids split on the first ':' to recover the session, so
    the runtime session id must not contain ':' even when the trace id does."""
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", _real_rows())
    gen = _make_gen([str(trace_file)])
    for s in gen.sessions:
        assert s is not None
        assert ":" not in s.session_id, f"runtime session id carries ':': {s.session_id}"
        for e in s.graph.events.values():
            qualified = f"{s.session_id}:{e.event_id}"
            assert qualified.split(":")[0] == s.session_id


def test_tracelab_timing_events_gap(tmp_path: Path) -> None:
    """Gaps from timing_events timestamps land before the following round.

    Round 1 is submitted when its last *input* event arrives, so a round fed by
    a tool_result is timed from that tool_result (15.0s), not from the earlier
    user_message (12.5s): 15.0 - 12.0 = 3000ms. ``usage_report`` rows and the
    model-output events of the *previous* round never shorten this gap.
    """
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", _real_rows())
    gen = _make_gen([str(trace_file)])

    sessions = {s.source_id: s for s in gen.sessions if s is not None}
    sess = sessions["claude:abc"]
    events = sorted(sess.graph.events.values(), key=lambda e: e.t_start_ms)
    assert len(events) == 3
    # Round 0 -> round 1: round 0 completes at 12.0s, round 1 submits at 15.0s.
    assert events[1].wait_ms == 3000
    # Round 1 -> round 2: 16.000 -> 16.700 = 700ms.
    assert events[2].wait_ms == 700

    capped = _make_gen([str(trace_file)], trace_idle_gap_cap_seconds=0.6)
    sessions = {s.source_id: s for s in capped.sessions if s is not None}
    events = sorted(sessions["claude:abc"].graph.events.values(), key=lambda e: e.t_start_ms)
    assert events[1].wait_ms == 600
    assert events[2].wait_ms == 600


def test_tracelab_hf_join_uses_tool_calls_config() -> None:
    """The HF join must read the tool_calls/timing_events configs (not tools)
    and join them onto rounds by round_pk."""
    import sys
    from unittest.mock import patch

    rounds = [
        {
            "round_pk": 1,
            "session_id": "claude:s1",
            "round_index": 0,
            "provider": "claude",
            "model": "m",
            "input_tokens_total": 20,
            "prefix_tokens": 0,
            "newly_append_tokens": 20,
            "output_tokens": 5,
        },
        {
            "round_pk": 2,
            "session_id": "claude:s1",
            "round_index": 1,
            "provider": "claude",
            "model": "m",
            "input_tokens_total": 30,
            "prefix_tokens": 20,
            "newly_append_tokens": 10,
            "output_tokens": 5,
        },
    ]
    tool_calls = [
        {"round_pk": 1, "tool_name": "bash", "tool_wall_latency_ms": 1200, "tool_internal_latency_ms": None},
    ]
    timing_events = [
        {"round_pk": 1, "event_type": "user_message", "timestamp": "2026-06-01T10:00:10.000Z"},
        {"round_pk": 1, "event_type": "text", "timestamp": "2026-06-01T10:00:11.000Z"},
        {"round_pk": 2, "event_type": "user_message", "timestamp": "2026-06-01T10:00:11.500Z"},
        {"round_pk": 2, "event_type": "text", "timestamp": "2026-06-01T10:00:12.000Z"},
    ]

    def fake_load_dataset(path: str, **kwargs: Any) -> Any:
        name = kwargs.get("name")
        if name == "tool_calls":
            return tool_calls
        if name == "timing_events":
            return timing_events
        assert name is None, f"unexpected config name: {name!r}"
        return rounds

    fake_datasets = MagicMock()
    fake_datasets.load_dataset = fake_load_dataset
    with patch.dict(sys.modules, {"datasets": fake_datasets}):
        api_cfg = APIConfig(type=APIType.Chat, streaming=False)
        data_cfg = DataConfig(type=DataGenType.TraceLabTraceReplay)
        data_cfg.tracelab_trace_replay = TraceLabTraceReplayConfig(hf_dataset_path="UW-SyFI/TraceLab")
        gen = TraceLabTraceReplayDataGenerator(api_config=api_cfg, config=data_cfg, tokenizer=_mock_tokenizer())

    assert gen.get_session_count() == 1
    session = gen.sessions[0]
    assert session is not None
    events = sorted(session.graph.events.values(), key=lambda e: e.t_start_ms)
    assert len(events) == 2
    # Tools joined from tool_calls (1200ms on round 0); timing gap 500ms wins
    # as the recorded think time before round 1.
    assert events[1].wait_ms == 500


def test_tracelab_hf_missing_aux_table_is_loud() -> None:
    """A missing aux table warns under skip_invalid_files, else raises."""
    import sys
    from unittest.mock import patch

    rounds = [
        {
            "round_pk": 1,
            "session_id": "s",
            "round_index": 0,
            "provider": "claude",
            "model": "m",
            "input_tokens_total": 10,
            "output_tokens": 5,
        },
    ]

    def fake_load_dataset(path: str, **kwargs: Any) -> Any:
        if kwargs.get("name") in ("tool_calls", "timing_events"):
            raise ValueError("no such config")
        return rounds

    fake_datasets = MagicMock()
    fake_datasets.load_dataset = fake_load_dataset

    def _make(skip_invalid: bool) -> TraceLabTraceReplayDataGenerator:
        api_cfg = APIConfig(type=APIType.Chat, streaming=False)
        data_cfg = DataConfig(type=DataGenType.TraceLabTraceReplay)
        data_cfg.tracelab_trace_replay = TraceLabTraceReplayConfig(
            hf_dataset_path="UW-SyFI/TraceLab", skip_invalid_files=skip_invalid
        )
        return TraceLabTraceReplayDataGenerator(api_config=api_cfg, config=data_cfg, tokenizer=_mock_tokenizer())

    with patch.dict(sys.modules, {"datasets": fake_datasets}):
        gen = _make(True)
        assert gen.get_session_count() == 1
        with pytest.raises(ValueError, match="tool_calls"):
            _make(False)


def test_tracelab_hf_shaped_rows_without_tools_or_timing(tmp_path: Path) -> None:
    """Rounds-table-shaped rows (no tools/timing_events keys) load with zero waits."""
    rows = [
        {
            "round_pk": "s1-r1",
            "session_id": "sess-hf",
            "round_index": 0,
            "provider": "codex",
            "model": "gpt-5",
            "input_tokens_total": 100,
            "prefix_tokens": 0,
            "newly_append_tokens": 100,
            "output_tokens": 10,
        },
        {
            "round_pk": "s1-r2",
            "session_id": "sess-hf",
            "round_index": 1,
            "provider": "codex",
            "model": "gpt-5",
            "input_tokens_total": 150,
            "prefix_tokens": 90,
            "newly_append_tokens": 60,
            "output_tokens": 10,
        },
    ]
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", rows)
    gen = _make_gen([str(trace_file)])
    assert gen.get_session_count() == 1
    session = gen.sessions[0]
    assert session is not None
    events = sorted(session.graph.events.values(), key=lambda e: e.t_start_ms)
    assert len(events) == 2
    assert events[1].wait_ms == 0
    assert events[1].predecessor_event_ids == [events[0].event_id]


def test_tracelab_determinism_across_subprocesses(tmp_path: Path) -> None:
    """Same trace built under different PYTHONHASHSEEDs yields identical prompts.

    Guards against hash()-based offsets, which randomize per process.
    """
    import os
    import subprocess
    import sys

    trace_file = _write_jsonl(tmp_path / "trace.jsonl", _real_rows())
    child = (
        "import asyncio, json, sys, types;"
        "m = types.ModuleType('uvloop');"
        "m.EventLoopPolicy = asyncio.DefaultEventLoopPolicy;"
        "sys.modules['uvloop'] = m;"
        "from unittest.mock import MagicMock;"
        "from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType;"
        "from inference_perf.config.datagen.replay import TraceLabTraceReplayConfig;"
        "from inference_perf.datagen.replay.tracelab_trace_replay_datagen import TraceLabTraceReplayDataGenerator;"
        "tok = MagicMock();"
        "tok.get_tokenizer().encode = lambda text, **kw: list(range(len(text)));"
        "tok.get_tokenizer().decode = lambda toks, **kw: '<%d:%d>' % (toks[0], len(toks)) if toks else '';"
        "api_cfg = APIConfig(type=APIType.Chat, streaming=False);"
        "data_cfg = DataConfig(type=DataGenType.TraceLabTraceReplay);"
        "data_cfg.tracelab_trace_replay = TraceLabTraceReplayConfig(trace_files=[sys.argv[1]]);"
        "gen = TraceLabTraceReplayDataGenerator(api_config=api_cfg, config=data_cfg, tokenizer=tok);"
        "out = {};"
        "[(out.setdefault(s.source_id, {}).__setitem__(e.event_id, "
        "''.join((mm.get('content', '') if isinstance(mm, dict) else mm.text) for mm in e.call.messages))) "
        "for s in gen.sessions if s is not None for e in s.graph.events.values()];"
        "print(json.dumps(out, sort_keys=True))"
    )
    outputs = []
    for seed in ("0", "1"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        proc = subprocess.run(
            [sys.executable, "-c", child, str(trace_file)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).parent.parent),
            timeout=300,
        )
        assert proc.returncode == 0, f"seed {seed} failed:\n{proc.stderr[-2000:]}"
        outputs.append(proc.stdout)
    assert outputs[0] == outputs[1], "prompts differ across hash seeds"


# num_dataset_entries caps sessions for local sources too. With a limit of two,
# the third session file and a malformed fourth file are never read.
def test_tracelab_num_dataset_entries_caps_local_sessions(tmp_path: Path) -> None:
    rows = _rows()
    a = _write_jsonl(tmp_path / "a.jsonl", rows[:2])  # sess-A, two rounds
    b = _write_jsonl(tmp_path / "b.jsonl", rows[2:3])  # sess-B, one round
    c_row = dict(rows[2])
    c_row["session_id"] = "sess-C"
    c = _write_jsonl(tmp_path / "c.jsonl", [c_row])
    unreadable = tmp_path / "d.jsonl"
    unreadable.write_text("not json\n", encoding="utf-8")

    gen = _make_gen([str(a), str(b), str(c), str(unreadable)], num_dataset_entries=2)

    assert gen.get_session_count() == 2
    sessions = {s.source_id: s for s in gen.sessions if s is not None}
    assert set(sessions) == {"sess-A", "sess-B"}
    # The last kept session is complete: both of sess-A's rounds were replayed.
    assert len(sessions["sess-A"].graph.events) == 2


# The cap counts kept sessions, not scanned ones: a session rejected by the
# filter must not consume one of the available slots.
def test_tracelab_num_dataset_entries_counts_kept_sessions_only(tmp_path: Path) -> None:
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", _rows())

    gen = _make_gen(
        [str(trace_file)],
        num_dataset_entries=1,
        filter="lambda x: x['session_id'] == 'sess-B'",
    )

    assert gen.get_session_count() == 1
    assert next(s for s in gen.sessions if s is not None).source_id == "sess-B"


# A small HF limit must stop iterating the rounds table and join aux rows only
# for kept rounds. Four rows are enough to finish two sessions; the fifth row
# (a fourth session) must never be read.
def test_tracelab_num_dataset_entries_caps_hf_rounds(tmp_path: Path) -> None:
    import sys
    from unittest.mock import patch

    def _round(pk: str, session_id: str, index: int) -> Dict[str, Any]:
        return {
            "round_pk": pk,
            "session_id": session_id,
            "round_index": index,
            "provider": "claude",
            "model": "m",
            "input_tokens_total": 10,
            "prefix_tokens": 0,
            "newly_append_tokens": 10,
            "output_tokens": 2,
        }

    class _RoundsTable:
        """Rounds iterable that fails if the loader reads past ``fail_at``."""

        def __init__(self, rows: List[Dict[str, Any]], fail_at: int) -> None:
            self.rows = rows
            self.fail_at = fail_at
            self.read = 0

        def __iter__(self) -> Any:
            for i, row in enumerate(self.rows):
                assert i < self.fail_at, f"loader read row {i}; it should have stopped at {self.fail_at}"
                self.read += 1
                yield row

    rounds = _RoundsTable(
        [
            _round("1", "s1", 0),
            _round("2", "s1", 1),
            _round("3", "s2", 0),
            _round("4", "s3", 0),
            _round("5", "s4", 0),
        ],
        fail_at=4,
    )
    tool_calls = [{"round_pk": "1", "tool_name": "bash", "tool_wall_latency_ms": 700, "tool_internal_latency_ms": None}]

    def fake_load_dataset(path: str, **kwargs: Any) -> Any:
        name = kwargs.get("name")
        if name == "tool_calls":
            return tool_calls
        if name == "timing_events":
            return []
        assert name is None, f"unexpected config name: {name!r}"
        return rounds

    fake_datasets = MagicMock()
    fake_datasets.load_dataset = fake_load_dataset
    with patch.dict(sys.modules, {"datasets": fake_datasets}):
        api_cfg = APIConfig(type=APIType.Chat, streaming=False)
        data_cfg = DataConfig(type=DataGenType.TraceLabTraceReplay)
        data_cfg.tracelab_trace_replay = TraceLabTraceReplayConfig(hf_dataset_path="UW-SyFI/TraceLab", num_dataset_entries=2)
        gen = TraceLabTraceReplayDataGenerator(api_config=api_cfg, config=data_cfg, tokenizer=_mock_tokenizer())

    assert gen.get_session_count() == 2
    assert rounds.read == 4
    sessions = {s.source_id: s for s in gen.sessions if s is not None}
    assert set(sessions) == {"s1", "s2"}
    # Tool latency for the kept first round still reaches the replay plan.
    events = sorted(sessions["s1"].graph.events.values(), key=lambda e: e.t_start_ms)
    assert events[1].wait_ms == 700


# TraceLab replay never substitutes live output, so the old flag is gone and a
# config that still sets it is rejected instead of silently doing nothing.
def test_tracelab_disable_output_substitution_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TraceLabTraceReplayConfig.model_validate({"trace_files": ["./traces/t.jsonl"], "disable_output_substitution": True})


# One TraceLab JSONL row: session/round identity plus the four recorded token
# counts. Extra keys (prefix_tokens, tools, timing_events) pass straight through.
def _row(session_id: str, round_index: int, in_tokens: int, out_tokens: int, **extra: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "session_id": session_id,
        "round_index": round_index,
        "provider": "claude",
        "model": "m",
        "input_tokens_total": in_tokens,
        "newly_append_tokens": in_tokens,
        "output_tokens": out_tokens,
    }
    row.update(extra)
    return row


# ISO-8601 UTC timestamp ``seconds`` after the 2026-06-01T10:00:00Z clock that
# every timing fixture below is measured from, e.g. _ts(30) -> 10:00:30Z.
def _ts(seconds: float) -> str:
    stamp = datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=seconds)
    return stamp.isoformat().replace("+00:00", "Z")


# One user_message/text pair spanning ``start`` -> ``end`` seconds, the shape a
# round with no tools records.
def _turn(start: float, end: float) -> List[Dict[str, str]]:
    return [
        {"event_type": "user_message", "timestamp": _ts(start)},
        {"event_type": "text", "timestamp": _ts(end)},
    ]


# Runtime wait (ms) before each event of the only session in ``gen``; the first
# event of a session always waits 0.
def _waits(gen: TraceLabTraceReplayDataGenerator) -> List[int]:
    per_session = list(gen._session_events.values())
    assert len(per_session) == 1, f"expected one session, got {len(per_session)}"
    return [e.wait_ms for e in per_session[0]]


# Model the HF loads for a config split-name table: IterableDatasetDict has no
# IterableDataset-like row access, only keys()/__getitem__.
class _SplitDict:
    def __init__(self, splits: Dict[str, List[Dict[str, Any]]]) -> None:
        self._splits = splits

    def keys(self) -> List[str]:
        return list(self._splits)

    def __getitem__(self, key: str) -> List[Dict[str, Any]]:
        return self._splits[key]


# trace_idle_gap_cap_seconds drives the runtime clamp: the inherited 15s
# max_wait_ms default would truncate anything above it, so an unset max_wait_ms
# is raised to the cap (60s by default, 30s here), a disabled cap sets the
# uncapped sentinel, and an explicit max_wait_ms is never overridden.
def test_tracelab_config_reconciles_max_wait_ms() -> None:
    assert TraceLabTraceReplayConfig(trace_files=["a.jsonl"]).max_wait_ms == 60_000
    assert TraceLabTraceReplayConfig(trace_files=["a.jsonl"], trace_idle_gap_cap_seconds=30.0).max_wait_ms == 30_000
    assert TraceLabTraceReplayConfig(trace_files=["a.jsonl"], trace_idle_gap_cap_seconds=None).max_wait_ms == UNCAPPED_WAIT_MS
    assert TraceLabTraceReplayConfig(trace_files=["a.jsonl"], trace_idle_gap_cap_seconds=0).max_wait_ms == UNCAPPED_WAIT_MS
    assert TraceLabTraceReplayConfig(trace_files=["a.jsonl"], trace_idle_gap_cap_seconds=-1.0).max_wait_ms == UNCAPPED_WAIT_MS
    # Explicit knobs win: a user who set max_wait_ms keeps that clamp.
    assert TraceLabTraceReplayConfig(trace_files=["a.jsonl"], max_wait_ms=5_000).max_wait_ms == 5_000
    # A cap below the inherited clamp leaves the larger clamp in place, which is
    # harmless because gaps are already capped at 5s before reaching the clamp.
    assert TraceLabTraceReplayConfig(trace_files=["a.jsonl"], trace_idle_gap_cap_seconds=5.0).max_wait_ms == 15_000


# A 29s human think pause must reach the replayed event: with the default 60s
# cap the runtime wait is 29_000ms, not the inherited 15_000ms max_wait_ms clamp.
def test_tracelab_wait_above_inherited_clamp(tmp_path: Path) -> None:
    rows = [
        _row("s", 0, 10, 2, timing_events=_turn(0, 1), prefix_tokens=0),
        _row("s", 1, 20, 2, timing_events=_turn(30, 31), prefix_tokens=10),
    ]
    gen = _make_gen([str(_write_jsonl(tmp_path / "trace.jsonl", rows))])

    assert gen.tracelab_config.max_wait_ms == 60_000
    assert _waits(gen) == [0, 29_000]


# Disabling the cap (None or <= 0) replays recorded think time verbatim, which
# is what a KV-cache TTL/eviction benchmark needs: a 5-minute pause stays
# 300_000ms instead of being clipped to the 60s default cap.
def test_tracelab_disabled_cap_replays_full_gap(tmp_path: Path) -> None:
    rows = [
        _row("s", 0, 10, 2, timing_events=_turn(0, 1)),
        _row("s", 1, 20, 2, timing_events=_turn(301, 302), prefix_tokens=10),
    ]
    trace_file = str(_write_jsonl(tmp_path / "trace.jsonl", rows))

    for cap in (None, 0, -1.0):
        gen = _make_gen([trace_file], trace_idle_gap_cap_seconds=cap)
        assert gen.tracelab_config.max_wait_ms == UNCAPPED_WAIT_MS, f"cap={cap!r} was clamped"
        assert _waits(gen) == [0, 300_000], f"cap={cap!r} truncated the gap"


# Codex writes a round's usage_report (token_count) after the tool it requested
# has finished, so it must not define either bound: round 0 completes at its
# text event (2s), not at the 9s usage_report, giving a 10s gap before round 1.
def test_tracelab_usage_report_excluded_from_round_bounds(tmp_path: Path) -> None:
    rows = [
        _row(
            "s",
            0,
            10,
            2,
            timing_events=[
                {"event_type": "user_message", "timestamp": _ts(0)},
                {"event_type": "text", "timestamp": _ts(2)},
                {"event_type": "usage_report", "timestamp": _ts(9)},
            ],
        ),
        _row(
            "s",
            1,
            20,
            2,
            prefix_tokens=10,
            timing_events=[
                {"event_type": "user_message", "timestamp": _ts(12)},
                {"event_type": "text", "timestamp": _ts(13)},
                {"event_type": "usage_report", "timestamp": _ts(20)},
            ],
        ),
    ]
    gen = _make_gen([str(_write_jsonl(tmp_path / "trace.jsonl", rows))])

    # 12s (round 1 submission) - 2s (round 0 text) = 10_000ms.
    assert _waits(gen) == [0, 10_000]


# A round fed by parallel tool calls is submitted when the LAST tool_result
# arrives, so two results at 5s and 8s make the request start at 8s: the gap
# after round 0 (which completed at 2s) is 6_000ms, not the 3_000ms the earliest
# result would give.
def test_tracelab_parallel_tool_results_submit_at_last_input(tmp_path: Path) -> None:
    rows = [
        _row("s", 0, 10, 2, timing_events=_turn(0, 2)),
        _row(
            "s",
            1,
            20,
            2,
            prefix_tokens=10,
            timing_events=[
                {"event_type": "tool_result", "timestamp": _ts(5)},
                {"event_type": "tool_result", "timestamp": _ts(8)},
                {"event_type": "text", "timestamp": _ts(9)},
            ],
        ),
    ]
    gen = _make_gen([str(_write_jsonl(tmp_path / "trace.jsonl", rows))])

    assert _waits(gen) == [0, 6_000]


# A round that recorded no input event still gets a submission time: the
# earliest model-output event is used, so round 1 starting with text at 6s gives
# a 4_000ms gap after round 0's 2s completion.
def test_tracelab_submission_falls_back_to_first_output_event(tmp_path: Path) -> None:
    rows = [
        _row("s", 0, 10, 2, timing_events=_turn(0, 2)),
        _row(
            "s",
            1,
            20,
            2,
            prefix_tokens=10,
            timing_events=[
                {"event_type": "reasoning", "timestamp": _ts(7)},
                {"event_type": "text", "timestamp": _ts(6)},
            ],
        ),
    ]
    gen = _make_gen([str(_write_jsonl(tmp_path / "trace.jsonl", rows))])

    assert _waits(gen) == [0, 4_000]


# Rows whose round_index disagrees with their timestamps are replayed in
# submission order (0s, 10s, 20s) rather than file or index order, so the
# conversation is not silently reordered.
def test_tracelab_rounds_ordered_by_submission_time(tmp_path: Path) -> None:
    rows = [
        _row("s", 0, 10, 2, timing_events=_turn(0, 1)),
        _row("s", 2, 30, 2, timing_events=_turn(20, 21), prefix_tokens=20),
        _row("s", 1, 20, 2, timing_events=_turn(10, 11), prefix_tokens=10),
    ]
    gen = _make_gen([str(_write_jsonl(tmp_path / "trace.jsonl", rows))])

    (session,) = [s for s in gen.sessions if s is not None]
    events = sorted(session.graph.events.values(), key=lambda e: e.t_start_ms)
    assert [e.call.total_input_tokens for e in events] == [10, 20, 30]
    assert _waits(gen) == [0, 9_000, 9_000]


# Two rows sharing a (session_id, round_index) pair are tie-broken by submission
# time instead of one silently replacing the other, so both rounds replay.
def test_tracelab_duplicate_round_index_keeps_both_rounds(tmp_path: Path) -> None:
    rows = [
        _row("s", 0, 10, 2, timing_events=_turn(0, 1)),
        _row("s", 0, 20, 2, timing_events=_turn(10, 11), prefix_tokens=10),
    ]
    gen = _make_gen([str(_write_jsonl(tmp_path / "trace.jsonl", rows))])

    (session,) = [s for s in gen.sessions if s is not None]
    events = sorted(session.graph.events.values(), key=lambda e: e.t_start_ms)
    assert [e.call.total_input_tokens for e in events] == [10, 20]
    assert _waits(gen) == [0, 9_000]


# Parallel tools run concurrently, so their [emitted_at, result_at] windows are
# merged before summing: two 5s tools overlapping from 0-5s and 2-7s occupy 7s
# of wall time, not the 10_000ms their latencies add up to.
def test_tracelab_parallel_tool_intervals_merged(tmp_path: Path) -> None:
    rows = [
        _row(
            "s",
            0,
            10,
            2,
            tools=[
                {"tool_name": "a", "tool_wall_latency_ms": 5000.0, "emitted_at": _ts(0), "result_at": _ts(5)},
                {"tool_name": "b", "tool_wall_latency_ms": 5000.0, "emitted_at": _ts(2), "result_at": _ts(7)},
            ],
        ),
        _row("s", 1, 20, 2, prefix_tokens=10),
    ]
    gen = _make_gen([str(_write_jsonl(tmp_path / "trace.jsonl", rows))])

    assert _waits(gen) == [0, 7_000]


# Non-overlapping tool windows still add up: 0-3s then 5-9s occupies 7s.
def test_tracelab_serial_tool_intervals_add_up(tmp_path: Path) -> None:
    rows = [
        _row(
            "s",
            0,
            10,
            2,
            tools=[
                {"tool_name": "a", "emitted_at": _ts(0), "result_at": _ts(3)},
                {"tool_name": "b", "emitted_at": _ts(5), "result_at": _ts(9)},
            ],
        ),
        _row("s", 1, 20, 2, prefix_tokens=10),
    ]
    gen = _make_gen([str(_write_jsonl(tmp_path / "trace.jsonl", rows))])

    assert _waits(gen) == [0, 7_000]


# A file whose stem contains dots (tracelab_v0.0.2.jsonl) must still be picked
# up by a trace_directory scan, and a gzipped JSON array (.json.gz) must be
# parsed as an array instead of falling through to line-by-line JSONL parsing.
def test_tracelab_dotted_stem_and_gzipped_json_array(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    _write_jsonl(traces / "tracelab_v0.0.2.jsonl", [_row("sess-A", 0, 10, 2)])
    with gzip.open(traces / "tracelab_v1.4.json.gz", "wt", encoding="utf-8") as f:
        json.dump([_row("sess-B", 0, 12, 3)], f)
    (traces / "notes.txt").write_text("not a trace", encoding="utf-8")

    api_cfg = APIConfig(type=APIType.Chat, streaming=False)
    data_cfg = DataConfig(type=DataGenType.TraceLabTraceReplay)
    data_cfg.tracelab_trace_replay = TraceLabTraceReplayConfig(trace_directory=str(traces))
    gen = TraceLabTraceReplayDataGenerator(api_config=api_cfg, config=data_cfg, tokenizer=_mock_tokenizer())

    assert {s.source_id for s in gen.sessions if s is not None} == {"sess-A", "sess-B"}


# A session that reappears after another session is a structural error: the
# streaming early stop assumes contiguity, and overwriting would drop the rounds
# already read. skip_invalid_files covers bad rows, not this.
def test_tracelab_non_contiguous_session_raises(tmp_path: Path) -> None:
    rows = [
        _row("s1", 0, 10, 2),
        _row("s2", 0, 10, 2),
        _row("s1", 1, 20, 2, prefix_tokens=10),
    ]
    trace_file = str(_write_jsonl(tmp_path / "trace.jsonl", rows))

    with pytest.raises(ValueError, match="reappears"):
        _make_gen([trace_file])
    with pytest.raises(ValueError, match="reappears"):
        _make_gen([trace_file], skip_invalid_files=True)


# The HF tables must be streamed by default so a small run never Arrow-builds
# the full ~665k-round corpus: every load_dataset call gets streaming=True, and
# an IterableDatasetDict split is selected by _as_split.
def test_tracelab_hf_streams_by_default() -> None:
    import sys
    from unittest.mock import patch

    seen: List[Dict[str, Any]] = []

    def fake_load_dataset(path: str, **kwargs: Any) -> Any:
        seen.append(kwargs)
        name = kwargs.get("name")
        if name in ("tool_calls", "timing_events"):
            return _SplitDict({"train": []})
        return _SplitDict(
            {
                "train": [
                    dict(_row("s", 0, 10, 2), round_pk="1"),
                    dict(_row("s", 1, 20, 2, prefix_tokens=10), round_pk="2"),
                ]
            }
        )

    fake_datasets = MagicMock()
    fake_datasets.load_dataset = fake_load_dataset
    with patch.dict(sys.modules, {"datasets": fake_datasets}):
        gen = _make_hf_gen("UW-SyFI/TraceLab")

    assert gen.get_session_count() == 1
    assert [kwargs.get("streaming") for kwargs in seen] == [True, True, True]


# A caller can opt back into full local caching from the config dict:
# {"path": ..., "streaming": false} reaches load_dataset as streaming=False
# (and is not also passed as a duplicate keyword).
def test_tracelab_hf_streaming_false_is_honoured() -> None:
    import sys
    from unittest.mock import patch

    seen: List[Dict[str, Any]] = []

    def fake_load_dataset(path: str, **kwargs: Any) -> Any:
        seen.append(kwargs)
        name = kwargs.get("name")
        if name in ("tool_calls", "timing_events"):
            return []
        return [dict(_row("s", 0, 10, 2), round_pk="1")]

    fake_datasets = MagicMock()
    fake_datasets.load_dataset = fake_load_dataset
    with patch.dict(sys.modules, {"datasets": fake_datasets}):
        _make_hf_gen({"path": "UW-SyFI/TraceLab", "streaming": False})

    assert [kwargs.get("streaming") for kwargs in seen] == [False, False, False]


# A small custom corpus is repeated as many times as the prompt needs: asking
# for 300 tokens from a ~80-token corpus yields exactly 300 tokens, not the
# ~160 the old single-wrap code produced with only a warning.
def test_tracelab_small_corpus_fills_long_prompt(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("line one\nline two\n", encoding="utf-8")
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", [_row("s", 0, 300, 5)])

    api_cfg = APIConfig(type=APIType.Chat, streaming=False)
    data_cfg = DataConfig(type=DataGenType.TraceLabTraceReplay)
    data_cfg.corpus_file_path = str(corpus)
    data_cfg.tracelab_trace_replay = TraceLabTraceReplayConfig(trace_files=[str(trace_file)])
    gen = TraceLabTraceReplayDataGenerator(api_config=api_cfg, config=data_cfg, tokenizer=_mock_tokenizer())

    (session,) = [s for s in gen.sessions if s is not None]
    (event,) = list(session.graph.events.values())
    prompt = "".join(m.get("content", "") if isinstance(m, dict) else m.text for m in event.call.messages)
    assert len(prompt) == 300


# A round whose recorded prefix_tokens exceeds its own input_tokens_total (a
# counting oddity after compaction) must still build a prompt of exactly
# input_tokens_total tokens: the shared run is trimmed to the round's size.
def test_tracelab_prefix_larger_than_prompt_is_trimmed(tmp_path: Path) -> None:
    rows = [
        _row("s", 0, 50, 5),
        _row("s", 1, 40, 5, prefix_tokens=90),
    ]
    gen = _make_gen([str(_write_jsonl(tmp_path / "trace.jsonl", rows))])

    tokenizer = _mock_tokenizer()
    (session,) = [s for s in gen.sessions if s is not None]
    for event in session.graph.events.values():
        built = _tokenize_messages(tokenizer, event.call.messages)
        assert built == event.call.total_input_tokens, f"built {built} vs recorded {event.call.total_input_tokens}"


# A round that recorded zero output tokens falls back to this replay's
# default_max_tokens: leaving the recorded 0 reaches the wire as max_tokens=0,
# which the client replaces with its own default instead of ours.
def test_tracelab_zero_output_tokens_uses_default_max_tokens(tmp_path: Path) -> None:
    rows = [_row("s", 0, 10, 0), _row("s", 1, 20, 7, prefix_tokens=10)]
    gen = _make_gen([str(_write_jsonl(tmp_path / "trace.jsonl", rows))], default_max_tokens=321)

    events = next(iter(gen._session_events.values()))
    assert events[0].max_tokens_recorded == 321
    assert events[1].max_tokens_recorded == 7


# With a real BOS-adding tokenizer (gemma-3-270m, bos_token_id=2), round i's
# tokenized prompt must start with its recorded prefix_tokens of round i-1's ids:
# a per-round <bos> would shift every later token right by one and misalign
# every fixed-size KV-cache block from block 0 onward.
def test_tracelab_bos_tokenizer_preserves_prefix_across_rounds(tmp_path: Path) -> None:
    tarball = Path(__file__).resolve().parents[1] / "e2e" / "testdata" / "models" / "google_gemma-3-270m.tar.gz"
    model_dir = tmp_path / "gemma"
    with tarfile.open(tarball) as tar:
        tar.extractall(model_dir, filter="data")
    tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path=str(model_dir)))
    raw = tokenizer.get_tokenizer()
    assert raw.encode("hi")[0] == 2, "expected the tokenizer to prepend <bos>"

    rows = [
        _row("s", 0, 200, 20, timing_events=_turn(0, 1)),
        _row("s", 1, 260, 20, prefix_tokens=200, timing_events=_turn(2, 3)),
        _row("s", 2, 300, 20, prefix_tokens=240, timing_events=_turn(4, 5)),
    ]
    trace_file = _write_jsonl(tmp_path / "trace.jsonl", rows)
    api_cfg = APIConfig(type=APIType.Chat, streaming=False)
    data_cfg = DataConfig(type=DataGenType.TraceLabTraceReplay)
    data_cfg.tracelab_trace_replay = TraceLabTraceReplayConfig(trace_files=[str(trace_file)])
    gen = TraceLabTraceReplayDataGenerator(api_config=api_cfg, config=data_cfg, tokenizer=tokenizer)

    (session,) = [s for s in gen.sessions if s is not None]
    events = sorted(session.graph.events.values(), key=lambda e: e.t_start_ms)
    assert len(events) == 3
    prompts = ["".join(m.get("content", "") if isinstance(m, dict) else m.text for m in e.call.messages) for e in events]
    ids = [raw.encode(prompt, add_special_tokens=False) for prompt in prompts]
    assert len(ids[0]) == 200
    assert ids[1][:200] == ids[0][:200]
    assert ids[2][:240] == ids[1][:240]


# The linear predecessor finder must reproduce the generic O(n^2) text-matching
# finder exactly for TraceLab's single-user-message rounds (a plain temporal
# chain), and defer to it unchanged when a call carries assistant messages.
def test_tracelab_linear_predecessor_finder_matches_generic() -> None:
    def _call(index: int) -> RawCall:
        return RawCall(
            call_id=f"round_{index}",
            trace_id="s",
            t_start_ms=index * 1000,
            t_end_ms=index * 1000,
            model="m",
            messages=[ReplayMessage(role="user", text=f"prompt {index}")],
            out_message=ReplayMessage(role="assistant", text=f"out {index}"),
            prompt_tokens=10,
            completion_tokens=5,
            temperature=0.0,
            max_tokens_recorded=5,
        )

    chain = [_call(i) for i in range(3)]
    assert _find_tracelab_predecessors(chain) == find_predecessors_by_text_matching(chain)

    with_assistant = [_call(0), _call(1)]
    with_assistant[1].messages = [
        ReplayMessage(role="user", text="prompt 1"),
        ReplayMessage(role="assistant", text="out 0"),
    ]
    assert _find_tracelab_predecessors(with_assistant) == find_predecessors_by_text_matching(with_assistant)


# Session shuffling uses a local random.Random(base_seed), so building a
# generator (a) leaves the process-global RNG untouched and (b) still produces
# the same session order on every run.
def test_tracelab_does_not_mutate_global_random_state(tmp_path: Path) -> None:
    trace_file = str(_write_jsonl(tmp_path / "trace.jsonl", _real_rows()))

    random.seed(1234)
    state = random.getstate()
    first = _make_gen([trace_file])
    assert random.getstate() == state

    second = _make_gen([trace_file])
    assert [s.source_id for s in first.sessions if s is not None] == [s.source_id for s in second.sessions if s is not None]
