#!/usr/bin/env python3

# Copyright 2025 The Kubernetes Authors.
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

"""Tests for LocalTraceSource (Phase 2).

Two things are being pinned:

  * Equivalence. LocalTraceSource must produce the same normalized records the
    eager _rows_for_file path produced, including its session-id fallbacks.
  * Boundedness. No aggregate Dataset is built, and record bodies are decoded
    only when their session is selected.

Boundedness is measured in JSON decodes, not file opens: format detection reads
content by design (detect_trace_format), and for a .json file that means parsing
it whole since it may be pretty-printed. Counting opens would fail for reasons
that have nothing to do with aggregation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pytest

from inference_perf.datagen.replay.otel_trace_replay_datagen import (
    _normalize_file_trace,
    _rows_for_file,
)
from inference_perf.datagen.replay.trace_source import (
    InvalidTraceError,
    LocalTraceFormat,
    LocalTraceSource,
    validate_record_schema,
)
from inference_perf.datagen.replay.wire_trace_converter import derive_trace_id


def otel_trace(trace_id: Optional[str] = "t-1", model: str = "m") -> Dict[str, Any]:
    doc: Dict[str, Any] = {
        "span_count": 1,
        "collected_at": "2026-01-01T00:00:00+00:00",
        "spans": [
            {
                "attributes": {
                    "gen_ai.request.model": model,
                    "gen_ai.usage.input_tokens": 3,
                    "gen_ai.usage.output_tokens": 4,
                }
            }
        ],
    }
    if trace_id is not None:
        doc["trace_id"] = trace_id
    return doc


def chat_record(session_id: Optional[str] = None) -> Dict[str, Any]:
    """One Chat Completions wire record, with the timing fields the converter reads."""
    record: Dict[str, Any] = {
        "path": "/v1/chat/completions",
        "start_unix": 1767225600.0,
        "total_ms": 1200,
        "request": json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
        "response": json.dumps(
            {
                "choices": [{"message": {"role": "assistant", "content": "yo"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5},
            }
        ),
    }
    if session_id:
        # One of _SESSION_ID_HEADERS in wire_trace_converter; others are ignored.
        record["headers"] = {"session_id": session_id}
    return record


def write_jsonl(path: Path, docs: List[Dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
    return path


def source_for(files: List[Path], skip_invalid: bool = False) -> LocalTraceSource:
    return LocalTraceSource(files, skip_invalid, _normalize_file_trace)


def scrub(record: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the fields convert_wire_file regenerates per call.

    span_id comes from a fresh uuid and collected_at from datetime.now(), so two
    conversions of one file never compare equal on them -- in either code path.
    """
    out = dict(record)
    out.pop("collected_at", None)
    out["spans"] = [{k: v for k, v in span.items() if k != "span_id"} for span in out["spans"]]
    return out


# --------------------------------------------------------------------------------------
# Record counting: files are not one-to-one with records
# --------------------------------------------------------------------------------------


def test_multi_record_jsonl_counts_records_not_files(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "traces.jsonl", [otel_trace(f"t{i}") for i in range(3)])
    assert len(source_for([tmp_path / "traces.jsonl"]).list_records()) == 3


def test_blank_lines_do_not_consume_a_row_index(tmp_path: Path) -> None:
    """Matches iter_otel_jsonl_traces, which skips blanks without yielding."""
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps(otel_trace("a")) + "\n\n" + json.dumps(otel_trace("b")) + "\n\n",
        encoding="utf-8",
    )
    source = source_for([path])
    assert len(source.list_records()) == 2
    assert [source.load_record(i)["trace_id"] for i in range(2)] == ["a", "b"]


def test_wire_capture_is_one_record_per_file(tmp_path: Path) -> None:
    """convert_wire_file returns a single document: one file is one session."""
    write_jsonl(tmp_path / "calls.jsonl", [chat_record(), chat_record(), chat_record()])
    assert len(source_for([tmp_path / "calls.jsonl"]).list_records()) == 1


def test_json_file_is_one_record(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps(otel_trace()), encoding="utf-8")
    assert len(source_for([tmp_path / "a.json"]).list_records()) == 1


# --------------------------------------------------------------------------------------
# Equivalence with the eager path
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 3])
def test_jsonl_session_ids_match_eager_path(tmp_path: Path, count: int) -> None:
    """Single-record files keep a plain id; multi-record files get row suffixes."""
    docs = [otel_trace(trace_id=None) for _ in range(count)]
    path = write_jsonl(tmp_path / "traces.jsonl", docs)

    source = source_for([path])
    lazy = [source.load_record(i)["session_id"] for i in range(count)]
    eager = [row["session_id"] for row in _rows_for_file(path, skip_invalid=False)]

    assert lazy == eager
    assert len(set(lazy)) == count


def test_json_record_matches_eager_path_exactly(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(otel_trace("keep-me")), encoding="utf-8")

    assert source_for([path]).load_record(0) == _rows_for_file(path, skip_invalid=False)[0]


def test_wire_record_matches_eager_path(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "calls.jsonl", [chat_record()])

    lazy = source_for([path]).load_record(0)
    eager = _rows_for_file(path, skip_invalid=False)[0]

    assert scrub(lazy) == scrub(eager)


# --------------------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------------------


def test_wire_identity_survives_duplicate_filenames(tmp_path: Path) -> None:
    """Capture trees name every session's file calls.jsonl, so stems collide."""
    paths = []
    for name in ("s1", "s2"):
        (tmp_path / name).mkdir()
        paths.append(write_jsonl(tmp_path / name / "calls.jsonl", [chat_record()]))

    suffixes = [r.session_id_suffix for r in source_for(paths).list_records()]
    assert len(set(suffixes)) == 2
    assert suffixes[0] == derive_trace_id(paths[0])


def test_wire_identity_prefers_recorded_session_header(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "calls.jsonl", [chat_record(session_id="abc-def")])
    assert source_for([path]).list_records()[0].session_id_suffix == "abcdef"


def test_multi_record_jsonl_source_ids_are_distinct(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "traces.jsonl", [otel_trace("a"), otel_trace("b")])
    source_ids = [r.source_id for r in source_for([path]).list_records()]
    assert source_ids == [f"{path}#0", f"{path}#1"]


def test_json_identity_comes_from_the_filename_not_the_slot(tmp_path: Path) -> None:
    """Scheduler ids must stay traceable to their file.

    A locator cannot see a record's embedded session_id, but falling back to a
    bare slot number would make session ids in logs and reports meaningless for
    .json corpora, where every file is one trace.
    """
    for name in ("run_a", "run_b"):
        (tmp_path / f"{name}.json").write_text(json.dumps(otel_trace(trace_id=None)), encoding="utf-8")

    suffixes = [r.session_id_suffix for r in source_for(sorted(tmp_path.glob("*.json"))).list_records()]

    assert suffixes == ["run_a", "run_b"]


def test_single_record_source_id_is_the_bare_path(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "solo.jsonl", [otel_trace()])
    assert source_for([path]).list_records()[0].source_id == str(path)


# --------------------------------------------------------------------------------------
# Boundedness
# --------------------------------------------------------------------------------------


def test_lazy_indexing_does_not_decode_jsonl_bodies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With validation deferred, the .jsonl scan counts lines without decoding them."""
    write_jsonl(tmp_path / "traces.jsonl", [otel_trace(f"t{i}") for i in range(25)])

    decodes: List[int] = []
    real_loads = json.loads

    def counting_loads(s: Any, **kwargs: Any) -> Any:
        decodes.append(1)
        return real_loads(s, **kwargs)

    monkeypatch.setattr(json, "loads", counting_loads)
    source = LocalTraceSource([tmp_path / "traces.jsonl"], False, _normalize_file_trace, validate_at_startup=False)
    indexing_decodes = len(decodes)

    # Format detection probes the first line; the 24 remaining records must not be
    # decoded to know the file holds 25 of them.
    assert indexing_decodes <= 2, f"indexing decoded {indexing_decodes} records"
    assert len(source.list_records()) == 25


def test_loading_the_last_row_of_a_jsonl_decodes_exactly_one_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Byte-range addressing: scanning to row n would decode every line before it."""
    path = write_jsonl(tmp_path / "traces.jsonl", [otel_trace(f"t{i}") for i in range(50)])
    source = source_for([path])

    decodes: List[Any] = []
    real_loads = json.loads

    def counting_loads(s: Any, **kwargs: Any) -> Any:
        decodes.append(s)
        return real_loads(s, **kwargs)

    monkeypatch.setattr(json, "loads", counting_loads)
    record = source.load_record(49)

    assert record["trace_id"] == "t49"
    assert len(decodes) == 1, f"loading the last row decoded {len(decodes)} records"


def test_loading_every_jsonl_row_stays_linear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One decode per record across a full pass, not O(N^2)."""
    n = 40
    path = write_jsonl(tmp_path / "traces.jsonl", [otel_trace(f"t{i}") for i in range(n)])
    source = source_for([path])

    decodes: List[Any] = []
    real_loads = json.loads

    def counting_loads(s: Any, **kwargs: Any) -> Any:
        decodes.append(s)
        return real_loads(s, **kwargs)

    monkeypatch.setattr(json, "loads", counting_loads)
    ids = [source.load_record(i)["trace_id"] for i in range(n)]

    assert ids == [f"t{i}" for i in range(n)]
    assert len(decodes) == n, f"{n} loads decoded {len(decodes)} records"


def test_jsonl_byte_ranges_are_recorded_for_every_row(tmp_path: Path) -> None:
    """The locator carries the range; other formats leave it unset."""
    jsonl = write_jsonl(tmp_path / "traces.jsonl", [otel_trace("a"), otel_trace("b")])
    (tmp_path / "solo.json").write_text(json.dumps(otel_trace()), encoding="utf-8")
    wire = write_jsonl(tmp_path / "calls.jsonl", [chat_record()])

    by_name = {r.path.name: r for r in source_for([jsonl, tmp_path / "solo.json", wire]).records}
    rows = [r for r in source_for([jsonl]).records]

    assert rows[0].byte_offset == 0
    assert rows[1].byte_offset == rows[0].byte_length
    assert all(r.byte_length and r.byte_length > 0 for r in rows)
    assert by_name["solo.json"].byte_offset is None
    assert by_name["calls.jsonl"].byte_offset is None


def test_blank_lines_do_not_shift_recorded_byte_ranges(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text(
        "\n" + json.dumps(otel_trace("a")) + "\n\n\n" + json.dumps(otel_trace("b")) + "\n",
        encoding="utf-8",
    )
    source = source_for([path])

    assert [source.load_record(i)["trace_id"] for i in range(2)] == ["a", "b"]


def test_loading_one_record_decodes_only_that_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = [write_jsonl(tmp_path / f"t{i}.jsonl", [otel_trace(f"t{i}")]) for i in range(10)]
    source = source_for(paths)

    reads: List[str] = []
    real_open = Path.open

    def counting_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        reads.append(self.name)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    source.load_record(4)

    assert reads == ["t4.jsonl"], f"expected only t4.jsonl to be read, got {reads}"


def test_no_aggregate_dataset_is_constructed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression this phase exists to prevent.

    Dataset.from_list on a whole corpus is what overflows Arrow's 32-bit offsets
    during fingerprinting.
    """
    import datasets

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Dataset.from_list must not be called for local sources")

    monkeypatch.setattr(datasets.Dataset, "from_list", explode)

    paths = [write_jsonl(tmp_path / f"t{i}.jsonl", [otel_trace(f"t{i}")]) for i in range(5)]
    source = source_for(paths)
    assert len(source.list_records()) == 5
    assert source.load_record(2)["trace_id"] == "t2"


def test_source_retains_no_record_bodies_after_streaming(tmp_path: Path) -> None:
    """Streaming every record must leave nothing cached on the source.

    The locator index is the only thing that grows with corpus size; a body held
    anywhere on the source would reintroduce the memory the phase removes.
    """
    paths = [write_jsonl(tmp_path / f"t{i}.jsonl", [otel_trace(f"t{i}")]) for i in range(8)]
    source = source_for(paths)

    for i in range(len(source.records)):
        source.load_record(i)

    holds_spans = [
        name
        for name, value in vars(source).items()
        if "spans" in repr(value)[:4000] or (isinstance(value, dict) and "spans" in value)
    ]
    assert holds_spans == [], f"source attributes retained trace bodies: {holds_spans}"

    # The index itself must stay lightweight: locators only, no span payloads.
    for record in source.records:
        assert not hasattr(record, "spans")
        assert "spans" not in repr(record)


# --------------------------------------------------------------------------------------
# Validation and errors
# --------------------------------------------------------------------------------------


def test_skip_invalid_true_drops_the_bad_file(tmp_path: Path) -> None:
    (tmp_path / "good.json").write_text(json.dumps(otel_trace()), encoding="utf-8")
    (tmp_path / "bad.jsonl").write_text("{not json at all", encoding="utf-8")
    files = sorted(tmp_path.glob("*.json*"))

    assert len(source_for(files, skip_invalid=True).list_records()) == 1


def test_skip_invalid_false_raises_naming_the_file(tmp_path: Path) -> None:
    (tmp_path / "good.json").write_text(json.dumps(otel_trace()), encoding="utf-8")
    (tmp_path / "bad.jsonl").write_text("{not json at all", encoding="utf-8")
    files = sorted(tmp_path.glob("*.json*"))

    with pytest.raises(ValueError, match=r"bad\.jsonl"):
        source_for(files, skip_invalid=False)


def test_empty_jsonl_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        source_for([tmp_path / "empty.jsonl"], skip_invalid=False)


def test_wire_file_with_no_convertible_records_raises_on_load_when_lazy(tmp_path: Path) -> None:
    """Startup validation catches this earlier; deferred, it still raises on load."""
    record = chat_record()
    record["request"] = "{not valid json"
    path = write_jsonl(tmp_path / "calls.jsonl", [record])
    source = LocalTraceSource([path], False, _normalize_file_trace, validate_at_startup=False)

    with pytest.raises(InvalidTraceError, match="no convertible spans"):
        source.load_record(0)


def test_validate_record_schema_requires_spans() -> None:
    validate_record_schema({"spans": []}, "x")

    with pytest.raises(InvalidTraceError, match="missing the required 'spans'"):
        validate_record_schema({"trace_id": "t"}, "x")
    with pytest.raises(InvalidTraceError, match="'spans' must be a list"):
        validate_record_schema({"spans": "nope"}, "x")
    # A top-level JSON array reaches this check as a list at runtime, which is
    # exactly what it exists to reject.
    not_a_mapping = cast(Dict[str, Any], ["not", "a", "dict"])
    with pytest.raises(InvalidTraceError, match="expected a trace object"):
        validate_record_schema(not_a_mapping, "x")


# --------------------------------------------------------------------------------------
# Filtering support
# --------------------------------------------------------------------------------------


def test_restrict_to_keeps_only_selected_records(tmp_path: Path) -> None:
    paths = [write_jsonl(tmp_path / f"t{i}.jsonl", [otel_trace(f"t{i}")]) for i in range(5)]
    source = source_for(paths)

    source.restrict_to([1, 3])

    assert [r.session_id_suffix for r in source.list_records()] == ["t1", "t3"]
    assert source.load_record(0)["trace_id"] == "t1"
    assert source.load_record(1)["trace_id"] == "t3"


def test_records_expose_format_per_input_shape(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps(otel_trace()), encoding="utf-8")
    write_jsonl(tmp_path / "b.jsonl", [otel_trace()])
    write_jsonl(tmp_path / "c.jsonl", [chat_record()])

    formats = {r.path.name: r.trace_format for r in source_for(sorted(tmp_path.glob("*.json*"))).records}

    assert formats["a.json"] is LocalTraceFormat.OTEL_JSON
    assert formats["b.jsonl"] is LocalTraceFormat.OTEL_JSONL
    assert formats["c.jsonl"] is LocalTraceFormat.WIRE


# --------------------------------------------------------------------------------------
# Reload determinism
#
# Lazy loading converts on every access, and a session's graph is built twice --
# once in the parent to enumerate its events, once in the assigned worker. Event
# ids embed the span id, so a span id that changes between conversions makes those
# two graphs disagree and every event-id lookup miss, silently dropping per-event
# reporting (TFUT among it). The eager path was immune only because it converted
# once into a Dataset and re-read the same rows.
# --------------------------------------------------------------------------------------


def test_wire_span_ids_are_stable_across_reloads(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "calls.jsonl", [chat_record() for _ in range(4)])
    source = source_for([path])

    first = [span["span_id"] for span in source.load_record(0)["spans"]]
    second = [span["span_id"] for span in source.load_record(0)["spans"]]

    assert first == second, "span ids must not change between conversions of one record"
    assert len(set(first)) == len(first), "span ids must stay unique within a trace"


def test_wire_span_ids_differ_across_files(tmp_path: Path) -> None:
    """Distinct captures must not collide, even when identically named."""
    paths = []
    for name in ("s1", "s2"):
        (tmp_path / name).mkdir()
        paths.append(write_jsonl(tmp_path / name / "calls.jsonl", [chat_record() for _ in range(3)]))

    source = source_for(paths)
    a = {span["span_id"] for span in source.load_record(0)["spans"]}
    b = {span["span_id"] for span in source.load_record(1)["spans"]}

    assert a and b
    assert not (a & b), "span ids collided across separate capture files"


def test_startup_validation_catches_a_bad_line_mid_jsonl(tmp_path: Path) -> None:
    """A malformed record deep in a file must not wait for dispatch."""
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps(otel_trace("a")) + "\n" + "{not json\n" + json.dumps(otel_trace("c")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"traces\.jsonl"):
        source_for([path], skip_invalid=False)


def test_startup_validation_catches_a_jsonl_record_missing_spans(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "traces.jsonl", [otel_trace("a"), {"trace_id": "no-spans"}])

    with pytest.raises(ValueError, match="spans"):
        source_for([path], skip_invalid=False)


def test_startup_validation_reads_past_the_first_wire_span(tmp_path: Path) -> None:
    """A bad line after a good one must fail at startup, not at dispatch."""
    path = tmp_path / "calls.jsonl"
    path.write_text(
        json.dumps(chat_record()) + "\n" + json.dumps(chat_record()) + "\n" + "{not valid json\n",
        encoding="utf-8",
    )

    with pytest.raises(InvalidTraceError, match=r"calls\.jsonl:3"):
        source_for([path], skip_invalid=False)


def test_startup_validation_accepts_unconvertible_records_among_valid_ones(tmp_path: Path) -> None:
    """iter_wire_spans skips span-less records, so validation must not reject them."""
    unconvertible = {"path": "/v1/chat/completions", "request": "{}", "response": ""}
    path = write_jsonl(tmp_path / "calls.jsonl", [chat_record(), unconvertible, chat_record()])

    assert len(source_for([path], skip_invalid=False).list_records()) == 1


def test_jsonl_error_reports_the_physical_line_number(tmp_path: Path) -> None:
    """Blank lines must not shift the reported line, or it points at the wrong row."""
    path = tmp_path / "traces.jsonl"
    path.write_text(
        "\n" + json.dumps(otel_trace("a")) + "\n\n" + json.dumps(otel_trace("b")) + "\n\n" + "{not json\n",
        encoding="utf-8",
    )

    with pytest.raises(InvalidTraceError, match=r"traces\.jsonl:6"):
        source_for([path], skip_invalid=False)


def test_startup_validation_catches_an_unconvertible_wire_capture(tmp_path: Path) -> None:
    record = chat_record()
    record["request"] = "{not valid json"
    path = write_jsonl(tmp_path / "calls.jsonl", [record])

    with pytest.raises(ValueError, match="no convertible spans"):
        source_for([path], skip_invalid=False)


def test_startup_validation_honors_skip_invalid(tmp_path: Path) -> None:
    (tmp_path / "good.json").write_text(json.dumps(otel_trace()), encoding="utf-8")
    write_jsonl(tmp_path / "bad.jsonl", [otel_trace("a"), {"trace_id": "no-spans"}])

    source = source_for(sorted(tmp_path.glob("*.json*")), skip_invalid=True)

    assert len(source.list_records()) == 1


def test_lazy_validation_defers_the_same_failure_to_load(tmp_path: Path) -> None:
    """validate_at_startup=False trades fail-fast for a cheaper startup."""
    path = write_jsonl(tmp_path / "traces.jsonl", [otel_trace("a"), {"trace_id": "no-spans"}])

    source = LocalTraceSource([path], False, _normalize_file_trace, validate_at_startup=False)
    assert len(source.list_records()) == 2

    assert source.load_record(0)["trace_id"] == "a"
    with pytest.raises(InvalidTraceError, match="spans"):
        source.load_record(1)


def test_startup_validation_does_not_retain_record_bodies(tmp_path: Path) -> None:
    """The pass must stay bounded: decode, check, drop."""
    write_jsonl(tmp_path / "traces.jsonl", [otel_trace(f"t{i}") for i in range(50)])
    source = source_for([tmp_path / "traces.jsonl"])

    assert len(source.list_records()) == 50
    for record in source.records:
        assert "spans" not in repr(record)


def test_wire_event_ids_are_stable_across_reloads(tmp_path: Path) -> None:
    """The property that actually matters: parent and worker must agree on event ids."""
    from inference_perf.datagen.replay.otel_trace_to_replay_graph import (
        build_graph,
        build_raw_calls,
        tag_user_facing_events,
    )

    path = write_jsonl(tmp_path / "calls.jsonl", [chat_record() for _ in range(3)])
    source = source_for([path])

    def build() -> tuple[set[str], set[str]]:
        record = source.load_record(0)
        calls, _ = build_raw_calls(record["spans"], include_errors=True)
        graph = build_graph(calls, source_file="x")
        tag_user_facing_events(graph, all_spans=record["spans"])
        return set(graph.events), {eid for eid, e in graph.events.items() if e.is_user_facing}

    parent_events, parent_user_facing = build()
    worker_events, worker_user_facing = build()

    assert parent_events == worker_events, "event ids diverged between builds"
    assert parent_user_facing == worker_user_facing, "user-facing event ids diverged between builds"
