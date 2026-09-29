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

"""Tests for the TraceSource abstraction (Phase 1).

Phase 1 is a structural refactor: HF replay must behave exactly as it did when
the generator held a Dataset directly. These tests pin the parts that are easy
to break silently -- the session-ID fallbacks and one-row-at-a-time access.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from datasets import Dataset

from inference_perf.datagen.replay.trace_source import (
    HFDatasetTraceSource,
    TraceReadError,
    TraceRecordMetadata,
)


def _span(model: str = "m", in_tok: int = 1, out_tok: int = 2) -> Dict[str, Any]:
    return {
        "attributes": {
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": in_tok,
            "gen_ai.usage.output_tokens": out_tok,
        }
    }


def _rows(n: int, with_ids: bool = True) -> List[Dict[str, Any]]:
    rows = []
    for i in range(n):
        row: Dict[str, Any] = {"spans": [_span()]}
        if with_ids:
            row["session_id"] = f"sess{i}"
            row["source_id"] = f"/traces/{i}.json"
        rows.append(row)
    return rows


def test_list_records_returns_one_entry_per_row() -> None:
    source = HFDatasetTraceSource(Dataset.from_list(_rows(3)))
    records = source.list_records()

    assert len(records) == 3
    assert records[0] == TraceRecordMetadata(session_id_suffix="sess0", source_id="/traces/0.json")


def test_list_records_leaves_missing_ids_empty_for_generator_fallback() -> None:
    """A source must not invent IDs.

    Both fallbacks (`session_{slot}` and source_id -> session_id) depend on the
    generator's post-shuffle slot, which the source cannot see. Defaulting here
    would silently change session IDs for corpora whose rows carry no session_id.
    """
    source = HFDatasetTraceSource(Dataset.from_list(_rows(2, with_ids=False)))
    records = source.list_records()

    assert [r.session_id_suffix for r in records] == ["", ""]
    assert [r.source_id for r in records] == ["", ""]


def test_load_record_returns_a_plain_dict_with_spans() -> None:
    source = HFDatasetTraceSource(Dataset.from_list(_rows(2)))
    record = source.load_record(1)

    assert isinstance(record, dict)
    assert record["session_id"] == "sess1"
    assert len(record["spans"]) == 1


def test_load_record_reads_only_the_requested_row() -> None:
    """The point of the abstraction: one row per call, not the whole corpus."""
    reads: List[int] = []
    dataset = Dataset.from_list(_rows(5))
    original_getitem = Dataset.__getitem__

    def spy(self: Dataset, key: Any) -> Any:
        if isinstance(key, int):
            reads.append(key)
        return original_getitem(self, key)

    Dataset.__getitem__ = spy
    try:
        source = HFDatasetTraceSource(dataset)
        source.list_records()
        assert reads == [], "list_records must not read row bodies"

        source.load_record(3)
        assert reads == [3]
    finally:
        Dataset.__getitem__ = original_getitem


def test_load_record_wraps_failures_in_trace_read_error() -> None:
    class Broken:
        def __len__(self) -> int:
            return 1

        @property
        def column_names(self) -> List[str]:
            return []

        def __getitem__(self, key: Any) -> Any:
            raise RuntimeError("arrow exploded")

    source = HFDatasetTraceSource(Broken())
    with pytest.raises(TraceReadError, match="arrow exploded"):
        source.load_record(0)


def test_out_of_range_index_still_raises_index_error() -> None:
    """IndexError is a programming error, not a bad trace -- keep it distinct."""
    source = HFDatasetTraceSource(Dataset.from_list(_rows(2)))
    with pytest.raises(IndexError):
        source.load_record(99)


def test_dataset_property_exposes_wrapped_dataset_for_filtering() -> None:
    dataset = Dataset.from_list(_rows(2))
    assert HFDatasetTraceSource(dataset).dataset is dataset
