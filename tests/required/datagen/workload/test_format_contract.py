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

import json
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Sequence

import pytest

from inference_perf.workload import (
    Delivery,
    Message,
    FilterExpressionError,
    Record,
    Row,
    Schedule,
    Selection,
    Session,
    TextPart,
    TextSpec,
    TraceFormat,
    TraceLocation,
    format_class,
    in_seeded_order,
    read_sessions,
    register,
    sessions_of,
    with_waits,
)
from inference_perf.workload import format as format_module


# A format for these tests. A row is {"s": session, "n": tokens, "t": send
# time in ms}: each row is one request of `n` tokens asking for 1 token,
# sent at `t` and waiting 100 ms for the row before it in its session. The
# filter sees a session's row count and its largest `n`.
class UnitFormat(TraceFormat):
    name = "unit"
    session_key: ClassVar[Optional[str]] = "s"
    default_block_size = 8

    def check(self, row: Row) -> None:
        if not isinstance(row.get("n"), int):
            raise ValueError("n must be an integer")

    def summary(self, key: str, rows: Sequence[Row]) -> Dict[str, object]:
        return {"session_id": key, "rows": len(rows), "largest": max(int(str(row["n"])) for row in rows)}

    def session(self, key: str, rows: Sequence[Row]) -> Session:
        records: List[Record] = []
        schedules: Dict[str, Schedule] = {}
        for index, row in enumerate(rows):
            schedules[f"{key}:{index}"] = Schedule(
                start_ms=int(str(row["t"])),
                wait_ms=100 if records else 0,
                after=[records[-1].id] if records else [],
            )
            records.append(
                Record(
                    id=f"{key}:{index}",
                    messages=[
                        Message(
                            role="user",
                            parts=[TextPart(spec=TextSpec(num_tokens=int(str(row["n"])), block_size=self.block_size))],
                        )
                    ],
                    output_tokens=1,
                )
            )
        return Session(id=key, records=records, delivery=Delivery(schedules=schedules))


# The same rows read with every row as a session of its own.
class RowFormat(UnitFormat):
    name = "unit-rows"
    session_key = None


def write_rows(path: Path, rows: Sequence[object]) -> Path:
    path.write_text("\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows) + "\n")
    return path


# Sessions a (two rows), b (one row) and c (two rows), sent from 5000 ms on.
ROWS = [
    {"s": "a", "n": 10, "t": 5000},
    {"s": "a", "n": 20, "t": 5400},
    {"s": "b", "n": 90, "t": 6000},
    {"s": "c", "n": 30, "t": 7000},
    {"s": "c", "n": 40, "t": 7500},
]


def session_ids(sessions: Sequence[Session]) -> List[str]:
    return [session.id for session in sessions]


# A format registered as "unit" is found by that name, and any format by the
# module path of its class, registered or not. A name nobody registered fails
# listing the known formats, and so does a module path that names something
# other than a format.
def test_a_format_is_picked_by_name_or_import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(format_module._REGISTERED, "unit", raising=False)
    assert format_class(f"{UnitFormat.__module__}:UnitFormat").name == "unit"
    with pytest.raises(ValueError, match="Unknown trace format 'unit'"):
        format_class("unit")
    assert register(UnitFormat) is UnitFormat
    assert format_class("unit") is UnitFormat
    with pytest.raises(ValueError, match="known formats: .*unit"):
        format_class("NoSuchFormat")
    with pytest.raises(ValueError, match="is not a TraceFormat"):
        format_class("inference_perf.workload.record:Record")


# The format's block size is 8 unless one is given: a 10-token row is built
# with 8-token blocks by default and 32-token blocks when asked.
def test_block_size_defaults_to_the_formats(tmp_path: Path) -> None:
    trace = write_rows(tmp_path / "t.jsonl", ROWS[:1])
    for block_size, expected in ((None, 8), (32, 32)):
        (session,) = read_sessions(UnitFormat(block_size=block_size), TraceLocation(files=[str(trace)]))
        part = session.records[0].messages[0].parts[0]
        assert isinstance(part, TextPart) and part.spec is not None and part.spec.block_size == expected


# Rows group into sessions by the format's session field, in the order
# read: a, b, c with 2, 1 and 2 requests. The earliest recorded start
# (5000 ms) moves to zero and the rest keep their distance from it.
def test_rows_group_into_sessions_starting_at_zero(tmp_path: Path) -> None:
    trace = write_rows(tmp_path / "t.jsonl", ROWS)
    sessions = read_sessions(UnitFormat(), TraceLocation(files=[str(trace)]))
    assert session_ids(sessions) == ["a", "b", "c"]
    assert [len(session.records) for session in sessions] == [2, 1, 2]
    starts = [session.delivery.of(record.id).start_ms for session in sessions for record in session.records]
    assert starts == [0, 400, 1000, 2000, 2500]


# A caller that reads its own data hands over rows instead of a place: the
# same rows give the same sessions, and a bad row is named by its position.
def test_rows_in_hand_give_the_same_sessions(tmp_path: Path) -> None:
    from_file = read_sessions(UnitFormat(), TraceLocation(files=[str(write_rows(tmp_path / "t.jsonl", ROWS))]))
    assert sessions_of(UnitFormat(), iter(ROWS)) == from_file
    with pytest.raises(ValueError, match="Invalid unit row at dataset:2: n must be an integer"):
        sessions_of(UnitFormat(), [ROWS[0], {"s": "a", "n": "many", "t": 1}], source="dataset")


# With no session field every row is a session of one, named by its place
# in the trace.
def test_rows_without_a_session_field_stand_alone(tmp_path: Path) -> None:
    trace = write_rows(tmp_path / "t.jsonl", ROWS)
    sessions = read_sessions(RowFormat(), TraceLocation(files=[str(trace)]))
    assert session_ids(sessions) == ["0", "1", "2", "3", "4"]
    assert all(len(session.records) == 1 for session in sessions)


# A session whose rows are split by another session is an error naming the
# file and line where it comes back, not a silent merge.
def test_a_session_split_by_another_fails(tmp_path: Path) -> None:
    trace = write_rows(tmp_path / "t.jsonl", [ROWS[0], ROWS[2], ROWS[1]])
    with pytest.raises(ValueError, match=r"'a' reappears at .*t\.jsonl:3"):
        read_sessions(UnitFormat(), TraceLocation(files=[str(trace)]))


# The filter sees the format's summary: keeping sessions with more than one
# row leaves a and c. A filter that reads a field the summary lacks fails as
# a filter error, and one that is not a lambda fails before any row is read.
def test_filter_keeps_sessions_by_their_summary(tmp_path: Path) -> None:
    where = TraceLocation(files=[str(write_rows(tmp_path / "t.jsonl", ROWS))])
    kept = read_sessions(UnitFormat(), where, Selection(filter="lambda x: x['rows'] > 1"))
    assert session_ids(kept) == ["a", "c"]
    with pytest.raises(FilterExpressionError, match="session 'a'"):
        read_sessions(UnitFormat(), where, Selection(filter="lambda x: x['missing'] > 1"))
    with pytest.raises(ValueError, match="not callable"):
        read_sessions(UnitFormat(), where, Selection(filter="3"))


# The cap counts kept sessions and stops the read: with a cap of 1 and a
# filter that rejects a, only b is kept, and the second file, which is not
# even JSON, is never opened. Without the cap that file fails the load.
def test_cap_counts_kept_sessions_and_stops_reading(tmp_path: Path) -> None:
    first = write_rows(tmp_path / "1.jsonl", ROWS)
    broken = write_rows(tmp_path / "2.jsonl", ["{not json"])
    where = TraceLocation(files=[str(first), str(broken)])
    kept = read_sessions(UnitFormat(), where, Selection(filter="lambda x: x['largest'] > 50", max_sessions=1))
    assert session_ids(kept) == ["b"]
    with pytest.raises(ValueError, match=r"2\.jsonl:1: not JSON"):
        read_sessions(UnitFormat(), where)


# A row the format rejects fails the load with its file and line, or is
# skipped when asked: the bad second row of session a is dropped and a keeps
# one request. A trace with nothing usable still fails.
def test_invalid_rows_fail_or_are_skipped(tmp_path: Path) -> None:
    rows = [ROWS[0], {"s": "a", "n": "many", "t": 1}, ROWS[2]]
    where = TraceLocation(files=[str(write_rows(tmp_path / "t.jsonl", rows))])
    with pytest.raises(ValueError, match=r"Invalid unit row at .*t\.jsonl:2: n must be an integer"):
        read_sessions(UnitFormat(), where)
    kept = read_sessions(UnitFormat(), where, Selection(skip_invalid=True))
    assert [(session.id, len(session.records)) for session in kept] == [("a", 1), ("b", 1)]
    only_bad = TraceLocation(files=[str(write_rows(tmp_path / "bad.jsonl", rows[1:2]))])
    with pytest.raises(ValueError, match="No valid unit sessions"):
        read_sessions(UnitFormat(), only_bad, Selection(skip_invalid=True))


# A directory is read as its trace files in name order, whatever else is in
# it: b.jsonl holds session b and a.jsonl session a, so a comes first and the
# notes file is ignored. A directory with no trace file, a missing directory
# and a missing file each fail by name.
def test_a_directory_is_its_trace_files(tmp_path: Path) -> None:
    write_rows(tmp_path / "b.jsonl", ROWS[2:3])
    write_rows(tmp_path / "a.jsonl", ROWS[:2])
    (tmp_path / "notes.txt").write_text("not a trace")
    assert session_ids(read_sessions(UnitFormat(), TraceLocation(directory=str(tmp_path)))) == ["a", "b"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No unit trace files"):
        read_sessions(UnitFormat(), TraceLocation(directory=str(empty)))
    with pytest.raises(ValueError, match="not a directory"):
        read_sessions(UnitFormat(), TraceLocation(directory=str(tmp_path / "nowhere")))
    with pytest.raises(ValueError, match="Trace file does not exist"):
        read_sessions(UnitFormat(), TraceLocation(files=[str(tmp_path / "nowhere.jsonl")]))


# A trace is read from exactly one place: none, or a directory and files
# together, is refused.
def test_a_trace_has_exactly_one_location(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        TraceLocation()
    with pytest.raises(ValueError, match="exactly one"):
        TraceLocation(directory=str(tmp_path), files=["t.jsonl"])


# Recorded waits are shaped by a function of the wait: capping at 60 ms
# turns each 100 ms wait into 60 and leaves the zero waits alone. Sessions
# run in a shuffle of their sorted ids that depends only on the seed.
def test_waits_are_shaped_and_sessions_ordered_by_seed(tmp_path: Path) -> None:
    sessions = read_sessions(UnitFormat(), TraceLocation(files=[str(write_rows(tmp_path / "t.jsonl", ROWS))]))
    capped = with_waits(sessions, lambda wait_ms: min(wait_ms, 60))
    assert [[s.delivery.of(r.id).wait_ms for r in s.records] for s in capped] == [[0, 60], [0], [0, 60]]
    assert [[s.delivery.of(r.id).wait_ms for r in s.records] for s in sessions] == [[0, 100], [0], [0, 100]]
    # shaping the waits changes the delivery and leaves the records alone
    assert [s.records for s in capped] == [s.records for s in sessions]
    assert session_ids(in_seeded_order(sessions, 1)) == session_ids(in_seeded_order(list(reversed(sessions)), 1))
    assert session_ids(in_seeded_order(sessions, 1)) == ["b", "c", "a"]
