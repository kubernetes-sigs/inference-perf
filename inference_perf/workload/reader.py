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
"""From a place and a format to the sessions a run replays.

Everything here is the same for every trace format: where rows come from (a
directory, a list of files or a Hugging Face dataset), how rows group into
sessions, which sessions are kept and how many, and the shaping a replay
applies to recorded waits. The format is asked only what a kept session's
rows mean.

Rows are streamed and reading stops once enough sessions are kept, so a small
run never reads a large trace to the end. That is why a session's rows have
to sit together in the source.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

from inference_perf.workload.format import Row, TraceFormat
from inference_perf.workload.record import Delivery, Message, Record, Schedule, Session

logger = logging.getLogger(__name__)

# A repo id, or {"path": repo id, ...} whose other keys go to load_dataset.
HFDatasetPath = Union[str, Dict[str, object]]

_SourcedRow = Tuple[str, int, Row]


class FilterExpressionError(ValueError):
    """Raised when a session filter fails to evaluate."""


@dataclass(frozen=True)
class TraceLocation:
    """Where a trace's rows are. Exactly one of the three is set."""

    directory: Optional[str] = None
    files: Optional[Sequence[str]] = None
    hf_dataset: Optional[HFDatasetPath] = None

    def __post_init__(self) -> None:
        given = sum(place is not None for place in (self.directory, self.files, self.hf_dataset))
        if given != 1:
            raise ValueError("a trace is read from exactly one of a directory, a list of files or a Hugging Face dataset")


@dataclass(frozen=True)
class Selection:
    """Which sessions are kept. `filter` is a lambda over a format's session
    summary, `max_sessions` stops the read once that many are kept, and with
    `skip_invalid` a bad row or file is logged and skipped instead of failing
    the load."""

    filter: Optional[str] = None
    max_sessions: Optional[int] = None
    skip_invalid: bool = False


def read_sessions(fmt: TraceFormat, where: TraceLocation, select: Optional[Selection] = None) -> List[Session]:
    """The kept sessions of a trace in the order they were read, with the
    earliest recorded start moved to zero."""
    select = select or Selection()
    kept = _select(fmt, _blocks(fmt, _rows(fmt, where, select.skip_invalid), select.skip_invalid), select)
    if where.hf_dataset is not None and fmt.side_tables:
        _join_side_tables(fmt, kept, where.hf_dataset, select.skip_invalid)
    return starting_at_zero([fmt.session(key, rows) for key, rows in kept.items()])


def sessions_of(
    fmt: TraceFormat, rows: Iterable[Row], select: Optional[Selection] = None, source: str = "row"
) -> List[Session]:
    """The same as `read_sessions`, for a caller that reads its own data and
    has the rows in hand. `source` names them in error messages."""
    select = select or Selection()
    sourced = ((source, index, row) for index, row in enumerate(rows, start=1))
    kept = _select(fmt, _blocks(fmt, sourced, select.skip_invalid), select)
    return starting_at_zero([fmt.session(key, block) for key, block in kept.items()])


def _rows(fmt: TraceFormat, where: TraceLocation, skip_invalid: bool) -> Iterator[_SourcedRow]:
    if where.hf_dataset is not None:
        return _hf_rows(where.hf_dataset, skip_invalid)
    return _local_rows(fmt, where, skip_invalid)


def _local_rows(fmt: TraceFormat, where: TraceLocation, skip_invalid: bool) -> Iterator[_SourcedRow]:
    if where.directory is not None:
        trace_dir = Path(where.directory)
        if not trace_dir.is_dir():
            raise ValueError(f"Trace directory does not exist or is not a directory: {trace_dir}")
        # endswith, so a dotted stem (trace_v0.0.2.jsonl) is still recognised
        files = sorted(p for p in trace_dir.iterdir() if p.is_file() and p.name.endswith(fmt.file_suffixes))
        if not files:
            raise ValueError(f"No {fmt.name} trace files found in {trace_dir}")
    else:
        files = [Path(name) for name in where.files or []]
        for path in files:
            if not path.is_file():
                raise ValueError(f"Trace file does not exist: {path}")

    for path in files:
        try:
            for line_no, row in fmt.read(path):
                yield str(path), line_no, row
        except Exception as e:
            logger.error(f"Failed to load trace file {path.name}: {e}")
            if not skip_invalid:
                raise


def _blocks(fmt: TraceFormat, rows: Iterator[_SourcedRow], skip_invalid: bool) -> Iterator[Tuple[str, List[Row], int]]:
    """Group rows into sessions as (key, rows, invalid rows skipped).

    A session is yielded as soon as the next one starts, which is what lets
    the caller stop early. So a session that comes back after another started
    is an error, not a silent merge with the rows already handed over.
    """
    current: Optional[str] = None
    block: List[Row] = []
    invalid = 0
    seen: Set[str] = set()
    index = 0
    for source, line_no, row in rows:
        try:
            fmt.check(row)
            if fmt.session_key is not None and fmt.session_key not in row:
                raise ValueError(f"no {fmt.session_key!r} field")
        except Exception as e:
            logger.error(f"Failed to validate {fmt.name} row {source}:{line_no}: {e}")
            if not skip_invalid:
                raise ValueError(f"Invalid {fmt.name} row at {source}:{line_no}: {e}") from e
            invalid += 1
            continue

        if fmt.session_key is None:
            yield str(index), [row], invalid
            index, invalid = index + 1, 0
            continue

        key = str(row[fmt.session_key])
        if key != current:
            if current is not None:
                yield current, block, invalid
                invalid = 0
            if key in seen:
                raise ValueError(
                    f"{fmt.name} session {key!r} reappears at {source}:{line_no} after another session. "
                    "Every row of a session has to sit together; sort the trace by session, or use one file per session."
                )
            seen.add(key)
            current, block = key, []
        block.append(row)
    if current is not None:
        yield current, block, invalid


def _compile_filter(expression: Optional[str]) -> Optional[Callable[..., object]]:
    """The filter as a callable. It is evaluated with eval(), so it is trusted
    input: an operator's own config, never anything from a user."""
    if not expression:
        return None
    try:
        compiled = eval(expression, {"__builtins__": {}}, {})
    except Exception as e:
        raise ValueError(f"Invalid filter expression {expression!r}: {e}") from e
    if not callable(compiled):
        raise ValueError(f"Filter expression {expression!r} is not callable")
    keep: Callable[..., object] = compiled
    return keep


def _select(fmt: TraceFormat, blocks: Iterator[Tuple[str, List[Row], int]], select: Selection) -> Dict[str, List[Row]]:
    keep = _compile_filter(select.filter)
    kept: Dict[str, List[Row]] = {}
    filtered = invalid = 0
    for key, rows, skipped in blocks:
        invalid += skipped
        if keep is not None:
            summary = fmt.summary(key, rows)
            try:
                wanted = bool(keep(summary))
            except Exception as e:
                raise FilterExpressionError(f"Filter expression failed on session {key!r}: {e!r}") from e
            if not wanted:
                filtered += 1
                continue
        kept[key] = rows
        if select.max_sessions is not None and len(kept) >= select.max_sessions:
            logger.info(f"Stopped reading {fmt.name} input after {len(kept)} sessions (the session cap)")
            break
    if keep is not None:
        logger.info(f"Filter applied: {len(kept)} sessions kept, {filtered} rejected")
    if invalid:
        logger.warning(f"Skipped {invalid} invalid rows")
    if not kept:
        raise ValueError(f"No valid {fmt.name} sessions found")
    logger.info(f"Loaded {len(kept)} {fmt.name} sessions ({sum(len(rows) for rows in kept.values())} rows)")
    return kept


def _hf_table(hf_dataset: HFDatasetPath, name: Optional[str] = None) -> Tuple[str, Iterable[object]]:
    """One table of a Hugging Face dataset, streamed unless the dataset path
    says `streaming: False`. Streaming is what keeps a small run from
    downloading and building a table of millions of rows first."""
    from datasets import load_dataset

    if isinstance(hf_dataset, dict):
        kwargs = {k: v for k, v in hf_dataset.items() if k != "path"}
        path = str(hf_dataset["path"])
    else:
        kwargs, path = {}, hf_dataset
    streaming = kwargs.pop("streaming", True)
    if name is not None:
        kwargs["name"] = name
    table: object = load_dataset(path, streaming=streaming, **kwargs)
    # a load that named no split gives a mapping of splits, not a table
    if isinstance(table, Mapping) and table:
        splits = dict(table)
        split = str(kwargs.get("split", "train"))
        table = splits[split] if split in splits else splits[next(iter(splits))]
    if not isinstance(table, Iterable):
        raise ValueError(f"Expected a dataset or a mapping of splits, got {type(table).__name__}")
    return path, table


def _hf_rows(hf_dataset: HFDatasetPath, skip_invalid: bool) -> Iterator[_SourcedRow]:
    try:
        path, table = _hf_table(hf_dataset)
    except Exception as e:
        raise ValueError(f"Failed to load Hugging Face dataset {hf_dataset!r}: {e}") from e
    logger.info(f"Reading trace rows from Hugging Face dataset: {path}")
    for index, row in enumerate(table, start=1):
        if not isinstance(row, dict):
            if not skip_invalid:
                raise ValueError(f"Unexpected row type {type(row)} in Hugging Face dataset {path}")
            continue
        yield f"hf:{path}", index, row


def _join_side_tables(fmt: TraceFormat, kept: Dict[str, List[Row]], hf_dataset: HFDatasetPath, skip_invalid: bool) -> None:
    """Fill the fields a format keeps in other tables, for the kept rows only.

    The side tables are read after the selection so only rows that will be
    replayed are indexed, not the whole dataset.
    """
    if fmt.join_key is None:
        raise ValueError(f"{fmt.name} names side tables but no join_key")
    joined: Dict[str, Dict[str, object]] = {}
    for key, rows in kept.items():
        copies: List[Row] = []
        for row in rows:
            copy = dict(row)
            copies.append(copy)
            if copy.get(fmt.join_key) is not None:
                joined[str(copy[fmt.join_key])] = copy
        kept[key] = copies
    if not joined:
        return

    for field, table_name in fmt.side_tables.items():
        try:
            path, table = _hf_table(hf_dataset, name=table_name)
        except Exception as e:
            message = f"Could not load '{table_name}' table of {hf_dataset!r}: {e}. Rows will replay without their {field}."
            if not skip_invalid:
                raise ValueError(message) from e
            logger.warning(message)
            continue
        found: Dict[str, List[object]] = {}
        for side_row in table:
            if isinstance(side_row, dict) and side_row.get(fmt.join_key) is not None:
                owner = str(side_row[fmt.join_key])
                if owner in joined:
                    found.setdefault(owner, []).append(side_row)
        for owner, side_rows in found.items():
            joined[owner][field] = side_rows


def starting_at_zero(sessions: Sequence[Session]) -> List[Session]:
    """The same sessions with every recorded start shifted so the earliest is 0."""
    starts = [
        schedule.start_ms for s in sessions for schedule in s.delivery.schedules.values() if schedule.start_ms is not None
    ]
    origin = min(starts, default=0)
    if origin == 0:
        return list(sessions)

    def shifted(schedule: Schedule) -> Schedule:
        if schedule.start_ms is None:
            return schedule
        return schedule.model_copy(update={"start_ms": schedule.start_ms - origin})

    return [_redelivered(session, shifted) for session in sessions]


def with_waits(sessions: Sequence[Session], shape: Callable[[int], int]) -> List[Session]:
    """The same sessions with `shape` applied to every recorded wait: a cap,
    a scale, or zero to run a session back to back."""

    def shaped(schedule: Schedule) -> Schedule:
        return schedule.model_copy(update={"wait_ms": shape(schedule.wait_ms)})

    return [_redelivered(session, shaped) for session in sessions]


def _redelivered(session: Session, change: Callable[[Schedule], Schedule]) -> Session:
    """The same records under a delivery with `change` applied to every schedule."""
    schedules = {record_id: change(schedule) for record_id, schedule in session.delivery.schedules.items()}
    return session.model_copy(update={"delivery": Delivery(schedules=schedules)})


def continues(session: Session, record: Record) -> Optional[Record]:
    """The record of `session` whose whole conversation opens `record`'s, if any.

    `record` continues another when it waited for it, directly or through
    other records, and its messages start with that record's messages followed
    by that record's reply. A runtime that keeps live history then sends only
    the messages after them, with the live reply where the recorded one was.
    The longest such conversation wins.
    """
    by_id = {other.id: other for other in session.records}
    waited_for: List[str] = []
    pending = list(session.delivery.of(record.id).after)
    while pending:
        other_id = pending.pop()
        if other_id not in waited_for:
            waited_for.append(other_id)
            pending.extend(session.delivery.of(other_id).after)
    found: Optional[Record] = None
    found_length = 0
    for other_id in waited_for:
        other = by_id[other_id]
        conversation = [*other.messages, Message(role="assistant", parts=other.reply)]
        if len(conversation) > len(record.messages) or len(conversation) <= found_length:
            continue
        if record.messages[: len(conversation)] == conversation:
            found, found_length = other, len(conversation)
    return found


def in_seeded_order(sessions: Sequence[Session], seed: int) -> List[Session]:
    """The sessions in a shuffle of their sorted ids. The RNG is local, so
    ordering a trace never touches the process-global random state."""
    ordered = sorted(sessions, key=lambda session: session.id)
    random.Random(seed).shuffle(ordered)
    return ordered
