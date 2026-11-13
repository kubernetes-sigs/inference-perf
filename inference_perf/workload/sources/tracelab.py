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
"""TraceLab coding-agent traces (github.com/uw-syfi/TraceLab, CC BY 4.0).

One row per LLM round, read from JSONL, a JSON array, either of them gzipped,
or the Hugging Face tables. The fields read here:

    session_id           the session the round belongs to
    round_index          the round's order within its session
    input_tokens_total   the whole prompt, = prefix_tokens + newly_append_tokens
    prefix_tokens        the leading part the engine served from cache
    output_tokens        generated tokens
    model                the model the round ran on
    timing_events[]      typed events with ISO-8601 timestamps
    tools[]              the round's tool calls: latency and, when recorded,
                         the wall-clock window each one ran in
    round_pk             join key of the Hugging Face tool_calls and
                         timing_events tables

No text and no content hashes are recorded. A round records its whole prompt,
not the messages that were added, so every round is its own record of its
session: a user turn whose synthetic part is that prompt and an assistant turn
that asks for the recorded output. Block ids are session-scoped and minted so
that the first `prefix_tokens` are the same blocks the previous round sent and
the rest are new. That reproduces the recorded prefix reuse to block
granularity, and when an agent compacts its context (prefix smaller than the
previous prompt) the later blocks are simply not reused. Sharing between
sessions cannot be reproduced: the corpus carries nothing to identify it.

Each round is one node depending on the previous round of its session. A round
is submitted when its last input event (`user_message`, `tool_result`) arrives
and completes at its last model-output event (`reasoning`, `text`,
`tool_call`), and the think time before a round is the gap between the
previous round's completion and its own submission. `usage_report` rows are
written after the tool ran, so they bound neither. A round without usable
timestamps waits for as long as the previous round's tools ran, with
overlapping tool windows merged. Subagent sessions are recorded as separate
sessions with no link to their parent and are replayed as such.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

from pydantic import BaseModel, Field

from inference_perf.workload.arrangement import Arrangement, Node
from inference_perf.workload.record import ContentPart, Record, SyntheticPart, TextPart, Turn
from inference_perf.workload.sources.base import Workload, WorkloadSource

logger = logging.getLogger(__name__)

# The corpus records no block structure, so the size is a choice; 64 matches
# what the engines it was captured against use.
TRACELAB_BLOCK_SIZE = 64

# Matched with str.endswith so a dotted stem (tracelab_v0.0.2.jsonl) is still
# recognised: joining Path.suffixes yields ".0.0.2.jsonl" for that name.
TRACE_FILE_SUFFIXES = (".jsonl", ".jsonl.gz", ".json", ".json.gz")

# TraceLab event types (see the dataset's README): the LLM request is submitted
# when the last input event arrives and finishes at the last model-output event.
INPUT_EVENT_TYPES = frozenset({"user_message", "tool_result"})
OUTPUT_EVENT_TYPES = frozenset({"reasoning", "text", "tool_call"})

DEFAULT_HF_DATASET = "UW-SyFI/TraceLab"

# Sent for a round that names no model.
DEFAULT_MODEL = "tracelab-model"

# A repo id, or {"path": repo id, ...} whose other keys go to load_dataset.
HFDatasetPath = Union[str, Dict[str, Any]]


class FilterExpressionError(ValueError):
    """Raised when a user-supplied filter expression fails to evaluate."""


class TraceLabToolCall(BaseModel):
    model_config = {"extra": "ignore"}

    tool_name: Optional[str] = None
    tool_wall_latency_ms: Optional[float] = None
    tool_internal_latency_ms: Optional[float] = None
    # Wall-clock window of the tool execution. Used to merge overlapping
    # intervals when parallel tools ran concurrently.
    emitted_at: Optional[str] = None
    result_at: Optional[str] = None


class TraceLabTimingEvent(BaseModel):
    model_config = {"extra": "ignore"}

    event_type: Optional[str] = None
    timestamp: Optional[str] = None


class TraceLabRound(BaseModel):
    """One TraceLab JSONL row (one LLM invocation). Extra fields ignored."""

    model_config = {"extra": "ignore", "populate_by_name": True}

    session_id: str
    round_id: Optional[str] = None
    round_pk: Optional[Union[int, str]] = None  # HF join key for tool_calls/timing_events
    round_index: int
    provider: Optional[str] = None
    model: Optional[str] = None
    input_tokens_total: int
    prefix_tokens: int = 0
    newly_append_tokens: int = 0
    output_tokens: int
    tools: List[TraceLabToolCall] = Field(default_factory=list)
    timing_events: List[TraceLabTimingEvent] = Field(default_factory=list)


def _parse_ts(ts: Optional[str]) -> Optional[float]:
    """Parse an ISO-8601 timestamp to epoch seconds, None when unparseable."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _round_time_bounds(rnd: TraceLabRound) -> Tuple[Optional[float], Optional[float]]:
    """Recorded (submission, completion) time of a round.

    The LLM request for a round is submitted when its last *input* event
    arrives: ``user_message`` on a fresh turn, or ``tool_result`` once the last
    of several parallel tools has returned. Falling back to ``min`` of the
    model-output events keeps rounds that recorded no input event usable.

    The round completes at its last *model-output* event. ``usage_report`` rows
    (Codex writes them after the tool call has finished) describe the response
    that was already produced, so including them would push ``prev_end`` past
    the tool execution and collapse the measured think time to zero.

    Returns ``(None, None)`` when the round carries no usable event.
    """
    events = [(e.event_type or "", _parse_ts(e.timestamp)) for e in rnd.timing_events or []]
    timed = [(event_type, stamp) for event_type, stamp in events if stamp is not None]
    inputs = [stamp for event_type, stamp in timed if event_type in INPUT_EVENT_TYPES]
    outputs = [stamp for event_type, stamp in timed if event_type in OUTPUT_EVENT_TYPES]
    if not inputs and not outputs:
        return (None, None)
    start = max(inputs) if inputs else min(outputs)
    end = max(outputs) if outputs else max(inputs)
    return (start, end)


def round_gap_ms(prev: TraceLabRound, curr: TraceLabRound) -> Optional[int]:
    """Recorded think time before ``curr``: its submission time minus the
    previous round's completion time, in milliseconds.

    Returns None when either round lacks usable ``timing_events``
    timestamps so the caller can fall back to tool latency.
    """
    _, prev_end = _round_time_bounds(prev)
    curr_start, _ = _round_time_bounds(curr)
    if prev_end is None or curr_start is None:
        return None
    return max(int((curr_start - prev_end) * 1000.0), 0)


def tool_latency_ms(rnd: TraceLabRound) -> int:
    """Wall time the round's tools occupied before the next round started.

    Parallel tools run concurrently, so overlapping ``[emitted_at,
    result_at]`` intervals are merged instead of summed: N concurrent 5s
    tools cost 5s, not 5×N. Tools that lack both timestamps fall back to
    their recorded latency, which is added on top of the merged total.
    """
    intervals: List[Tuple[float, float]] = []
    serial_ms = 0.0
    for t in rnd.tools or []:
        emitted = _parse_ts(t.emitted_at)
        result = _parse_ts(t.result_at)
        if emitted is not None and result is not None and result >= emitted:
            intervals.append((emitted, result))
            continue
        lat = t.tool_wall_latency_ms if t.tool_wall_latency_ms is not None else t.tool_internal_latency_ms
        if lat:
            serial_ms += max(float(lat), 0.0)

    merged_ms = 0.0
    if intervals:
        intervals.sort()
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged_ms += (cur_end - cur_start) * 1000.0
                cur_start, cur_end = start, end
        merged_ms += (cur_end - cur_start) * 1000.0
    return max(int(merged_ms + serial_ms), 0)


def _iter_jsonl_rows(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield TraceLab rows from .json / .jsonl / .jsonl.gz files lazily.

    Yields (line_number, row) tuples so validation errors can name the file
    and line. JSON files report line 0. Lazily so callers can stop reading
    once the requested number of sessions has been collected.
    """
    name = path.name
    opener: Callable[..., Any]
    if name.endswith(".gz"):
        opener = gzip.open
    else:
        opener = open

    with opener(path, "rt", encoding="utf-8") as f:
        first = f.read(1)
        if not first:
            return
        f.seek(0)
        # A .json.gz file is still a JSON array; Path.suffix is ".gz" for it, so
        # the extension test has to look at the whole name.
        if name.endswith((".json", ".json.gz")):
            blob = json.load(f)
            if isinstance(blob, list):
                for r in blob:
                    if isinstance(r, dict):
                        yield 0, r
            elif isinstance(blob, dict):
                yield 0, blob
            return
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_no}: not JSON: {e}") from e
                yield line_no, row


def _finalize_block(rounds: List[TraceLabRound]) -> List[TraceLabRound]:
    """Order a session's rounds and renumber them densely.

    Real traces carry occasional duplicate ``(session_id, round_index)``
    pairs and rows whose ``round_index`` disagrees with their timestamps.
    When every round has a submission timestamp, the timestamps decide
    (tie-breaking on round_index/round_id); otherwise round_index does.
    """
    if not rounds:
        return rounds
    stamped: List[Tuple[Optional[float], TraceLabRound]] = [(_round_time_bounds(r)[0], r) for r in rounds]

    def stamped_key(item: Tuple[Optional[float], TraceLabRound]) -> Tuple[float, int, str]:
        start, rnd = item
        assert start is not None
        return (start, rnd.round_index, rnd.round_id or "")

    if all(start is not None for start, _ in stamped):
        ordered = [rnd for _, rnd in sorted(stamped, key=stamped_key)]
    else:
        ordered = sorted(rounds, key=lambda r: (r.round_index, r.round_id or ""))
    # Re-assign dense round indices so downstream timing is stable even
    # when the trace carries sparse/duplicate indices.
    for i, r in enumerate(ordered):
        r.round_index = i
    return ordered


def _augment_for_filter(session_id: str, rounds: List[TraceLabRound]) -> Dict[str, Any]:
    per_round = [(r.input_tokens_total or 0) + (r.output_tokens or 0) for r in rounds]
    providers = sorted({r.provider for r in rounds if r.provider})
    models = sorted({r.model for r in rounds if r.model})
    return {
        "session_id": session_id,
        "provider": providers[0] if len(providers) == 1 else providers,
        "providers": providers,
        "models": models,
        "max_tokens": max(per_round, default=0),
        "total_tokens": sum(per_round),
        "num_rounds": len(rounds),
        "num_turns": len(rounds),
    }


def _keep_session(augmented: Dict[str, Any], filter_func: Optional[Any]) -> bool:
    if filter_func is None:
        return True
    try:
        return bool(filter_func(augmented))
    except Exception as e:
        raise FilterExpressionError(f"Filter expression failed on session {augmented.get('session_id')!r}: {e!r}") from e


def _hf_source(hf_path: HFDatasetPath) -> Tuple[str, Dict[str, Any]]:
    if isinstance(hf_path, dict):
        return hf_path.get("path", DEFAULT_HF_DATASET), {k: v for k, v in hf_path.items() if k != "path"}
    return hf_path or DEFAULT_HF_DATASET, {}


def _as_split(ds: Any, kwargs: Dict[str, Any]) -> Any:
    if hasattr(ds, "keys"):
        split = kwargs.get("split", "train")
        keys = list(ds.keys())
        return ds[split] if split in keys else ds[keys[0]]
    return ds


def _load_hf_table(path: str, kwargs: Dict[str, Any], name: Optional[str] = None) -> Any:
    """Load one TraceLab table, streaming by default.

    Without ``streaming=True`` datasets downloads and Arrow-builds the whole
    table (~665k rounds, ~744k tool_calls, ~2.69M timing_events) before the
    first row is yielded, which defeats the session cap and the early stop.
    Callers can opt back into full local caching with ``streaming=False`` in
    ``hf_dataset_path``.
    """
    from datasets import load_dataset

    call_kwargs = dict(kwargs)
    streaming = call_kwargs.pop("streaming", True)
    if name is not None:
        call_kwargs["name"] = name
    return _as_split(load_dataset(path, streaming=streaming, **call_kwargs), call_kwargs)


@dataclass
class TraceLabReader:
    """Reads rounds from one place (a directory, a list of files or the
    Hugging Face dataset) and groups them into sessions.

    `max_sessions` stops the read once that many sessions are kept, and
    `filter` is a lambda over a session summary that decides which are.
    With `skip_invalid` a bad row or file is logged and skipped instead of
    failing the load.
    """

    trace_directory: Optional[str] = None
    trace_files: Optional[Sequence[str]] = None
    hf_dataset_path: Optional[HFDatasetPath] = None
    filter: Optional[str] = None
    max_sessions: Optional[int] = None
    skip_invalid: bool = False

    def sessions(self) -> Dict[str, List[TraceLabRound]]:
        """The kept sessions in the order they were read, each a list of its
        rounds in replay order.

        Both local files and the Hugging Face ``rounds`` table are streamed, so
        requesting a small number of sessions never materializes the full
        corpus. Rows for one session must be contiguous in the source; TraceLab
        JSONL exports and the HF ``rounds`` table both group rows by session.
        """
        # imported here because the datagen package imports the generators
        # built on this module
        from inference_perf.datagen.replay.otel_trace_utils import _compile_filter

        filter_func = _compile_filter(self.filter)
        if filter_func:
            logger.info(f"Using filter expression: {self.filter}")

        if self.hf_dataset_path:
            blocks = self._iter_session_blocks(self._iter_hf_rounds(self.hf_dataset_path))
        else:
            blocks = self._iter_session_blocks(self._iter_local_rows())

        kept: Dict[str, List[TraceLabRound]] = {}
        n_filtered = 0
        n_invalid = 0

        for session_id, rounds, invalid in blocks:
            n_invalid += invalid
            if not rounds:
                continue
            if not _keep_session(_augment_for_filter(session_id, rounds), filter_func):
                n_filtered += 1
                continue
            kept[session_id] = rounds
            if self.max_sessions is not None and len(kept) >= self.max_sessions:
                logger.info(f"Stopped reading TraceLab input after {len(kept)} sessions (the session cap)")
                break

        # Join the HF aux tables only after selection so their rows are indexed
        # for the kept rounds, not for the whole dataset.
        if self.hf_dataset_path and kept:
            self._attach_hf_aux_tables(kept, self.hf_dataset_path)

        if filter_func:
            logger.info(f"Filter applied: {len(kept)} sessions kept, {n_filtered} rejected")
        if n_invalid:
            logger.warning(f"Skipped {n_invalid} invalid rows")
        if not kept:
            raise ValueError("No valid TraceLab sessions found")
        logger.info(f"Loaded {len(kept)} TraceLab sessions ({sum(len(v) for v in kept.values())} rounds)")
        return kept

    def _iter_local_rows(self) -> Iterator[Tuple[str, int, Dict[str, Any]]]:
        """Yield rows from local trace files lazily so callers can stop early."""
        if self.trace_directory:
            trace_dir = Path(self.trace_directory)
            if not trace_dir.exists() or not trace_dir.is_dir():
                raise ValueError(f"Trace directory does not exist or is not a directory: {trace_dir}")
            files = sorted(p for p in trace_dir.iterdir() if p.is_file() and p.name.endswith(TRACE_FILE_SUFFIXES))
            if not files:
                raise ValueError(f"No TraceLab trace files found in {trace_dir}")
        else:
            files = []
            for path in self.trace_files or []:
                f = Path(path)
                if not f.is_file():
                    raise ValueError(f"Trace file does not exist: {path}")
                files.append(f)

        for f in files:
            try:
                for line_no, blob in _iter_jsonl_rows(f):
                    yield str(f), line_no, blob
            except Exception as e:
                logger.error(f"Failed to load trace file {f.name}: {e}")
                if not self.skip_invalid:
                    raise

    def _iter_session_blocks(
        self, rows: Iterator[Tuple[str, int, Dict[str, Any]]]
    ) -> Iterator[Tuple[str, List[TraceLabRound], int]]:
        """Group contiguous rows into validated, ordered sessions.

        Yields ``(session_id, rounds, invalid_rows)``. A session is yielded as
        soon as the next session starts, so the consumer can stop reading
        without waiting for the rest of the corpus.

        Contiguity is what makes that early stop safe, so a session_id that
        reappears after another session is a hard error instead of silently
        replacing (and dropping) the rounds already collected for it.
        """
        current_id: Optional[str] = None
        rounds: List[TraceLabRound] = []
        n_invalid = 0
        seen: Set[str] = set()

        for source, line_no, blob in rows:
            try:
                rnd = TraceLabRound.model_validate(blob)
            except Exception as e:
                logger.error(f"Failed to validate TraceLab row {source}:{line_no}: {e}")
                if not self.skip_invalid:
                    raise ValueError(f"Invalid TraceLab row at {source}:{line_no}: {e}") from e
                n_invalid += 1
                continue

            if rnd.session_id != current_id:
                if current_id is not None:
                    yield current_id, _finalize_block(rounds), n_invalid
                if rnd.session_id in seen:
                    raise ValueError(
                        f"TraceLab session {rnd.session_id!r} reappears at {source}:{line_no} after another "
                        "session. Streaming early-stop requires every row of a session to be contiguous; "
                        "sort the JSONL by session_id, or export one file per session."
                    )
                seen.add(rnd.session_id)
                current_id = rnd.session_id
                rounds = []
                n_invalid = 0
            rounds.append(rnd)

        if current_id is not None:
            yield current_id, _finalize_block(rounds), n_invalid

    def _iter_hf_rounds(self, hf_path: HFDatasetPath) -> Iterator[Tuple[str, int, Dict[str, Any]]]:
        """Yield the HF ``rounds`` table row by row, without loading aux tables.

        The UW-SyFI/TraceLab dataset has multiple configs: the default
        ("rounds") table has one row per LLM invocation, "tool_calls" one row
        per tool call, and "timing_events" one row per timing event, both with
        a ``round_pk`` foreign key into rounds.
        """
        path, kwargs = _hf_source(hf_path)
        logger.info(f"Downloading TraceLab traces from Hugging Face dataset: {path}")
        try:
            ds_rounds = _load_hf_table(path, kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load Hugging Face dataset {path}: {e}") from e

        source = f"hf:{path}"
        for idx, rnd in enumerate(ds_rounds, start=1):
            if not isinstance(rnd, dict):
                if not self.skip_invalid:
                    raise ValueError(f"Unexpected row type {type(rnd)} in Hugging Face dataset {path}")
                continue
            yield source, idx, rnd

    def _attach_hf_aux_tables(self, kept: Dict[str, List[TraceLabRound]], hf_path: HFDatasetPath) -> None:
        """Join tool_calls/timing_events onto the kept rounds only.

        Indexing just the kept ``round_pk`` values avoids building dictionaries
        over the full aux tables (hundreds of thousands to millions of rows).
        """
        round_pks = {str(r.round_pk) for rounds in kept.values() for r in rounds if r.round_pk is not None}
        if not round_pks:
            return

        path, kwargs = _hf_source(hf_path)

        def _aux_rows(name: str) -> Iterator[Dict[str, Any]]:
            try:
                table = _load_hf_table(path, kwargs, name=name)
            except Exception as e:
                msg = (
                    f"Could not load '{name}' table from {path}: {e}. "
                    "Rounds will replay without it (tool-latency waits or "
                    "timestamp gaps from that table will be 0)."
                )
                if not self.skip_invalid:
                    raise ValueError(msg) from e
                logger.warning(msg)
                return
            for row in table:
                if isinstance(row, dict):
                    yield row

        tools_by_round: Dict[str, List[TraceLabToolCall]] = {}
        for t in _aux_rows("tool_calls"):
            rpk = t.get("round_pk")
            if rpk is not None and str(rpk) in round_pks:
                tools_by_round.setdefault(str(rpk), []).append(
                    TraceLabToolCall.model_validate(
                        {
                            "tool_name": t.get("tool_name"),
                            "tool_wall_latency_ms": t.get("tool_wall_latency_ms"),
                            "tool_internal_latency_ms": t.get("tool_internal_latency_ms"),
                            "emitted_at": t.get("emitted_at"),
                            "result_at": t.get("result_at"),
                        }
                    )
                )

        timing_by_round: Dict[str, List[TraceLabTimingEvent]] = {}
        for te in _aux_rows("timing_events"):
            rpk = te.get("round_pk")
            if rpk is not None and str(rpk) in round_pks:
                timing_by_round.setdefault(str(rpk), []).append(
                    TraceLabTimingEvent.model_validate({"event_type": te.get("event_type"), "timestamp": te.get("timestamp")})
                )

        for rounds in kept.values():
            for r in rounds:
                if r.round_pk is None:
                    continue
                key = str(r.round_pk)
                if key in tools_by_round:
                    r.tools = tools_by_round[key]
                if key in timing_by_round:
                    r.timing_events = timing_by_round[key]


def build_workload(sessions: Mapping[str, Sequence[TraceLabRound]], source_id: str, block_size: int) -> Workload:
    """Records and nodes for `sessions`, which run in the order given."""
    starts = [start for rounds in sessions.values() for start, _ in map(_round_time_bounds, rounds) if start is not None]
    origin = min(starts) if starts else None
    records: List[Record] = []
    nodes: List[Node] = []
    for session_id, rounds in sessions.items():
        records.extend(_records(session_id, rounds, block_size))
        nodes.extend(_nodes(session_id, rounds, origin))
    return Workload(source_id=source_id, records=records, arrangement=Arrangement(nodes=nodes))


def _round_id(session_id: str, index: int) -> str:
    return f"{session_id}:round_{index}"


def _records(session_id: str, rounds: Sequence[TraceLabRound], block_size: int) -> List[Record]:
    records: List[Record] = []
    blocks: List[int] = []  # the previous round's block ids, in prompt order
    next_id = 0
    for index, rnd in enumerate(rounds):
        input_tokens = max(rnd.input_tokens_total, 0)
        # a prefix can be recorded larger than the prompt it leads, after a compaction
        ids = blocks[: min(max(rnd.prefix_tokens, 0), input_tokens) // block_size]
        while len(ids) < math.ceil(input_tokens / block_size):
            ids.append(next_id)
            next_id += 1
        blocks = ids
        # a round that recorded no input still has to send a message
        prompt: ContentPart = (
            SyntheticPart(num_tokens=input_tokens, block_ids=ids, block_size=block_size, scope="session")
            if input_tokens
            else TextPart(text=" ")
        )
        records.append(
            Record(
                id=_round_id(session_id, index),
                session_id=session_id,
                turns=[
                    Turn(role="user", parts=[prompt]),
                    Turn(role="assistant", output_tokens=max(rnd.output_tokens, 0)),
                ],
                metadata={"model": rnd.model or DEFAULT_MODEL},
            )
        )
    return records


def _nodes(session_id: str, rounds: Sequence[TraceLabRound], origin: Optional[float]) -> List[Node]:
    nodes: List[Node] = []
    for index, rnd in enumerate(rounds):
        start, _ = _round_time_bounds(rnd)
        think_ms = 0
        if index > 0:
            gap = round_gap_ms(rounds[index - 1], rnd)
            think_ms = gap if gap is not None else tool_latency_ms(rounds[index - 1])
        nodes.append(
            Node(
                id=_round_id(session_id, index),
                record_id=_round_id(session_id, index),
                turn=1,
                send_at_ms=int(round((start - origin) * 1000)) if start is not None and origin is not None else None,
                depends_on=[nodes[-1].id] if nodes else [],
                think_ms=think_ms,
            )
        )
    return nodes


class TraceLabSource(WorkloadSource):
    format = "TraceLab"
    default_block_size = TRACELAB_BLOCK_SIZE

    def load(self, path: Path) -> Workload:
        reader = TraceLabReader(trace_directory=str(path)) if path.is_dir() else TraceLabReader(trace_files=[str(path)])
        return build_workload(reader.sessions(), source_id=path.name, block_size=self.block_size)
