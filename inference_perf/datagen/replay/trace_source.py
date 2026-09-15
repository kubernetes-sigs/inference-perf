# Copyright 2025 The Kubernetes Authors.
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

"""Trace sources for OTel replay.

A trace source supplies replay records to OTelTraceReplayDataGenerator without
requiring every input to become one Hugging Face Dataset. The generator asks for
record metadata once at startup (to derive session IDs and the shuffled order),
then loads one record body at a time as sessions are built.

Two properties matter: ``list_records()`` returns lightweight metadata and never
retains trace bodies, so startup memory scales with the number of records rather
than total span content; ``load_record(i)`` returns one normalized record, which
callers drop after building the graph.

This bounds local-corpus memory and keeps all spans out of any single Arrow
column -- an aggregate column can exceed Arrow's 32-bit offsets and fail with
"offset overflow while concatenating arrays" during dataset fingerprinting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, cast

from datasets import Dataset

from inference_perf.datagen.replay.wire_trace_converter import (
    FORMAT_OTEL_JSONL,
    WIRE_FORMATS,
    convert_wire_file,
    derive_trace_id,
    detect_trace_format,
    iter_wire_spans,
)

logger = logging.getLogger(__name__)


class TraceSourceError(ValueError):
    """Base class for trace-source failures.

    Subclasses ValueError because that is what the eager loader propagated for a
    bad corpus (json.JSONDecodeError and the converter's own raises are both
    ValueError), and callers catch it. The subclasses below let the generator
    distinguish causes without breaking that.
    """


class TraceReadError(TraceSourceError):
    """A record's bytes could not be read (missing file, I/O error, bad encoding)."""


class InvalidTraceError(TraceSourceError):
    """A record was read but is not a usable trace (bad JSON, failed schema check)."""


class UnsupportedTraceFormatError(TraceSourceError):
    """A record's format is not one this build knows how to convert."""


@dataclass(frozen=True)
class TraceRecordMetadata:
    """What the generator needs about a record before loading its body.

    ``session_id_suffix`` trails the scheduler session ID, which the generator
    prefixes with the shuffled slot (``trace{slot}_{suffix}``); unique suffixes
    keep logs readable when a corpus repeats filenames. ``source_id`` is the
    traceability handle carried into metrics and graph serialization.

    Either is empty when the record carries no such identity: both fallbacks
    derive from the post-shuffle slot, which a source cannot see, so applying them
    is the generator's job.
    """

    session_id_suffix: str
    source_id: str


class TraceSource(Protocol):
    """Supplies replay records by index.

    Indices are positions in the sequence ``list_records()`` returns and stay
    valid for the source's lifetime.
    """

    def list_records(self) -> Sequence[TraceRecordMetadata]:
        """Return metadata for every record, in a stable order.

        Called once during initialization. Must not retain record bodies.
        """
        ...

    def load_record(self, record_index: int) -> Dict[str, Any]:
        """Return one normalized record.

        Raises:
            TraceReadError: the record's bytes could not be read.
            InvalidTraceError: the bytes are not a usable trace.
            UnsupportedTraceFormatError: the format is unknown to this build.
        """
        ...


class HFDatasetTraceSource:
    """Reads records from a Hugging Face Dataset, one row at a time.

    The dataset stays memory-mapped: only the small id columns are read upfront,
    and each row's spans are read from disk when that row is loaded. This mirrors
    the behavior the generator had before trace sources existed -- for large
    corpora, materializing every row costs tens of GB resident in the parent,
    inherited by every forked worker.
    """

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    @property
    def dataset(self) -> Dataset:
        """The wrapped dataset.

        Exposed for filtering, which Dataset implements natively and more
        efficiently than a row-by-row pass.
        """
        return self._dataset

    def list_records(self) -> Sequence[TraceRecordMetadata]:
        num_rows = len(self._dataset)
        cols = self._dataset.column_names

        # Reading whole id columns is what the generator did before; they are
        # small next to spans. Spans stay on disk.
        session_ids = list(self._dataset["session_id"]) if "session_id" in cols else [None] * num_rows
        source_ids = list(self._dataset["source_id"]) if "source_id" in cols else [None] * num_rows

        # Missing ids stay empty rather than being defaulted here: both fallbacks
        # depend on the generator's post-shuffle slot, which this source cannot see.
        return [
            TraceRecordMetadata(
                session_id_suffix=session_ids[row] or "",
                source_id=source_ids[row] or "",
            )
            for row in range(num_rows)
        ]

    def load_record(self, record_index: int) -> Dict[str, Any]:
        try:
            return cast(Dict[str, Any], dict(self._dataset[record_index]))
        except IndexError:
            raise
        except Exception as e:
            raise TraceReadError(f"Failed to read dataset row {record_index}: {e}") from e


class LocalTraceFormat(str, Enum):
    """How a local file maps to replay records.

    OTEL_JSON and WIRE are one record per file. OTEL_JSONL is one record per
    non-blank line, so its records need a row index to be addressable.
    """

    OTEL_JSON = "otel_json"
    OTEL_JSONL = "otel_jsonl"
    WIRE = "wire"


@dataclass(frozen=True)
class LocalTraceRecord:
    """Locates one replay record inside the local corpus.

    ``row_index`` is None for the one-record-per-file formats; for OTEL_JSONL it
    is the position among non-blank lines, since blank lines never consume an
    index. ``normalize_row_index`` is the ``row_index`` handed to
    ``_normalize_file_trace``, set only when a .jsonl held more than one record so
    single-record files keep a plain session_id (the eager ``len(docs) > 1``).

    ``byte_offset``/``byte_length`` bound the record's line, in bytes, and are set
    only for OTEL_JSONL. Loading seeks straight there; scanning to row n instead
    would decode every line before it, making a full pass O(N^2). The file must
    not change between indexing and loading.
    """

    path: Path
    trace_format: LocalTraceFormat
    row_index: Optional[int]
    normalize_row_index: Optional[int]
    session_id_suffix: str
    source_id: str
    byte_offset: Optional[int] = None
    byte_length: Optional[int] = None


# Injected by the generator, which owns normalization: (data, source_name,
# source_path, row_index) -> normalized record.
NormalizeFn = Callable[[Dict[str, Any], str, str, Optional[int]], Dict[str, Any]]


def validate_record_schema(record: Dict[str, Any], where: str) -> None:
    """Per-record replacement for the old dataset-level schema check.

    The eager path validated ``dataset[0]`` only -- one row stood in for the
    corpus. Checking every record as it is built is strictly stronger and does
    not need an aggregate dataset to exist.
    """
    if not isinstance(record, dict):
        raise InvalidTraceError(f"{where}: expected a trace object, got {type(record).__name__}")
    if "spans" not in record:
        raise InvalidTraceError(f"{where}: missing the required 'spans' field")
    spans = record["spans"]
    if not isinstance(spans, (list, tuple)):
        raise InvalidTraceError(f"{where}: 'spans' must be a list, got {type(spans).__name__}")


class LocalTraceSource:
    """Reads replay records from local .json / .jsonl / wire-capture files.

    Records are addressed by locator, never accumulated: nothing here builds a
    Hugging Face Dataset, so a corpus whose aggregate spans exceed Arrow's 32-bit
    offsets cannot fail at load time.

    Startup is bounded but not free: format detection reads content by design
    (a .json file is parsed whole, since it may be pretty-printed), and .jsonl
    files are scanned line-by-line because their record count -- which the
    generator needs upfront -- is not in the file metadata. Neither retains a body.

    ``validate_at_startup`` (the default) checks every record first, keeping the
    eager path's fail-fast contract. Memory stays bounded to one record, but the
    whole corpus is read -- for wire captures a full conversion pass, tens of
    seconds for a few hundred large ones. False defers validation to
    load_record(), where a bad record surfaces at dispatch instead.
    """

    def __init__(
        self,
        files: Sequence[Path],
        skip_invalid: bool,
        normalize: NormalizeFn,
        validate_at_startup: bool = True,
    ) -> None:
        self._skip_invalid = skip_invalid
        self._normalize = normalize
        self._validate_at_startup = validate_at_startup
        self._records: List[LocalTraceRecord] = self._index(files)

    def _index(self, files: Sequence[Path]) -> List[LocalTraceRecord]:
        records: List[LocalTraceRecord] = []
        for path in files:
            try:
                records.extend(self._index_one(path))
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                if not self._skip_invalid:
                    raise
        return records

    def _index_one(self, path: Path) -> List[LocalTraceRecord]:
        trace_format = detect_trace_format(path)

        if trace_format == FORMAT_OTEL_JSONL:
            byte_ranges = _scan_jsonl_records(path, validate=self._validate_at_startup)
            count = len(byte_ranges)
            if count == 0:
                raise InvalidTraceError(f"{path}: no OTel documents found")
            multi = count > 1
            return [
                LocalTraceRecord(
                    path=path,
                    trace_format=LocalTraceFormat.OTEL_JSONL,
                    row_index=row,
                    byte_offset=byte_ranges[row][0],
                    byte_length=byte_ranges[row][1],
                    normalize_row_index=row if multi else None,
                    # Mirrors _normalize_file_trace's fallback so multi-record
                    # files cannot collide, and single-record ones stay plain.
                    session_id_suffix=f"{path.stem}_{row}" if multi else path.stem,
                    source_id=f"{path}#{row}" if multi else str(path),
                )
                for row in range(count)
            ]

        if trace_format in WIRE_FORMATS:
            # One file is one session (see convert_wire_file). Identity comes from
            # the converter, which prefers a recorded session-id header and falls
            # back to a path hash -- capture files are commonly all named
            # calls.jsonl, so a bare stem would collide across sessions.
            trace_id = derive_trace_id(path)
            if self._validate_at_startup:
                _validate_wire_convertible(path, trace_id)
            return [
                LocalTraceRecord(
                    path=path,
                    trace_format=LocalTraceFormat.WIRE,
                    row_index=None,
                    normalize_row_index=None,
                    session_id_suffix=trace_id,
                    source_id=str(path),
                )
            ]

        # A .json file was already parsed whole by detect_trace_format (it may be
        # pretty-printed), and reaching here means it carried a "spans" key, so its
        # schema is validated by construction -- no extra read needed.
        return [
            LocalTraceRecord(
                path=path,
                trace_format=LocalTraceFormat.OTEL_JSON,
                row_index=None,
                normalize_row_index=None,
                # The stem, matching _normalize_file_trace's own fallback. The eager
                # path used the record's embedded session_id/trace_id here, which a
                # locator cannot see; the stem keeps scheduler ids traceable to their
                # file instead of degrading to a bare slot number.
                session_id_suffix=path.stem,
                source_id=str(path),
            )
        ]

    @property
    def records(self) -> Sequence[LocalTraceRecord]:
        """The record index. Exposed for filtering and for tests."""
        return self._records

    def list_records(self) -> Sequence[TraceRecordMetadata]:
        return [TraceRecordMetadata(session_id_suffix=r.session_id_suffix, source_id=r.source_id) for r in self._records]

    def restrict_to(self, keep_indices: Sequence[int]) -> None:
        """Keep only the listed records, preserving their relative order.

        Used by the filter pass, which must run before duplication and
        num_sessions selection to keep those semantics unchanged.
        """
        self._records = [self._records[i] for i in keep_indices]

    def load_record(self, record_index: int) -> Dict[str, Any]:
        record = self._records[record_index]
        raw = self._read_raw(record)
        normalized = self._normalize(
            raw,
            record.path.name,
            str(record.path),
            record.normalize_row_index,
        )
        validate_record_schema(normalized, str(record.path))
        return normalized

    def _read_raw(self, record: LocalTraceRecord) -> Dict[str, Any]:
        """Read one record's raw trace document."""
        path = record.path
        if record.trace_format is LocalTraceFormat.OTEL_JSONL:
            # Seek to the range recorded at indexing; scanning here would be O(N^2).
            assert record.byte_offset is not None and record.byte_length is not None
            try:
                with path.open("rb") as stream:
                    stream.seek(record.byte_offset)
                    line = stream.read(record.byte_length)
            except OSError as e:
                raise TraceReadError(f"{path}: {e}") from e
            if not line.strip():
                raise InvalidTraceError(f"{path}: row {record.row_index} is empty; did the file change since indexing?")
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                raise InvalidTraceError(f"{path}: row {record.row_index}: invalid JSON: {e}") from e
            if not isinstance(doc, dict):
                raise InvalidTraceError(f"{path}: row {record.row_index}: expected a JSON object, got {type(doc).__name__}")
            return cast(Dict[str, Any], doc)

        if record.trace_format is LocalTraceFormat.WIRE:
            try:
                converted = convert_wire_file(path)
            except ValueError as e:
                raise InvalidTraceError(str(e)) from e
            except OSError as e:
                raise TraceReadError(f"{path}: {e}") from e
            if not converted["spans"]:
                raise InvalidTraceError(f"{path}: wire capture produced no convertible spans")
            return converted

        try:
            return cast(Dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as e:
            raise InvalidTraceError(f"{path}: not valid JSON: {e}") from e
        except OSError as e:
            raise TraceReadError(f"{path}: {e}") from e


def _scan_jsonl_records(path: Path, validate: bool) -> List[Tuple[int, int]]:
    """Return the (byte_offset, byte_length) of every non-blank line.

    Blank lines are skipped without consuming an index, matching
    iter_otel_jsonl_traces. readline() rather than iteration, because iteration
    reads ahead and tell() would then report the buffer position, not the line's.

    ``validate`` also decodes and schema-checks each line, then drops it, so a bad
    record fails here instead of at dispatch; without it nothing is parsed.
    """
    byte_ranges: List[Tuple[int, int]] = []
    line_no = 0
    try:
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                line_no += 1
                if not line.strip():
                    continue
                if validate:
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError as e:
                        # Physical line, so the message points where the reader will look.
                        raise InvalidTraceError(f"{path}:{line_no}: invalid JSON: {e}") from e
                    validate_record_schema(doc, f"{path}#{len(byte_ranges)}")
                byte_ranges.append((offset, len(line)))
    except OSError as e:
        raise TraceReadError(f"{path}: {e}") from e
    return byte_ranges


def _validate_wire_convertible(path: Path, trace_id: str) -> None:
    """Check a whole wire capture converts, without retaining its spans.

    Consumed to the end, so malformed JSON anywhere fails here rather than at
    dispatch (what the eager convert_wire_file() gave); spans are counted and
    dropped rather than collected. Records yielding no span are skipped, as in
    iter_wire_spans -- the file need only be readable and produce one span.
    """
    span_count = 0
    try:
        for _span in iter_wire_spans(path, trace_id):
            span_count += 1
    except ValueError as e:
        raise InvalidTraceError(str(e)) from e
    except OSError as e:
        raise TraceReadError(f"{path}: {e}") from e

    if span_count == 0:
        raise InvalidTraceError(f"{path}: wire capture produced no convertible spans")
