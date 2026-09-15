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

Two properties matter:

  * ``list_records()`` returns lightweight metadata only. It must not retain
    trace bodies, so startup memory is proportional to the number of records
    rather than to total span content.
  * ``load_record(i)`` returns exactly one normalized record. Callers drop it
    after building the graph, so at most a small number are ever live.

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
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, cast

from datasets import Dataset

from inference_perf.datagen.replay.wire_trace_converter import (
    FORMAT_OTEL_JSONL,
    WIRE_FORMATS,
    convert_wire_file,
    derive_trace_id,
    detect_trace_format,
    iter_otel_jsonl_traces,
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

    ``session_id_suffix`` is the trailing part of the scheduler session ID; the
    generator prefixes it with the shuffled slot (``trace{slot}_{suffix}``), so
    the suffix alone need not be unique -- but making it unique keeps logs and
    reports readable when a corpus repeats filenames.

    It is empty when the record carries no usable identity. The fallback is the
    generator's to apply, not the source's: it is derived from the post-shuffle
    slot, which a source cannot see.

    ``source_id`` is the traceability handle carried into metrics and graph
    serialization. It is usually a path, optionally with a record locator. Empty
    when the record carries none, which likewise leaves the fallback to the
    generator.
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

    ``row_index`` is None for the one-record-per-file formats. For OTEL_JSONL it
    is the position among non-blank lines, matching how the eager path enumerated
    ``iter_otel_jsonl_traces`` output -- blank lines are skipped without consuming
    an index.

    ``normalize_row_index`` is what gets passed to ``_normalize_file_trace`` as
    its ``row_index``. It is None for single-record files (which keep a plain
    session_id) and set only when a .jsonl held more than one record, preserving
    the eager path's ``multi = len(docs) > 1`` behavior.
    """

    path: Path
    trace_format: LocalTraceFormat
    row_index: Optional[int]
    normalize_row_index: Optional[int]
    session_id_suffix: str
    source_id: str


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

    Startup cost is bounded but not zero. Format detection reads content by
    design (``detect_trace_format``), and for a .json file that means parsing it
    whole, since it may be pretty-printed. Native .jsonl files are additionally
    scanned line-by-line to count records, because a file's record count cannot
    be known from its metadata and the generator needs the total upfront. Neither
    step retains a trace body.

    ``validate_at_startup`` (the default) checks every record before replay
    begins, keeping the eager path's fail-fast contract: a malformed corpus fails
    during initialization rather than partway through a run. Set it False to defer
    validation to load_record(), where a bad record surfaces when its session is
    dispatched. Memory is bounded either way.
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
            count = _validate_jsonl_records(path) if self._validate_at_startup else _count_jsonl_records(path)
            if count == 0:
                raise InvalidTraceError(f"{path}: no OTel documents found")
            multi = count > 1
            return [
                LocalTraceRecord(
                    path=path,
                    trace_format=LocalTraceFormat.OTEL_JSONL,
                    row_index=row,
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
                _probe_wire_convertible(path, trace_id)
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
            assert record.row_index is not None
            try:
                for i, doc in enumerate(iter_otel_jsonl_traces(path)):
                    if i == record.row_index:
                        return doc
            except ValueError as e:
                raise InvalidTraceError(str(e)) from e
            except OSError as e:
                raise TraceReadError(f"{path}: {e}") from e
            raise InvalidTraceError(f"{path}: row {record.row_index} no longer present")

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


def _count_jsonl_records(path: Path) -> int:
    """Count non-blank lines without decoding their JSON.

    Blank lines are skipped without consuming an index, matching
    iter_otel_jsonl_traces. Reading bytes rather than text keeps this off the
    JSON decoder entirely -- the cost is I/O, not parsing.
    """
    count = 0
    try:
        with path.open("rb") as stream:
            for line in stream:
                if line.strip():
                    count += 1
    except OSError as e:
        raise TraceReadError(f"{path}: {e}") from e
    return count


def _validate_jsonl_records(path: Path) -> int:
    """Count records, checking each decodes to a trace object with spans.

    The counting-only variant reads bytes, so a malformed line deep in a file
    would not surface until its session was dispatched. Each line is decoded and
    dropped immediately, keeping memory bounded to one record.
    """
    count = 0
    try:
        for doc in iter_otel_jsonl_traces(path):
            validate_record_schema(doc, f"{path}#{count}")
            count += 1
    except ValueError as e:
        # iter_otel_jsonl_traces raises ValueError naming the file and line.
        raise InvalidTraceError(str(e)) from e
    except OSError as e:
        raise TraceReadError(f"{path}: {e}") from e
    return count


def _probe_wire_convertible(path: Path, trace_id: str) -> None:
    """Check a wire capture yields at least one convertible span.

    Stops at the first span instead of converting the whole file, which is far
    cheaper and still raises the failure the eager path raised, plus any malformed
    JSON ahead of that span.
    """
    try:
        for _span in iter_wire_spans(path, trace_id):
            return
    except ValueError as e:
        raise InvalidTraceError(str(e)) from e
    except OSError as e:
        raise TraceReadError(f"{path}: {e}") from e
    raise InvalidTraceError(f"{path}: wire capture produced no convertible spans")
