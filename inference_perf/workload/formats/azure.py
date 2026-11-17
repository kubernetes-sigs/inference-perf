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
"""Azure LLM inference traces (github.com/Azure/AzurePublicDataset).

A CSV with an optional header, one request per line:

    TIMESTAMP,ContextTokens,GeneratedTokens
    2023-11-16 18:15:46.6805900,374,44

`TIMESTAMP` is an absolute time. Only the lengths are recorded: no text and
no prefix structure, so every request is a text spec with no blocks, and
requests share nothing. Each line is a session of one request, sent at its
recorded time and waiting for nothing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import takewhile
from pathlib import Path
from typing import Iterator, Sequence, Tuple

from inference_perf.workload.format import Row, TraceFormat
from inference_perf.workload.record import Delivery, Message, Record, Schedule, Session, TextPart, TextSpec

_COLUMNS = ("TIMESTAMP", "ContextTokens", "GeneratedTokens")
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def _parse_timestamp_ms(raw: str) -> int:
    """Seconds since the epoch in ms, for the dataset's `YYYY-MM-DD HH:MM:SS.fffffff`
    stamps and the ISO variants (`T` separator, `Z` suffix) that show up in
    re-exports. Fractions longer than six digits are truncated; the format
    never carries a zone, so it is read as UTC."""
    text = raw.strip().strip('"').replace("T", " ").rstrip("Z").strip()
    if "." in text:
        head, frac = text.split(".", 1)
        digits = "".join(takewhile(str.isdigit, frac))
        text = f"{head}.{digits[:6].ljust(6, '0')}"
    else:
        text = f"{text}.000000"
    stamp = datetime.strptime(text, _TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
    return int(round(stamp.timestamp() * 1000))


def _parse(row: Row) -> Tuple[int, int, int]:
    """(send time in ms, input tokens, output tokens) of one line."""
    stamp = str(row.get("TIMESTAMP", ""))
    try:
        stamp_ms = _parse_timestamp_ms(stamp)
    except ValueError as e:
        raise ValueError(f"bad timestamp {stamp!r}") from e
    counts = []
    for name in ("ContextTokens", "GeneratedTokens"):
        value = str(row.get(name, ""))
        if not value.isdigit():
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        counts.append(int(value))
    return stamp_ms, counts[0], counts[1]


class AzurePublicDatasetFormat(TraceFormat):
    name = "AzurePublicDataset"
    # no sessions are recorded: every request stands alone
    session_key = None
    file_suffixes = (".csv",)

    def read(self, path: Path) -> Iterator[Tuple[int, Row]]:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                if line_num == 1 and line.lstrip().upper().startswith("TIMESTAMP"):
                    continue
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != len(_COLUMNS):
                    raise ValueError(f"{path}:{line_num}: expected {','.join(_COLUMNS)}, got {len(fields)} fields")
                yield line_num, dict(zip(_COLUMNS, fields, strict=True))

    def check(self, row: Row) -> None:
        _parse(row)

    def session(self, key: str, rows: Sequence[Row]) -> Session:
        stamp_ms, input_tokens, output_tokens = _parse(rows[0])
        record = Record(
            id=key,
            messages=[
                Message(role="user", parts=[TextPart(spec=TextSpec(num_tokens=input_tokens, block_size=self.block_size))])
            ],
            output_tokens=output_tokens,
        )
        return Session(id=key, records=[record], delivery=Delivery(schedules={key: Schedule(start_ms=stamp_ms)}))
