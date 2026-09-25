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

`TIMESTAMP` is an absolute time; the first line's is the origin. Only the
lengths are recorded: no text and no prefix structure, so every request is
a synthetic part with no blocks, and requests share nothing. Each line is
one node sent at its offset from the origin with no dependencies.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import takewhile
from pathlib import Path
from typing import List

from inference_perf.workload.arrangement import Arrangement, Node
from inference_perf.workload.record import Record, SyntheticPart, Turn
from inference_perf.workload.sources.base import Workload, WorkloadSource

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


class AzurePublicDatasetSource(WorkloadSource):
    format = "AzurePublicDataset"

    def load(self, path: Path) -> Workload:
        rows: List[tuple[int, int, int]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                if line_num == 1 and line.lstrip().upper().startswith("TIMESTAMP"):
                    continue
                rows.append(self._parse_line(line, line_num, path))

        origin_ms = rows[0][0] if rows else 0
        records: List[Record] = []
        nodes: List[Node] = []
        for stamp_ms, input_tokens, output_tokens in rows:
            record_id = str(len(records))
            records.append(
                Record(
                    id=record_id,
                    turns=[
                        Turn(role="user", parts=[SyntheticPart(num_tokens=input_tokens, block_size=self.block_size)]),
                        Turn(role="assistant", output_tokens=output_tokens),
                    ],
                )
            )
            nodes.append(Node(id=record_id, record_id=record_id, turn=1, send_at_ms=max(0, stamp_ms - origin_ms)))
        return Workload(source_id=path.name, records=records, arrangement=Arrangement(nodes=nodes))

    def _parse_line(self, line: str, line_num: int, path: Path) -> tuple[int, int, int]:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_num}: expected TIMESTAMP,ContextTokens,GeneratedTokens, got {len(fields)} fields")
        try:
            stamp_ms = _parse_timestamp_ms(fields[0])
        except ValueError as e:
            raise ValueError(f"{path}:{line_num}: bad timestamp {fields[0]!r}") from e
        counts: List[int] = []
        for name, value in (("ContextTokens", fields[1]), ("GeneratedTokens", fields[2])):
            if not value.isdigit():
                raise ValueError(f"{path}:{line_num}: {name} must be a non-negative integer, got {value!r}")
            counts.append(int(value))
        return stamp_ms, counts[0], counts[1]
