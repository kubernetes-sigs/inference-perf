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
"""Mooncake FAST'25 traces (github.com/kvcache-ai/Mooncake, FAST25-release).

One JSON object per line:

    {"timestamp": 27482, "input_length": 6955, "output_length": 52,
     "hash_ids": [46, 47, 48, ...]}

`timestamp` is the arrival time in milliseconds from the start of the trace.
`hash_ids` are the request's prefix blocks, 512 tokens each, numbered across
the whole file: two requests with the same leading ids share that prefix. No
text is recorded. Each line is a session of one request: a user message holding
a text spec, with the output length to ask for, sent at the recorded time and
waiting for nothing.
"""

from __future__ import annotations

from typing import List, Sequence, TypedDict

from inference_perf.workload.format import Row, TraceFormat
from inference_perf.workload.record import Delivery, Message, Record, Schedule, Session, TextPart, TextSpec

MOONCAKE_BLOCK_SIZE = 512
_FIELDS = ("timestamp", "input_length", "output_length", "hash_ids")


class MooncakeRow(TypedDict):
    """One trace line after _parse has checked every field."""

    timestamp: int
    input_length: int
    output_length: int
    hash_ids: List[int]


def _parse(row: Row) -> MooncakeRow:
    missing = [k for k in _FIELDS if k not in row]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")
    counts: List[int] = []
    for k in ("timestamp", "input_length", "output_length"):
        value = row[k]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{k} must be a non-negative integer, got {value!r}")
        counts.append(value)
    hash_ids = row["hash_ids"]
    if not isinstance(hash_ids, list) or not all(isinstance(h, int) and not isinstance(h, bool) for h in hash_ids):
        raise ValueError("hash_ids must be a list of integers")
    return MooncakeRow(timestamp=counts[0], input_length=counts[1], output_length=counts[2], hash_ids=list(hash_ids))


class MooncakeFormat(TraceFormat):
    name = "Mooncake"
    # no sessions are recorded: every request stands alone
    session_key = None
    default_block_size = MOONCAKE_BLOCK_SIZE

    def check(self, row: Row) -> None:
        _parse(row)

    def session(self, key: str, rows: Sequence[Row]) -> Session:
        entry = _parse(rows[0])
        prompt = TextSpec(
            num_tokens=entry["input_length"], block_ids=entry["hash_ids"], block_size=self.block_size, scope="trace"
        )
        record = Record(
            id=key,
            messages=[Message(role="user", parts=[TextPart(spec=prompt)])],
            output_tokens=entry["output_length"],
        )
        delivery = Delivery(schedules={key: Schedule(start_ms=entry["timestamp"])})
        return Session(id=key, records=[record], delivery=delivery)
