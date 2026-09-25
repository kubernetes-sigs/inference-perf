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
text is recorded. Each line becomes a one-exchange record (a user turn holding
a synthetic part, an assistant turn holding the output length) and one node
sent at the recorded time with no dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from inference_perf.workload.arrangement import Arrangement, Node
from inference_perf.workload.record import Record, SyntheticPart, Turn
from inference_perf.workload.sources.base import Workload, WorkloadSource

MOONCAKE_BLOCK_SIZE = 512
_FIELDS = ("timestamp", "input_length", "output_length", "hash_ids")


class MooncakeSource(WorkloadSource):
    format = "Mooncake"
    default_block_size = MOONCAKE_BLOCK_SIZE

    def load(self, path: Path) -> Workload:
        records: List[Record] = []
        nodes: List[Node] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                entry = self._parse_line(line, line_num, path)
                record_id = str(len(records))
                records.append(
                    Record(
                        id=record_id,
                        turns=[
                            Turn(
                                role="user",
                                parts=[
                                    SyntheticPart(
                                        num_tokens=entry["input_length"],
                                        block_ids=entry["hash_ids"],
                                        block_size=self.block_size,
                                        scope="trace",
                                    )
                                ],
                            ),
                            Turn(role="assistant", output_tokens=entry["output_length"]),
                        ],
                    )
                )
                nodes.append(Node(id=record_id, record_id=record_id, turn=1, send_at_ms=entry["timestamp"]))
        return Workload(source_id=path.name, records=records, arrangement=Arrangement(nodes=nodes))

    def _parse_line(self, line: str, line_num: int, path: Path) -> Dict[str, Any]:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_num}: not JSON: {e}") from e
        if not isinstance(entry, dict):
            raise ValueError(f"{path}:{line_num}: expected an object, got {type(entry).__name__}")
        missing = [k for k in _FIELDS if k not in entry]
        if missing:
            raise ValueError(f"{path}:{line_num}: missing {', '.join(missing)}")
        for k in ("timestamp", "input_length", "output_length"):
            if not isinstance(entry[k], int) or isinstance(entry[k], bool) or entry[k] < 0:
                raise ValueError(f"{path}:{line_num}: {k} must be a non-negative integer, got {entry[k]!r}")
        hash_ids = entry["hash_ids"]
        if not isinstance(hash_ids, list) or not all(isinstance(h, int) and not isinstance(h, bool) for h in hash_ids):
            raise ValueError(f"{path}:{line_num}: hash_ids must be a list of integers")
        return entry
