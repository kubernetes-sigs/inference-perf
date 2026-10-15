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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List

from inference_perf.workload.arrangement import Arrangement
from inference_perf.workload.record import Record


@dataclass
class Workload:
    """What a source produces: the records, the arrangement over them, and an
    id for the source (the file name, normally) that scopes trace-wide
    synthetic block ids."""

    source_id: str
    records: List[Record]
    arrangement: Arrangement
    _by_id: Dict[str, Record] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {record.id: record for record in self.records}
        if len(self._by_id) != len(self.records):
            raise ValueError("record ids must be unique")
        for node in self.arrangement.nodes:
            record = self._by_id.get(node.record_id)
            if record is None:
                raise ValueError(f"node {node.id!r} names unknown record {node.record_id!r}")
            if node.turn >= len(record.turns) or record.turns[node.turn].role != "assistant":
                raise ValueError(f"node {node.id!r} must elicit an assistant turn of record {record.id!r}")

    def record(self, record_id: str) -> Record:
        return self._by_id[record_id]


class WorkloadSource(ABC):
    """A trace format. The only thing a new format has to write."""

    # The name the config uses to pick this format.
    format: ClassVar[str]

    @abstractmethod
    def load(self, path: Path) -> Workload:
        raise NotImplementedError
