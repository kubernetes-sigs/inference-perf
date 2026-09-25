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
"""Trace formats. Each one is a parser that produces records and an
arrangement; none of them schedules, tokenizes or builds payloads.

A format is added by registering its class in `SOURCES` under the name the
config uses to pick it."""

from pathlib import Path
from typing import Dict, Optional, Type

from .base import Workload, WorkloadSource
from .mooncake import MooncakeSource

SOURCES: Dict[str, Type[WorkloadSource]] = {
    MooncakeSource.format: MooncakeSource,
}


def load_workload(format: str, file: str, block_size: Optional[int] = None) -> Workload:
    """Parse `file` with the registered source named `format`."""
    source_cls = SOURCES.get(format)
    if source_cls is None:
        known = ", ".join(sorted(SOURCES)) or "none"
        raise ValueError(f"Unknown workload format {format!r}; registered formats: {known}")
    return source_cls(block_size=block_size).load(Path(file))


__all__ = ["MooncakeSource", "SOURCES", "Workload", "WorkloadSource", "load_workload"]
