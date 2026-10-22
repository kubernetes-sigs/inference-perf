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
"""The arrangement: how records become requests over time.

One node type covers every delivery shape. A node with a send time and no
dependencies is a timestamped independent request (Azure, Mooncake, BurstGPT).
A node with dependencies is one call of a session graph (OTel, Weka, TraceLab),
and `think_ms` is how long after its last dependency finishes it goes out.
Which scheduler runs an arrangement follows from whether any node has
dependencies; the arrangement itself does not say.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field, model_validator

from inference_perf.config.common import StrictBaseModel


class Node(StrictBaseModel):
    """One request. It elicits `record.turns[turn]`, which must be an assistant
    turn; the turns before it are the context that goes on the wire."""

    id: str
    record_id: str
    turn: int = Field(ge=0)
    send_at_ms: Optional[int] = Field(default=None, ge=0)
    depends_on: List[str] = []
    think_ms: int = Field(default=0, ge=0)


class Arrangement(StrictBaseModel):
    nodes: List[Node]

    @model_validator(mode="after")
    def _dependencies_runnable(self) -> Arrangement:
        ids = [node.id for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("node ids must be unique")
        known = set(ids)
        for node in self.nodes:
            for dep in node.depends_on:
                if dep not in known:
                    raise ValueError(f"node {node.id!r} depends on unknown node {dep!r}")
        # a node in a cycle never becomes runnable, so the replay would stall on
        # it instead of failing here.
        waiting = {node.id: set(node.depends_on) for node in self.nodes}
        dependents: Dict[str, List[str]] = {node_id: [] for node_id in waiting}
        for node_id, deps in waiting.items():
            for dep in deps:
                dependents[dep].append(node_id)
        ready = [node_id for node_id, deps in waiting.items() if not deps]
        while ready:
            done = ready.pop()
            del waiting[done]
            for child in dependents[done]:
                waiting[child].discard(done)
                if not waiting[child]:
                    ready.append(child)
        if waiting:
            raise ValueError(f"node dependencies form a cycle, so these nodes can never run: {sorted(waiting)}")
        return self

    def has_dependencies(self) -> bool:
        return any(node.depends_on for node in self.nodes)
