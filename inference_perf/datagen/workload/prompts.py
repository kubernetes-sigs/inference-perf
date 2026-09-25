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
"""From a record and a target turn to the messages that go on the wire.

This is the one place the record's content rules are applied: the prompt for
a node is the turns before its target, cut down to the nearest self-contained
turn when there is one, with every synthetic part materialized under the
scope the part asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from inference_perf.workload.materialize import BlockTextMaterializer
from inference_perf.workload.record import MediaPart, Record, SyntheticPart, TextPart, Turn


@dataclass
class PromptMessage:
    role: str
    text: str
    # Token count the record declared for this message, when every part
    # declared one; None when the count has to be measured.
    declared_tokens: Optional[int]


class RecordPrompter:
    def __init__(self, source_id: str, materializer: BlockTextMaterializer) -> None:
        self.source_id = source_id
        self.materializer = materializer

    def messages(self, record: Record, turn: int) -> List[PromptMessage]:
        """The messages that elicit `record.turns[turn]`."""
        prompt_turns = record.turns[:turn]
        start = 0
        for i in range(len(prompt_turns) - 1, -1, -1):
            if prompt_turns[i].self_contained:
                start = i
                break
        return [self._message(record, i, prompt_turns[i]) for i in range(start, len(prompt_turns))]

    def _message(self, record: Record, turn_index: int, turn: Turn) -> PromptMessage:
        pieces: List[str] = []
        declared = 0
        all_declared = True
        for part_index, part in enumerate(turn.parts):
            if isinstance(part, TextPart):
                pieces.append(part.text)
                all_declared = False
            elif isinstance(part, SyntheticPart):
                scope_key = self.source_id if part.scope == "trace" else (record.session_id or record.id)
                built = self.materializer.materialize(part, scope_key, tail_key=f"{record.id}:{turn_index}:{part_index}")
                pieces.append(built.text)
                declared += part.num_tokens
            elif isinstance(part, MediaPart):
                raise NotImplementedError("media parts are not sent by the workload generators yet")
        return PromptMessage(
            role=turn.role,
            text="\n".join(pieces),
            declared_tokens=declared if all_declared and pieces else None,
        )
