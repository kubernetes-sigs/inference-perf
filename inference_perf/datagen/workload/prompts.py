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
a node is every turn before its target, with every synthetic part
materialized under the scope the part asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from inference_perf.workload.materialize import BlockTextMaterializer
from inference_perf.workload.record import MediaPart, Record, SyntheticPart, TextPart, Turn


def _unwrapped(text: str) -> str:
    return text


@dataclass
class PromptMessage:
    role: str
    text: str
    # Token count the record declared for this message, when every part
    # declared one; None when the count has to be measured.
    declared_tokens: Optional[int]


class RecordPrompter:
    def __init__(self, source_id: str, materializer: BlockTextMaterializer, special_tokens_in_lengths: bool = True) -> None:
        self.source_id = source_id
        self.materializer = materializer
        self.special_tokens_in_lengths = special_tokens_in_lengths

    def messages(self, record: Record, turn: int) -> List[PromptMessage]:
        """The messages that elicit `record.turns[turn]`."""
        return [self.message(record, i) for i in range(turn)]

    def message(self, record: Record, turn_index: int) -> PromptMessage:
        """One turn as a message, whatever its role."""
        turn: Turn = record.turns[turn_index]
        pieces: List[str] = []
        declared = 0
        all_declared = True
        for part_index, part in enumerate(turn.parts):
            if isinstance(part, TextPart):
                pieces.append(part.text)
                all_declared = False
            elif isinstance(part, SyntheticPart):
                scope_key = self.source_id if part.scope == "trace" else (record.session_id or record.id)
                if part.num_tokens == len(part.block_ids) * part.block_size:
                    # Named entirely by its blocks: the same part is the same
                    # text in every record of the scope, even when landing
                    # the exact count needs a tail.
                    tail_key = f"{scope_key}:blocks:{part.block_size}:{','.join(map(str, part.block_ids))}"
                else:
                    tail_key = f"{record.id}:{turn_index}:{part_index}"
                # An identity wrap lands the count without special tokens.
                wrap_fn = None if self.special_tokens_in_lengths else _unwrapped
                built = self.materializer.materialize(part, scope_key, tail_key=tail_key, wrap_fn=wrap_fn)
                pieces.append(built.text)
                declared += part.num_tokens
            elif isinstance(part, MediaPart):
                raise NotImplementedError("media parts are not sent by the workload generators yet")
        return PromptMessage(
            role=turn.role,
            text="\n".join(pieces),
            declared_tokens=declared if all_declared and pieces else None,
        )
