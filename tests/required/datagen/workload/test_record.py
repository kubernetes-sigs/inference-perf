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

import pytest
from pydantic import ValidationError

from inference_perf.workload import Arrangement, MediaPart, Node, Record, SyntheticPart, TextPart, Turn
from inference_perf.workload.sources.base import Workload


# A synthetic part of 1000 tokens with 512-token blocks has room for two
# blocks (the second one partial). Three block ids are rejected; two are
# accepted and cover min(2 * 512, 1000) = 1000 prefix tokens.
def test_synthetic_part_blocks_must_fit() -> None:
    with pytest.raises(ValidationError, match="cannot fit"):
        SyntheticPart(num_tokens=1000, block_ids=[1, 2, 3], block_size=512)
    part = SyntheticPart(num_tokens=1000, block_ids=[1, 2], block_size=512)
    assert part.prefix_tokens == 1000
    assert SyntheticPart(num_tokens=1000, block_ids=[1], block_size=512).prefix_tokens == 512


# output_tokens is the reference output length, so it only makes sense on an
# assistant turn. A user turn carrying it is rejected; an assistant turn with
# no parts and output_tokens=52 is the normal shape for a lengths-only trace.
def test_output_tokens_only_on_assistant_turns() -> None:
    with pytest.raises(ValidationError, match="assistant"):
        Turn(role="user", output_tokens=5)
    turn = Turn(role="assistant", output_tokens=52)
    assert turn.parts == []


# A record with a user turn, an assistant turn and a second user turn has
# exactly one assistant turn, at index 1. Zero turns is rejected.
def test_record_assistant_turn_indices() -> None:
    record = Record(
        id="r",
        turns=[
            Turn(role="user", parts=[TextPart(text="hi")]),
            Turn(role="assistant", parts=[TextPart(text="hello")], output_tokens=1),
            Turn(role="user", parts=[TextPart(text="bye")]),
        ],
    )
    assert record.assistant_turn_indices() == [1]
    with pytest.raises(ValidationError):
        Record(id="empty", turns=[])


# Dumping a record with all three part types and validating the dump gives
# back the same record, with each part's concrete type restored from `type`.
def test_record_round_trips_through_dump() -> None:
    record = Record(
        id="r",
        session_id="s",
        metadata={"origin": "test"},
        turns=[
            Turn(
                role="user",
                parts=[
                    TextPart(text="describe"),
                    MediaPart(kind="image", ref="file:///cat.png"),
                    SyntheticPart(num_tokens=600, block_ids=[7, 8], block_size=512, scope="session"),
                ],
            ),
            Turn(role="assistant", output_tokens=20),
        ],
    )
    restored = Record.model_validate(record.model_dump())
    assert restored == record
    assert [type(p) for p in restored.turns[0].parts] == [TextPart, MediaPart, SyntheticPart]


# The schema is strict: a part with a key it does not define is rejected
# rather than silently dropped.
def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TextPart.model_validate({"type": "text", "text": "x", "tokens": 3})


# Two nodes with the same id, or a node depending on an id no node has, are
# rejected. A valid two-node chain reports that it has dependencies; a single
# timestamped node does not.
def test_arrangement_ids_and_dependencies() -> None:
    with pytest.raises(ValidationError, match="unique"):
        Arrangement(nodes=[Node(id="a", record_id="r", turn=1), Node(id="a", record_id="r", turn=1)])
    with pytest.raises(ValidationError, match="unknown node"):
        Arrangement(nodes=[Node(id="a", record_id="r", turn=1, depends_on=["zzz"])])
    chain = Arrangement(nodes=[Node(id="a", record_id="r", turn=1), Node(id="b", record_id="r", turn=3, depends_on=["a"])])
    assert chain.has_dependencies()
    assert not Arrangement(nodes=[Node(id="a", record_id="r", turn=1, send_at_ms=10)]).has_dependencies()


# A workload checks its nodes against its records: a node naming a record
# that is not there, or pointing at a user turn instead of an assistant turn,
# is rejected at construction.
def test_workload_nodes_must_elicit_assistant_turns() -> None:
    record = Record(id="r", turns=[Turn(role="user", parts=[TextPart(text="q")]), Turn(role="assistant", output_tokens=3)])
    Workload(source_id="t", records=[record], arrangement=Arrangement(nodes=[Node(id="n", record_id="r", turn=1)]))
    with pytest.raises(ValueError, match="unknown record"):
        Workload(source_id="t", records=[record], arrangement=Arrangement(nodes=[Node(id="n", record_id="x", turn=1)]))
    with pytest.raises(ValueError, match="assistant turn"):
        Workload(source_id="t", records=[record], arrangement=Arrangement(nodes=[Node(id="n", record_id="r", turn=0)]))
