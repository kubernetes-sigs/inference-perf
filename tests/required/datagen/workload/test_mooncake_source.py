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

import json
from pathlib import Path
from typing import Sequence

import pytest

from inference_perf.workload import SyntheticPart
from inference_perf.workload.sources.mooncake import MooncakeSource

# Three lines of the FAST'25 conversation trace shape. The first two share
# their first 12 prefix blocks; the third has no blocks at all.
LINES = [
    {"timestamp": 0, "input_length": 6955, "output_length": 52, "hash_ids": list(range(46, 58)) + [2353, 2354]},
    {"timestamp": 3053, "input_length": 6472, "output_length": 26, "hash_ids": list(range(46, 58)) + [2366]},
    {"timestamp": 3100, "input_length": 40, "output_length": 5, "hash_ids": []},
]


def write_trace(tmp_path: Path, lines: Sequence[object]) -> Path:
    path = tmp_path / "conversation_trace.jsonl"
    path.write_text("\n".join(json.dumps(line) if not isinstance(line, str) else line for line in lines) + "\n")
    return path


# Loading the three-line trace gives three records, each a user turn holding
# one synthetic part (input_length tokens, the line's hash_ids as 512-token
# trace-scoped blocks) and an assistant turn holding output_length, plus one
# node per record sent at the recorded millisecond with no dependencies.
def test_load_shapes_records_and_nodes(tmp_path: Path) -> None:
    workload = MooncakeSource().load(write_trace(tmp_path, LINES))
    assert workload.source_id == "conversation_trace.jsonl"
    assert [r.id for r in workload.records] == ["0", "1", "2"]
    first = workload.records[0]
    assert [t.role for t in first.turns] == ["user", "assistant"]
    part = first.turns[0].parts[0]
    assert isinstance(part, SyntheticPart)
    assert part.num_tokens == 6955 and part.block_size == 512 and part.scope == "trace"
    assert part.block_ids == LINES[0]["hash_ids"]
    assert first.turns[1].output_tokens == 52
    nodes = workload.arrangement.nodes
    assert [n.send_at_ms for n in nodes] == [0, 3053, 3100]
    assert all(n.turn == 1 and n.depends_on == [] for n in nodes)
    assert not workload.arrangement.has_dependencies()
    third = workload.records[2].turns[0].parts[0]
    assert isinstance(third, SyntheticPart) and third.block_ids == [] and third.prefix_tokens == 0


# The two requests share 12 leading hash ids, so the first 12 * 512 = 6144
# tokens of each are declared shared, which is what the trace README says.
# Every token of both prompts is covered by a block (the last block is
# partial), so each part's prefix_tokens is its whole length.
def test_shared_hash_ids_declare_the_shared_prefix(tmp_path: Path) -> None:
    workload = MooncakeSource().load(write_trace(tmp_path, LINES))
    a = workload.records[0].turns[0].parts[0]
    b = workload.records[1].turns[0].parts[0]
    assert isinstance(a, SyntheticPart) and isinstance(b, SyntheticPart)
    assert a.block_ids[:12] == b.block_ids[:12]
    assert 12 * 512 <= min(a.prefix_tokens, b.prefix_tokens)
    assert a.prefix_tokens == 6955 and b.prefix_tokens == 6472


# Blank lines are skipped. A line that is not JSON, one missing hash_ids, one
# with a negative length and one whose hash_ids hold a string each fail with
# the file name and line number in the message.
def test_malformed_lines_fail_with_line_numbers(tmp_path: Path) -> None:
    ok = json.dumps(LINES[2])
    workload = MooncakeSource().load(write_trace(tmp_path, [ok, "", "   ", ok]))
    assert len(workload.records) == 2
    bad_cases = [
        ("{not json", "not JSON"),
        (json.dumps({"timestamp": 1, "input_length": 2, "output_length": 3}), "missing hash_ids"),
        (json.dumps({"timestamp": 1, "input_length": -2, "output_length": 3, "hash_ids": []}), "input_length"),
        (json.dumps({"timestamp": 1, "input_length": 2, "output_length": 3, "hash_ids": ["a"]}), "list of integers"),
    ]
    for line, message in bad_cases:
        path = write_trace(tmp_path, [ok, line])
        with pytest.raises(ValueError, match=message) as info:
            MooncakeSource().load(path)
        assert f"{path}:2" in str(info.value)
