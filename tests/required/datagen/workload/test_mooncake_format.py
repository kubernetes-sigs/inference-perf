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
from typing import List, Sequence

import pytest

from inference_perf.workload import Session, TextPart, TextSpec, TraceLocation, format_class, read_sessions
from inference_perf.workload.formats.mooncake import MooncakeFormat

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


# The sessions of a trace file, read as Mooncake.
def load(path: Path) -> List[Session]:
    return read_sessions(MooncakeFormat(), TraceLocation(files=[str(path)]))


# The text spec that is the prompt of request `n`.
def prompt_part(sessions: Sequence[Session], n: int) -> TextSpec:
    part = sessions[n].records[0].messages[0].parts[0]
    assert isinstance(part, TextPart) and part.spec is not None
    return part.spec


# Reading the three-line trace gives three sessions of one request each,
# named by their place in the file. A request is a user message holding one
# text spec (input_length tokens, the line's hash_ids as 512-token
# trace-scoped blocks) and asks for output_length of output, sent at
# the recorded millisecond and waiting for nothing.
def test_each_line_is_a_request_of_its_own(tmp_path: Path) -> None:
    sessions = load(write_trace(tmp_path, LINES))
    assert [s.id for s in sessions] == ["0", "1", "2"]
    assert [[r.id for r in s.records] for s in sessions] == [["0"], ["1"], ["2"]]
    first = sessions[0].records[0]
    assert [m.role for m in first.messages] == ["user"] and first.reply == []
    part = prompt_part(sessions, 0)
    assert part.num_tokens == 6955 and part.block_size == 512 and part.scope == "trace"
    assert part.block_ids == LINES[0]["hash_ids"]
    assert first.output_tokens == 52
    assert [s.delivery.of(s.records[0].id).start_ms for s in sessions] == [0, 3053, 3100]
    assert [s.delivery.of(s.records[0].id).after for s in sessions] == [[], [], []]
    third = prompt_part(sessions, 2)
    assert third.block_ids == [] and third.prefix_tokens == 0


# A trace that starts 27482 ms in is replayed from zero: its two requests,
# recorded 3053 ms apart, are sent at 0 and 3053.
def test_the_first_request_is_sent_at_zero(tmp_path: Path) -> None:
    late = [dict(LINES[0], timestamp=27482), dict(LINES[1], timestamp=30535)]
    sessions = load(write_trace(tmp_path, late))
    assert [s.delivery.of(s.records[0].id).start_ms for s in sessions] == [0, 3053]


# The two requests share 12 leading hash ids, so the first 12 * 512 = 6144
# tokens of each are declared shared, which is what the trace README says.
# Every token of both prompts is covered by a block (the last block is
# partial), so each part's prefix_tokens is its whole length.
def test_shared_hash_ids_declare_the_shared_prefix(tmp_path: Path) -> None:
    sessions = load(write_trace(tmp_path, LINES))
    a, b = prompt_part(sessions, 0), prompt_part(sessions, 1)
    assert a.block_ids[:12] == b.block_ids[:12]
    assert 12 * 512 <= min(a.prefix_tokens, b.prefix_tokens)
    assert a.prefix_tokens == 6955 and b.prefix_tokens == 6472


# Blank lines are skipped. A line that is not JSON, one missing hash_ids, one
# with a negative length and one whose hash_ids hold a string each fail with
# the file name and line number in the message.
def test_malformed_lines_fail_with_line_numbers(tmp_path: Path) -> None:
    ok = json.dumps(LINES[2])
    assert len(load(write_trace(tmp_path, [ok, "", "   ", ok]))) == 2
    bad_cases = [
        ("{not json", "not JSON"),
        (json.dumps({"timestamp": 1, "input_length": 2, "output_length": 3}), "missing hash_ids"),
        (json.dumps({"timestamp": 1, "input_length": -2, "output_length": 3, "hash_ids": []}), "input_length"),
        (json.dumps({"timestamp": 1, "input_length": 2, "output_length": 3, "hash_ids": ["a"]}), "list of integers"),
    ]
    for line, message in bad_cases:
        path = write_trace(tmp_path, [ok, line])
        with pytest.raises(ValueError, match=message) as info:
            load(path)
        assert f"{path}:2" in str(info.value)


# A config that says `format: Mooncake` gets this format.
def test_the_format_is_built_in_under_its_name() -> None:
    assert format_class("Mooncake") is MooncakeFormat
