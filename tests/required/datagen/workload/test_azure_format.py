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

from pathlib import Path
from typing import List, Optional, Sequence

import pytest

from inference_perf.workload import Session, TextPart, TraceLocation, format_class, read_sessions
from inference_perf.workload.formats.azure import AzurePublicDatasetFormat

HEADER = "TIMESTAMP,ContextTokens,GeneratedTokens"
LINES = [
    "2023-11-16 18:15:46.6805900,374,44",
    "2023-11-16 18:15:50.9951690,396,109",
    "2023-11-16 18:15:51.2224670,879,55",
]


def write_trace(tmp_path: Path, lines: Sequence[str]) -> Path:
    path = tmp_path / "AzureLLMInferenceTrace_conv.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


# The sessions of a trace file, read as an Azure trace.
def load(path: Path) -> List[Session]:
    return read_sessions(AzurePublicDatasetFormat(), TraceLocation(files=[str(path)]))


# When each request of a trace was sent, in ms from the earliest.
def send_times(sessions: Sequence[Session]) -> List[Optional[int]]:
    return [s.delivery.of(s.records[0].id).start_ms for s in sessions]


# Three dataset lines behind the header become three sessions of one request
# each: a user message holding one text spec of ContextTokens with no
# prefix blocks, asking for GeneratedTokens of output, sent at 0, 4314
# and 4541 ms (offsets from the earliest timestamp, sub-ms digits dropped)
# and waiting for nothing.
def test_each_line_is_a_request_of_its_own(tmp_path: Path) -> None:
    sessions = load(write_trace(tmp_path, [HEADER, *LINES]))
    assert [s.id for s in sessions] == ["0", "1", "2"]
    first = sessions[0].records[0]
    assert [m.role for m in first.messages] == ["user"] and first.reply == []
    part = first.messages[0].parts[0]
    assert isinstance(part, TextPart) and part.spec is not None
    assert part.spec.num_tokens == 374 and part.spec.block_ids == []
    assert first.output_tokens == 44
    assert send_times(sessions) == [0, 4314, 4541]
    assert [s.delivery.of(s.records[0].id).after for s in sessions] == [[], [], []]


# The header is optional, and the ISO spellings the dataset's re-exports use
# (a `T` separator, a `Z` suffix, no fraction) parse to the same offsets.
def test_header_is_optional_and_iso_variants_parse(tmp_path: Path) -> None:
    assert send_times(load(write_trace(tmp_path, LINES))) == [0, 4314, 4541]
    iso = ["2023-11-16T18:15:46Z,1,1", "2023-11-16T18:15:47.5Z,1,1", '"2023-11-16 18:15:48",1,1']
    assert send_times(load(write_trace(tmp_path, iso))) == [0, 1500, 2000]


# Offsets are from the earliest line, wherever it sits: with the first two
# lines swapped the later one is sent at 4314 ms and the earlier one at 0.
def test_offsets_are_from_the_earliest_line(tmp_path: Path) -> None:
    assert send_times(load(write_trace(tmp_path, [LINES[1], LINES[0]]))) == [4314, 0]


# Blank lines are skipped. A line with two fields, one with a bad timestamp
# and one with a negative token count each fail with the file name and line
# number in the message.
def test_malformed_lines_fail_with_line_numbers(tmp_path: Path) -> None:
    ok = LINES[0]
    assert len(load(write_trace(tmp_path, [ok, "", "  ", ok]))) == 2
    bad_cases = [
        ("2023-11-16 18:15:46.68,374", "got 2 fields"),
        ("yesterday,374,44", "bad timestamp"),
        ("2023-11-16 18:15:46.68,-374,44", "ContextTokens"),
    ]
    for line, message in bad_cases:
        path = write_trace(tmp_path, [ok, line])
        with pytest.raises(ValueError, match=message) as info:
            load(path)
        assert f"{path}:2" in str(info.value)


# A config that says `format: AzurePublicDataset` gets this format.
def test_the_format_is_built_in_under_its_name() -> None:
    assert format_class("AzurePublicDataset") is AzurePublicDatasetFormat
