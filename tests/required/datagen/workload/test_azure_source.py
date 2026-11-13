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
from typing import Sequence

import pytest

from inference_perf.workload import SyntheticPart
from inference_perf.workload.sources.azure import AzurePublicDatasetSource

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


# Three dataset lines behind the header become three records, each a user
# turn holding one synthetic part of ContextTokens with no prefix blocks and
# an assistant turn of GeneratedTokens, and three independent nodes sent at
# 0, 4314 and 4541 ms (offsets from the first line's timestamp, sub-ms
# digits dropped).
def test_load_shapes_records_and_nodes(tmp_path: Path) -> None:
    workload = AzurePublicDatasetSource().load(write_trace(tmp_path, [HEADER, *LINES]))
    assert workload.source_id == "AzureLLMInferenceTrace_conv.csv"
    assert [r.id for r in workload.records] == ["0", "1", "2"]
    first = workload.records[0]
    assert [t.role for t in first.turns] == ["user", "assistant"]
    part = first.turns[0].parts[0]
    assert isinstance(part, SyntheticPart) and part.num_tokens == 374 and part.block_ids == []
    assert first.turns[1].output_tokens == 44
    assert [n.send_at_ms for n in workload.arrangement.nodes] == [0, 4314, 4541]
    assert not workload.arrangement.has_dependencies()


# The header is optional, and the ISO spellings the dataset's re-exports use
# (a `T` separator, a `Z` suffix, no fraction) parse to the same offsets.
def test_header_is_optional_and_iso_variants_parse(tmp_path: Path) -> None:
    no_header = AzurePublicDatasetSource().load(write_trace(tmp_path, LINES))
    assert [n.send_at_ms for n in no_header.arrangement.nodes] == [0, 4314, 4541]
    iso = ["2023-11-16T18:15:46Z,1,1", "2023-11-16T18:15:47.5Z,1,1", '"2023-11-16 18:15:48",1,1']
    assert [n.send_at_ms for n in AzurePublicDatasetSource().load(write_trace(tmp_path, iso)).arrangement.nodes] == [
        0,
        1500,
        2000,
    ]


# Blank lines are skipped. A line with two fields, one with a bad timestamp
# and one with a negative token count each fail with the file name and line
# number in the message.
def test_malformed_lines_fail_with_line_numbers(tmp_path: Path) -> None:
    ok = LINES[0]
    assert len(AzurePublicDatasetSource().load(write_trace(tmp_path, [ok, "", "  ", ok])).records) == 2
    bad_cases = [
        ("2023-11-16 18:15:46.68,374", "got 2 fields"),
        ("yesterday,374,44", "bad timestamp"),
        ("2023-11-16 18:15:46.68,-374,44", "ContextTokens"),
    ]
    for line, message in bad_cases:
        path = write_trace(tmp_path, [ok, line])
        with pytest.raises(ValueError, match=message) as info:
            AzurePublicDatasetSource().load(path)
        assert f"{path}:2" in str(info.value)
