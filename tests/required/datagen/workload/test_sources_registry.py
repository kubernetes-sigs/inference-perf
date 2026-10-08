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

import pytest
from pydantic import ValidationError

from inference_perf.config import DataConfig, DataGenType, WorkloadReplayConfig
from inference_perf.workload import Arrangement, Node, Record, SyntheticPart, Turn
from inference_perf.workload.sources import SOURCES, Workload, WorkloadSource, load_workload


# A source registered as "unit" with a default block size of 8: loading
# through the registry without a block size gives 8, with 32 gives 32, and
# the source's load() runs on the path it was given.
def test_registered_source_is_loaded_by_name(tmp_path: Path) -> None:
    class UnitSource(WorkloadSource):
        format = "unit"
        default_block_size = 8

        def load(self, path: Path) -> Workload:
            record = Record(
                id="0",
                turns=[
                    Turn(role="user", parts=[SyntheticPart(num_tokens=self.block_size, block_size=self.block_size)]),
                    Turn(role="assistant", output_tokens=1),
                ],
            )
            return Workload(
                source_id=path.name, records=[record], arrangement=Arrangement(nodes=[Node(id="0", record_id="0", turn=1)])
            )

    SOURCES["unit"] = UnitSource
    try:
        by_default = load_workload("unit", str(tmp_path / "t.jsonl"))
        sized = load_workload("unit", str(tmp_path / "t.jsonl"), block_size=32)
    finally:
        del SOURCES["unit"]
    part = by_default.records[0].turns[0].parts[0]
    assert isinstance(part, SyntheticPart) and part.block_size == 8
    part = sized.records[0].turns[0].parts[0]
    assert isinstance(part, SyntheticPart) and part.block_size == 32
    assert by_default.source_id == "t.jsonl"


# An unregistered format fails naming the formats that are registered.
def test_unknown_format_lists_registered_ones() -> None:
    with pytest.raises(ValueError, match="registered formats"):
        load_workload("NoSuchFormat", "x.jsonl")


# data.type workload_replay without data.workload is a config error, and
# data.workload needs both a format and a file.
def test_workload_replay_config_requirements() -> None:
    with pytest.raises(ValidationError, match="data.workload"):
        DataConfig(type=DataGenType.WorkloadReplay)
    with pytest.raises(ValidationError):
        WorkloadReplayConfig.model_validate({"format": "Mooncake"})
    config = DataConfig(type=DataGenType.WorkloadReplay, workload=WorkloadReplayConfig(format="Mooncake", file="t.jsonl"))
    assert config.workload is not None and config.workload.block_size is None and config.workload.session is None
