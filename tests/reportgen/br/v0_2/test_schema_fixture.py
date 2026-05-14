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
import pathlib

import yaml

from inference_perf.reportgen.br.v0_2.schema import VERSION, BenchmarkReportV02


FIXTURE = pathlib.Path(__file__).parent / "br_v0_2_example.yaml"


def test_schema_version() -> None:
    assert VERSION == "0.2"


def test_upstream_example_validates_against_vendored_schema() -> None:
    """Catches schema drift: if upstream changes BR0.2 in a way our vendored
    schema can't parse, this fails."""
    with FIXTURE.open() as f:
        data = yaml.safe_load(f)
    report = BenchmarkReportV02.model_validate(data)
    assert report.version == "0.2"
    assert report.run.uid == data["run"]["uid"]
    assert report.scenario is not None and report.scenario.stack is not None
    assert len(report.scenario.stack) == len(data["scenario"]["stack"])
