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
import importlib.resources

import yaml

from inference_perf.reportgen.br.v0_2.schema import VERSION, BenchmarkReportV021


# The example report shipped inside the llmd-benchmark-report wheel.
EXAMPLE = importlib.resources.files("llmd_benchmark_report") / "br_v0_2_example.yaml"


# The facade exposes the schema line inference-perf emits. Expects "0.2.1".
def test_schema_version() -> None:
    assert VERSION == "0.2.1"


# Loads the package's example report and validates it through the facade.
# Expects version "0.2.1" plus the same run uid and stack length as the YAML.
def test_package_example_validates_through_facade() -> None:
    data = yaml.safe_load(EXAMPLE.read_text())
    report = BenchmarkReportV021.model_validate(data)
    assert report.version == "0.2.1"
    assert report.run.uid == data["run"]["uid"]
    assert report.scenario is not None and report.scenario.stack is not None
    assert len(report.scenario.stack) == len(data["scenario"]["stack"])
