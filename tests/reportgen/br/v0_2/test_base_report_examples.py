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
"""Happy-path tests over the example configs in base_report_examples/.

Each subdirectory is one example: a config.yaml that sets
report.benchmark_report.base_file, and the base.yaml it points at. Adding an
example needs no test code.
"""

from pathlib import Path

import pytest

from inference_perf.config import read_config
from inference_perf.config.reportgen.config import BenchmarkReportConfig
from inference_perf.reportgen.br.v0_2 import apply_base_report, load_base_report

from .base_report_helpers import generate_reports, partial_validation_errors, partials

EXAMPLES_DIR = Path(__file__).parent / "base_report_examples"
EXAMPLES = sorted(path.name for path in EXAMPLES_DIR.iterdir() if path.is_dir())


# Guards the parametrized test below against an empty or moved examples directory.
def test_examples_directory_has_examples() -> None:
    assert len(EXAMPLES) >= 3, EXAMPLES


# Loads each example's config.yaml the way the CLI does and runs two stages
# (4 requests, then 2). Each stage's partial equals the partial from the same run
# without a base, with base.yaml merged on top, and validation.json stays clean.
@pytest.mark.parametrize("example", EXAMPLES)
def test_example_config_builds_every_partial_on_its_base(example: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(EXAMPLES_DIR / example)
    config = read_config("config.yaml")
    base_file = config.report.benchmark_report.base_file
    assert base_file is not None
    base = load_base_report(base_file)

    reports = generate_reports(config.report)
    bare = partials(generate_reports(config.report.model_copy(update={"benchmark_report": BenchmarkReportConfig()})))

    stages = partials(reports)
    assert sorted(stages) == [0, 1]
    assert stages[0]["results"]["request_performance"]["aggregate"]["requests"]["total"] == 4
    assert stages[1]["results"]["request_performance"]["aggregate"]["requests"]["total"] == 2
    for stage_id, partial in stages.items():
        expected = apply_base_report(bare[stage_id], base)
        # Generated per call, so the two runs differ here and nowhere else.
        expected["run"]["uid"] = partial["run"]["uid"]
        if "eid" not in base.get("run", {}):
            expected["run"]["eid"] = partial["run"]["eid"]
        assert partial == expected
    assert partial_validation_errors(reports) == []
