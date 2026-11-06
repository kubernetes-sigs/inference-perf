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
"""Validity rules for ``inference_perf.config.reportgen``."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from inference_perf.config import read_config


def test_goodput_constraints_parsing() -> None:
    config_content = {
        "report": {
            "goodput": {
                "constraints": {
                    "ttft": 0.5,
                    "tpot": 0.1,
                },
            }
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config_content, tmp)
        tmp_path = tmp.name

    try:
        config = read_config(tmp_path)
        assert config.report.goodput is not None
        assert config.report.goodput.constraints["ttft"] == 0.5
        assert config.report.goodput.constraints["tpot"] == 0.1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# config.yaml points report.benchmark_report.base_file at a base that sets run.uid,
# which inference-perf writes. read_config fails before any load is sent, naming the field.
def test_read_config_rejects_a_base_report_that_sets_an_owned_field(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"run": {"uid": "mine"}}))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"report": {"benchmark_report": {"base_file": str(base)}}}))

    with pytest.raises(ValidationError, match=r"owns: run\.uid"):
        read_config(str(config))


# The same config with a base that only sets run.description loads, and keeps the path.
def test_read_config_accepts_a_fillable_base_report(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"run": {"description": "nightly"}}))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"report": {"benchmark_report": {"base_file": str(base)}}}))

    assert read_config(str(config)).report.benchmark_report.base_file == str(base)
