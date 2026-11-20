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

import pytest
import yaml
from pydantic import ValidationError

from inference_perf.config import RequestLifecycleMetricsReportConfig, read_config


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


# group_by_labels=[["tenant"], ["tenant", "priority"]] is accepted and kept as given.
def test_group_by_labels_accepts_distinct_groups() -> None:
    config = RequestLifecycleMetricsReportConfig(group_by_labels=[["tenant"], ["tenant", "priority"]])
    assert config.group_by_labels == [["tenant"], ["tenant", "priority"]]


# A group with no label keys is rejected: it would slice nothing.
def test_group_by_labels_rejects_empty_group() -> None:
    with pytest.raises(ValidationError, match="at least one label"):
        RequestLifecycleMetricsReportConfig(group_by_labels=[["tenant"], []])


# A group naming the same key twice, ["tenant", "tenant"], is rejected.
def test_group_by_labels_rejects_repeated_key_in_group() -> None:
    with pytest.raises(ValidationError, match="repeats a label"):
        RequestLifecycleMetricsReportConfig(group_by_labels=[["tenant", "tenant"]])


# The same group listed twice is rejected: both would write the same report files.
def test_group_by_labels_rejects_repeated_group() -> None:
    with pytest.raises(ValidationError, match="listed more than once"):
        RequestLifecycleMetricsReportConfig(group_by_labels=[["tenant"], ["tenant"]])
