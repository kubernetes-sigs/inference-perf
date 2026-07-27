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

import yaml

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
