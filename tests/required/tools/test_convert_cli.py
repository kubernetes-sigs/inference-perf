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
"""Exit behavior of ``inference-perf-convert``: 0 converted, 2 refused with

no config written, 1 usage error. The emitted file must load back through
``read_config`` and carry the conversion record in its comment block.
"""

from pathlib import Path

import pytest

from inference_perf.config import read_config
from inference_perf.tools.convert import run


def test_convert_writes_a_loadable_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "converted.yaml"
    argv = [
        "vllm-bench",
        "--peer-version",
        "v0.10.0",
        "-o",
        str(out),
        "--",
        "--model",
        "google/gemma-3-270m",
        "--dataset-name",
        "random",
        "--num-prompts",
        "600",
        "--request-rate",
        "40",
        "--ignore-eos",
    ]
    assert run(argv) == 0
    text = out.read_text()
    assert text.startswith("# Converted by inference-perf-convert (#755)")
    assert "# assumptions:" in text
    assert "# NO EQUIVALENT:" in text
    config = read_config(str(out))
    assert config.server is not None
    assert config.server.ignore_eos is True


def test_refusal_exits_2_and_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "converted.yaml"
    argv = [
        "vllm-bench",
        "--peer-version",
        "v0.10.0",
        "-o",
        str(out),
        "--",
        "--model",
        "m",
        "--temperature",
        "0.7",
        "--dataset-name",
        "sonnet",
    ]
    assert run(argv) == 2
    assert not out.exists()
    err = capsys.readouterr().err
    assert "REFUSE --temperature:" in err
    assert "REFUSE --dataset-name:" in err


def test_usage_error_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    # Missing required --model is a peer usage error, not a refusal.
    argv = ["vllm-bench", "--peer-version", "v0.10.0", "--", "--dataset-name", "random"]
    assert run(argv) == 1
    assert "error:" in capsys.readouterr().err


def test_aiperf_stdout_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = [
        "aiperf-profile",
        "--peer-version",
        "v0.12.0",
        "--",
        "--model",
        "m",
        "--endpoint-type",
        "completions",
        "--streaming",
        "--osl",
        "64",
    ]
    assert run(argv) == 0
    out = capsys.readouterr().out
    assert "aiperf profile @ v0.12.0" in out
    saved = tmp_path / "aiperf.yaml"
    saved.write_text(out)
    config = read_config(str(saved))
    assert config.api.streaming is True
