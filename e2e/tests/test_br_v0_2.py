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
"""End-to-end test that BR0.2 reports validate against the vendored schema
after a real inference-perf run against llm-d-inference-sim."""

import tempfile
from pathlib import Path

import pytest
import yaml

from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.testdata import extract_tarball

from inference_perf.reportgen.br.v0_2.schema import BenchmarkReportV02


TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"


def _write_partial_report(tmp_dir: Path, model_name: str) -> Path:
    """Write a partial BR0.2 report describing the simulated stack to a temp
    file and return its path."""
    partial = {
        "run": {
            "eid": "br-v0-2-e2e",
            "description": "br_v0_2 e2e",
            "keywords": ["e2e", "smoke"],
        },
        "scenario": {
            "stack": [
                {
                    "metadata": {"label": "sim-0", "schema_version": "0.0.1", "cfg_id": "sim"},
                    "standardized": {
                        "kind": "inference_engine",
                        "tool": "llm-d-inference-sim",
                        "tool_version": "test",
                        "role": "replica",
                        "replicas": 1,
                        "model": {"name": model_name},
                        "accelerator": {
                            "model": "cpu",
                            "count": 1,
                            "parallelism": {"tp": 1, "dp": 1, "dp_local": 1, "workers": 1},
                        },
                    },
                    "native": {"args": {}, "envars": {}},
                }
            ],
        },
    }
    path = tmp_dir / "partial_report.yaml"
    path.write_text(yaml.safe_dump(partial))
    return path


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_br_v0_2_emitted_and_valid() -> None:
    """Two-stage run with a partial report configured should produce one BR0.2
    report per stage, and each report must validate against the vendored
    schema and carry the partial's stack/run metadata."""
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)

    stages = [{"rate": 2, "duration": 5}, {"rate": 4, "duration": 5}]

    with tempfile.TemporaryDirectory() as tmp_dir:
        partial_path = _write_partial_report(Path(tmp_dir), model_name)

        async with LLMDInferenceSimRunner(model_name, port=18001) as sim:
            result = await run_benchmark_minimal(
                {
                    "data": {"type": "mock"},
                    "load": {"type": "constant", "stages": stages, "num_workers": 2},
                    "api": {"type": "completion", "streaming": True},
                    "server": {
                        "type": "vllm",
                        "model_name": model_name,
                        "base_url": f"http://{sim.host}:{sim.port}",
                        "ignore_eos": True,
                    },
                    "tokenizer": {"pretrained_model_name_or_path": str(model_path)},
                    "report": {
                        "request_lifecycle": {"summary": True, "per_stage": True},
                        "br_v0_2": {
                            "partial_report": {
                                "local": {"path": str(partial_path)},
                            },
                        },
                    },
                }
            )

    assert result.success, f"Benchmark failed: {result.stdout}"
    assert result.reports, "No reports produced"

    br_reports = {name: contents for name, contents in result.reports.items() if "benchmark_report_v0_2" in name}
    assert len(br_reports) == len(stages), f"expected one BR0.2 per stage, got {list(br_reports)}"

    for contents in br_reports.values():
        parsed = BenchmarkReportV02.model_validate(contents)
        assert parsed.scenario is not None and parsed.scenario.stack is not None
        assert parsed.scenario.stack[0].metadata.label == "sim-0"
        assert parsed.results.request_performance is not None
        assert parsed.results.request_performance.aggregate is not None
        assert parsed.results.request_performance.aggregate.requests is not None
        assert parsed.results.request_performance.aggregate.requests.total > 0
        assert parsed.run.description == "br_v0_2 e2e"
