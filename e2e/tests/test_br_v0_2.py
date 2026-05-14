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
"""End-to-end test that the per-stage BR0.2 partial inference-perf emits
validates against the vendored schema after a real run against
llm-d-inference-sim, and merges cleanly with a downstream-supplied partial
into a full BR0.2 document.

Inference-perf is always responsible for the inference-perf-owned slice of a
BR0.2 report (``version``, ``run.uid``, ``run.time``, ``results``); a
composer adds stack/scenario/run-metadata. This test pins both halves of
that contract.
"""

from typing import Any, Dict

import pytest

from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.testdata import extract_tarball

from inference_perf.reportgen.br.v0_2.schema import BenchmarkReportV02


TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"


def _composer_partial(model_name: str) -> Dict[str, Any]:
    """The slice a downstream composer (e.g. llm-d-benchmark CLI) would
    contribute: stack/scenario/run metadata that inference-perf cannot speak
    to. Merged on top of inference-perf's partial to produce a full BR0.2.
    """
    return {
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


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Mirrors ``yq '. * load(...)'`` deep-merge on maps."""
    out: Dict[str, Any] = {}
    for k in set(a) | set(b):
        if k in a and k in b and isinstance(a[k], dict) and isinstance(b[k], dict):
            out[k] = _deep_merge(a[k], b[k])
        elif k in b:
            out[k] = b[k]
        else:
            out[k] = a[k]
    return out


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_br_v0_2_partial_emitted_and_mergeable() -> None:
    """A multi-stage run emits one BR0.2 partial per stage; each partial
    validates standalone (run.uid/results required fields populated) and
    yields a full BR0.2 when deep-merged with a composer-supplied partial."""
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)

    stages = [{"rate": 2, "duration": 5}, {"rate": 4, "duration": 5}]

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
                },
            }
        )

    assert result.success, f"Benchmark failed: {result.stdout}"
    assert result.reports, "No reports produced"

    partials = {
        name: contents for name, contents in result.reports.items() if name.startswith("inference-perf.partial.stage_")
    }
    assert len(partials) == len(stages), f"expected one partial per stage, got {list(partials)}"

    composer_partial = _composer_partial(model_name)

    for name, partial in partials.items():
        # Inference-perf-owned fields are present and well-formed.
        assert partial["version"] == "0.2", f"{name}: wrong schema version {partial.get('version')!r}"
        assert partial["run"]["uid"].startswith("inference-perf-stage-"), f"{name}: missing/malformed run.uid"
        assert "time" in partial["run"] and partial["run"]["time"]["duration"].startswith("PT"), f"{name}: missing run.time"
        assert partial["results"]["request_performance"]["aggregate"]["requests"]["total"] > 0

        # Partial alone validates as a BR0.2 document (run.uid + results both
        # populated; scenario is optional).
        BenchmarkReportV02.model_validate(partial)

        # Merging the composer's partial on top produces a complete BR0.2
        # with both producers' fields preserved.
        merged = _deep_merge(partial, composer_partial)
        parsed = BenchmarkReportV02.model_validate(merged)
        assert parsed.run.uid == partial["run"]["uid"]
        assert parsed.run.description == "br_v0_2 e2e"
        assert parsed.scenario is not None and parsed.scenario.stack is not None
        assert parsed.scenario.stack[0].metadata.label == "sim-0"
        assert parsed.results.request_performance is not None
        assert parsed.results.request_performance.aggregate is not None
