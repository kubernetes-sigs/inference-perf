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
"""
A streamed response with one output token still has a TTFT (#942).

The sim runs in echo mode, so the one-word prompt "Testing" comes back as a
single chunk the client tokenizer reads as one token.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.net import get_free_port
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"
SIM_TTFT_SEC = 0.2


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
# 10 streamed requests whose response is the one token "Testing", first token held for 200ms.
# Every request, the summary and the llm-d benchmark report carry a TTFT of at least 0.2s. TPOT stays None.
async def test_single_token_stream_reports_ttft(tmp_path: Path) -> None:
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text(
        json.dumps([{"id": "1", "conversations": [{"from": "human", "value": "Testing"}, {"from": "gpt", "value": "OK"}]}])
    )
    model_path = extract_tarball(TEST_MODEL_TARBALL)

    async with LLMDInferenceSimRunner(
        TEST_MODEL_NAME, "--mode", "echo", "--time-to-first-token", str(int(SIM_TTFT_SEC * 1000)), port=get_free_port()
    ) as sim:
        result = await run_benchmark_minimal(
            {
                "data": {"type": "shareGPT", "path": str(dataset)},
                "load": {"type": "constant", "stages": [{"rate": 5, "duration": 2}]},
                "api": {"type": "completion", "streaming": True},
                "server": {
                    "type": "vllm",
                    "model_name": TEST_MODEL_NAME,
                    "base_url": f"http://{sim.host}:{sim.port}",
                    "ignore_eos": True,
                },
                "tokenizer": {"pretrained_model_name_or_path": str(model_path)},
                "report": {
                    "request_lifecycle": {
                        "summary": True,
                        "per_stage": True,
                        "per_request": True,
                        "per_request_fields": {"computed_metrics": True},
                    },
                },
            }
        )

    assert result.success, f"Benchmark failed: {result.stdout}"
    records: list[dict[str, Any]] = result.reports["per_request_lifecycle_metrics.json"]
    assert len(records) == 10

    for record in records:
        response_metrics = record["info"]["response_metrics"]
        # Guards the premise, so the TTFT checks below cannot pass on a two-token response.
        assert len(response_metrics["output_token_times"]) == 1
        assert response_metrics["server_usage"]["completion_tokens"] == 1

        computed = record["computed_metrics"]
        assert computed["time_to_first_token"] is not None
        assert SIM_TTFT_SEC <= computed["time_to_first_token"] <= computed["request_latency"]
        assert computed["time_per_output_token"] is None

    summary_ttft = result.reports["summary_lifecycle_metrics.json"]["successes"]["latency"]["time_to_first_token"]
    assert summary_ttft is not None and summary_ttft["min"] >= SIM_TTFT_SEC

    partial = next(report for name, report in result.reports.items() if name.startswith("inference-perf.partial.stage_"))
    br_ttft = partial["results"]["request_performance"]["aggregate"]["latency"].get("time_to_first_token")
    assert br_ttft is not None and br_ttft["min"] >= SIM_TTFT_SEC
