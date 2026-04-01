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
import pytest

from utils.benchmark import run_benchmark_minimal


@pytest.mark.asyncio
async def test_simple_mock_client_benchmark():
    result = await run_benchmark_minimal("e2e/configs/e2e_simple_mock_client.yaml", timeout_sec=None)
    assert result.success, "Benchmark failed"
    assert result.reports, "No reports generated from benchmark"
    assert result.reports["per_request_lifecycle_metrics.json"], "Missing requests report"
    assert result.reports["stage_0_lifecycle_metrics.json"], "Missing stage report"
    assert result.reports["summary_lifecycle_metrics.json"], "Missing summary report"

    requests_report = result.reports["per_request_lifecycle_metrics.json"]
    stage_report = result.reports["stage_0_lifecycle_metrics.json"]
    summary_report = result.reports["summary_lifecycle_metrics.json"]

    assert len(requests_report) == 10, "the number of requests should be 10"
    assert stage_report["load_summary"]["achieved_rate"] > 1 or stage_report["load_summary"]["achieved_rate"] == pytest.approx(
        1, abs=0.2
    ), "the achieved rate should be close to 1.0"
    assert summary_report["successes"]["count"] == 10
