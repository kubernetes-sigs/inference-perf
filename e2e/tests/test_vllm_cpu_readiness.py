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
"""The vLLM profile's readiness route against a real vLLM (#979).

The benchmark starts first, against a port where vLLM is still booting: the
run must wait on ``/health``, then auto-detect the model and complete every
request. Against an external server (``E2E_VLLM_BASE_URL``) the server is
already up, so the run only checks that the route the profile names exists and
answers 200.
"""

import asyncio

import pytest

from utils.accuracy import assert_successful_run
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.vllm_server import VLLMServerRunner

pytestmark = pytest.mark.xdist_group(name="vllm-cpu-server")

RATE = 2
DURATION = 3


# Benchmark launched before the server boots, no model_name -> the run waits on
# /health, infers the model and completes RATE * DURATION requests.
@pytest.mark.asyncio
@pytest.mark.skipif(not VLLMServerRunner.is_available(), reason="no vLLM server or executable available")
async def test_run_waits_for_vllm_health_before_first_stage() -> None:
    runner = VLLMServerRunner(port=get_free_port())
    benchmark = asyncio.create_task(
        run_benchmark_minimal(
            {
                "data": {"type": "mock"},
                "load": {"type": "constant", "stages": [{"rate": RATE, "duration": DURATION}], "num_workers": 1},
                "api": {"type": "completion", "streaming": True},
                "server": {"type": "vllm", "base_url": runner.base_url},
                "report": {"request_lifecycle": {"summary": True, "per_request": True}},
            },
            timeout_sec=900,
        )
    )
    try:
        async with runner:
            result = await benchmark
    finally:
        if not benchmark.done():
            benchmark.cancel()
    assert_successful_run(result, RATE * DURATION)
    assert "Server ready: " in result.stdout
