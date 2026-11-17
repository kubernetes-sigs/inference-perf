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
"""``server.wait_for`` against a gateway whose pool has scaled to zero (#1065).

The gateway is the absorber with the behavior of such a gateway added: its
readiness routes (``/health`` and ``/v1/models``) answer 503 until the first
completion request arrives, and that request starts a backend that takes
``COLD_START_S`` to come up. Completion requests are held until then, the way a
gateway queues requests for an empty pool. Nothing but a completion request
starts the backend.

- ``reachable``: the one probe gets a 503, which is an HTTP response, so the run
  starts; every request is held through the cold start and completes.
- ``none``: no probe at all; the same.
- ``ready``: the readiness wait only ever sees 503, and nothing else is sent
  until it ends, so the backend never starts and the run fails at the deadline.
  The CLI's deadline is 600 s; to keep the test short the client is built
  in-process with ``readiness_timeout_sec`` set to ``READY_DEADLINE_S``, which
  runs the same wait.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web

from inference_perf.client.modelserver.vllm_client import vLLMModelServerClient
from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig, WaitFor
from utils.absorber import AbsorberServer
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.testdata import extract_tarball

MODEL_NAME = "google/gemma-3-270m"
MODEL_PATH = str(extract_tarball("e2e/testdata/models/google_gemma-3-270m.tar.gz"))

# How long the backend takes to come up after the first completion request.
COLD_START_S = 3.0
# 20 requests at 5 per second: the first few arrive during the cold start.
RATE = 5
DURATION = 4
NUM_REQUESTS = RATE * DURATION
# The readiness deadline for the in-process `ready` case.
READY_DEADLINE_S = 5.0


class ScaledToZeroGateway(AbsorberServer):
    """The absorber behind a gateway whose pool is empty until the first completion request.

    ``readiness_probes`` lists the HTTP status of every ``/health`` and ``/v1/models``
    request, ``scale_up_s`` is when the first completion request arrived and
    ``backend_up_s`` when the backend came up (``time.perf_counter()`` readings, as
    the absorber's own timestamps are).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.readiness_probes: List[int] = []
        self.scale_up_s: Optional[float] = None
        self.backend_up_s: Optional[float] = None
        self._backend_up = asyncio.Event()

    async def _start_backend(self) -> None:
        await asyncio.sleep(COLD_START_S)
        self.backend_up_s = time.perf_counter()
        self._backend_up.set()

    async def _readiness(self, request: web.Request, answer_when_up: Any) -> web.Response:
        status = 200 if self._backend_up.is_set() else 503
        self.readiness_probes.append(status)
        if status == 503:
            return web.Response(status=503, text="no backend")
        return await answer_when_up(request)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return await self._readiness(request, super()._handle_health)

    async def _handle_models(self, request: web.Request) -> web.Response:
        return await self._readiness(request, super()._handle_models)

    async def _respond(self, request: web.Request, route: str) -> web.StreamResponse:
        if self.scale_up_s is None:
            self.scale_up_s = time.perf_counter()
            asyncio.get_running_loop().create_task(self._start_backend())
        await self._backend_up.wait()
        return await super()._respond(request, route)


# 20 mock prompts at 5 per second against the gateway, with model and tokenizer
# pinned (both non-default modes require model_name), unary completions, and a
# per-request report so every request's latency can be read back.
def _config(base_url: str, wait_for: str) -> Dict[str, Any]:
    return {
        "api": {"type": "completion", "streaming": False},
        "data": {"type": "mock"},
        "load": {"type": "constant", "stages": [{"rate": RATE, "duration": DURATION}], "num_workers": 1},
        "server": {"type": "vllm", "base_url": base_url, "model_name": MODEL_NAME, "wait_for": wait_for},
        "tokenizer": {"pretrained_model_name_or_path": MODEL_PATH},
        "report": {"request_lifecycle": {"summary": True, "per_stage": False, "per_request": True}},
    }


@pytest.mark.parametrize(("wait_for", "probes"), [("reachable", [503]), ("none", [])])
async def test_run_against_a_pool_scaled_to_zero_completes(wait_for: str, probes: List[int]) -> None:
    async with ScaledToZeroGateway(port=get_free_port(), model=MODEL_NAME) as gateway:
        result = await run_benchmark_minimal(_config(gateway.base_url, wait_for), timeout_sec=None)

    assert result.success, f"run failed (rc={result.return_code}):\n{result.stdout[-4000:]}"
    assert result.reports, "no report was written"
    # reachable probes once and takes the 503; none sends nothing before the run.
    assert gateway.readiness_probes == probes
    # The run itself started the backend, and every request waited for it.
    assert gateway.scale_up_s is not None and gateway.backend_up_s is not None
    assert len(gateway.requests) == NUM_REQUESTS
    assert all(r.done_s is not None and r.done_s >= gateway.backend_up_s for r in gateway.requests)

    summary = result.reports["summary_lifecycle_metrics.json"]
    assert summary["successes"]["count"] == NUM_REQUESTS
    assert summary["failures"]["count"] == 0
    # The cold start lands inside the measured run: the first request waited it out.
    per_request = result.reports["per_request_lifecycle_metrics.json"]
    slowest = max(float(entry["end_time"]) - float(entry["start_time"]) for entry in per_request)
    assert slowest >= COLD_START_S * 0.9, f"no request waited out the {COLD_START_S}s cold start (slowest {slowest:.2f}s)"


async def test_ready_never_starts_a_run_against_a_pool_scaled_to_zero() -> None:
    def build_client() -> vLLMModelServerClient:
        return vLLMModelServerClient(
            metrics_collector=MagicMock(),
            api_config=APIConfig(type=APIType.Completion, streaming=False),
            uri=gateway.base_url,
            model_name=MODEL_NAME,
            tokenizer_config=CustomTokenizerConfig(pretrained_model_name_or_path=MODEL_PATH),
            max_tcp_connections=4,
            additional_filters=[],
            wait_for=WaitFor.READY,
        )

    async with ScaledToZeroGateway(port=get_free_port(), model=MODEL_NAME) as gateway:
        with (
            patch.object(vLLMModelServerClient, "readiness_timeout_sec", READY_DEADLINE_S),
            pytest.raises(Exception, match=rf"did not answer 200 within {READY_DEADLINE_S}s \(last: HTTP 503\)"),
        ):
            await asyncio.to_thread(build_client)

    # The wait polled and only ever saw 503; no completion request was sent, so the
    # backend never started.
    assert gateway.readiness_probes and set(gateway.readiness_probes) == {503}
    assert gateway.requests == []
    assert gateway.scale_up_s is None
