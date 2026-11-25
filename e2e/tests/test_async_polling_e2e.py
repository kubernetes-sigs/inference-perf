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
"""End-to-end verification of HTTP 202 Accepted enqueue + GET /v1/requests/{id} polling.

Uses an ``AbsorberServer`` subclass that enqueues ``POST /v1/completions`` with 202,
answers 202 on the first poll, and returns either 200 (completion) or 504
(DEADLINE_EXCEEDED) on the second poll. Exercises real aiohttp transport, config
loading, header forwarding/exclusion, and summary/goodput report output.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Set

from aiohttp import web
import pytest

from utils.absorber import AbsorberServer
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.testdata import extract_tarball

MODEL_NAME = "google/gemma-3-270m"
MODEL_PATH = str(extract_tarball("e2e/testdata/models/google_gemma-3-270m.tar.gz"))


@dataclass
class _PendingJob:
    index: int
    payload: Dict[str, object]
    polls: int = 0


@dataclass
class AsyncEnqueueGateway(AbsorberServer):
    """Fake async-broker gateway answering 202 on POST and serving results on GET /v1/requests/{id}."""

    expire_indices: Set[int] = field(default_factory=set)
    poll_headers: List[Dict[str, str]] = field(default_factory=list)
    _jobs: Dict[str, _PendingJob] = field(default_factory=dict)

    # Start the aiohttp server with standard AbsorberServer routes plus GET /v1/requests/{id}.
    async def __aenter__(self) -> "AsyncEnqueueGateway":
        app = web.Application()
        app.router.add_post("/v1/completions", self._handle_completions)
        app.router.add_post("/v1/chat/completions", self._handle_chat)
        app.router.add_get("/v1/requests/{id}", self._handle_poll)
        app.router.add_get("/v1/models", self._handle_models)
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/metrics", self._handle_metrics)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        return self

    # Accept POST /v1/completions with HTTP 202 and store a pending job for subsequent GET polling.
    async def _respond(self, request: web.Request, route: str) -> web.StreamResponse:
        arrival_s = time.perf_counter()
        body = await request.json()
        absorbed = self._absorb(route, body, arrival_s)
        idx = len(self._jobs) + 1
        job_id = f"job-{idx}"
        n_out = absorbed.max_tokens or self.default_output_tokens
        chunks = self._chunk_texts(n_out)
        prompt_tokens = self._naive_prompt_tokens(absorbed)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": n_out,
            "total_tokens": prompt_tokens + n_out,
        }
        self._jobs[job_id] = _PendingJob(index=idx, payload=self._unary_body(route, chunks, usage))
        absorbed.done_s = time.perf_counter()
        return web.json_response({"id": job_id, "status": "pending"}, status=202)

    # Serve 202 on the first poll and 200 or 504 DEADLINE_EXCEEDED on the second poll.
    async def _handle_poll(self, request: web.Request) -> web.Response:
        self.poll_headers.append(dict(request.headers))
        job_id = request.match_info["id"]
        job = self._jobs.get(job_id)
        if job is None:
            return web.json_response({"error": {"message": "not found"}}, status=404)
        job.polls += 1
        if job.polls == 1:
            return web.json_response({"id": job_id, "status": "pending"}, status=202)
        if job.index in self.expire_indices:
            return web.json_response(
                {"error": {"code": "DEADLINE_EXCEEDED", "message": "deadline exceeded"}},
                status=504,
            )
        return web.json_response(job.payload, status=200)


# Run 6 mock requests through AsyncEnqueueGateway where 4 complete (202 -> 202 -> 200) and 2 expire (202 -> 202 -> 504).
# Verifies real aiohttp polling, header forwarding/exclusion, zero duplicate POST retries, and summary + goodput reports.
@pytest.mark.asyncio
async def test_async_polling_e2e_202_to_200_and_504() -> None:
    async with AsyncEnqueueGateway(port=get_free_port(), model=MODEL_NAME, expire_indices={2, 5}) as gateway:
        result = await run_benchmark_minimal(
            {
                "api": {
                    "type": "completion",
                    "streaming": False,
                    "slo_unit": "s",
                    "slo_request_latency_header": "x-llm-d-request-timeout-seconds",
                    "headers": {
                        "x-llm-d-async-mode": "enqueue",
                        "x-llm-d-tenant": "batch",
                        "x-llm-d-request-timeout-seconds": "10",
                    },
                    "async_polling": {
                        "poll_url_template": "/v1/requests/{id}",
                        "poll_interval_sec": 0.05,
                        "max_poll_duration_sec": 5.0,
                        "exclude_request_headers": ["x-llm-d-async-mode"],
                    },
                },
                "data": {"type": "mock"},
                "load": {
                    "type": "constant",
                    "stages": [{"rate": 3, "duration": 2}],
                    "num_workers": 1,
                    "request_retries": 2,
                },
                "server": {
                    "type": "vllm",
                    "base_url": gateway.base_url,
                    "model_name": MODEL_NAME,
                    "wait_for": "none",
                },
                "tokenizer": {"pretrained_model_name_or_path": MODEL_PATH},
                "report": {
                    "request_lifecycle": {
                        "summary": True,
                        "per_stage": False,
                        "per_request": True,
                    },
                },
            },
            timeout_sec=60,
        )

    assert result.success, f"run failed (rc={result.return_code}):\n{result.stdout[-4000:]}"
    assert result.reports, "no report was written"
    # Accepted 202 requests must never be re-POSTed even when polling ends in 504.
    assert len(gateway.requests) == 6
    assert len(gateway.poll_headers) == 12
    for hdrs in gateway.poll_headers:
        lower = {k.lower(): v for k, v in hdrs.items()}
        assert lower.get("x-llm-d-tenant") == "batch"
        assert "x-llm-d-async-mode" not in lower
        assert "content-type" not in lower

    summary = result.reports["summary_lifecycle_metrics.json"]
    assert summary["load_summary"]["count"] == 6
    assert summary["successes"]["count"] == 4
    assert summary["failures"]["count"] == 2
    assert summary["failures"]["by_label"]["504 - Gateway Timeout"]["count"] == 2

    goodput = summary["successes"]["goodput_metrics"]
    assert goodput is not None
    assert goodput["submitted_requests"] == 6
    assert goodput["good_requests"] == 4
    assert goodput["submitted_attainment_percentage"] == pytest.approx(400.0 / 6.0)
