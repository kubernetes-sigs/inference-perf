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
"""Bounded termination against a dead endpoint (#620, #606 Integration tier).

Two ways an endpoint can be dead, both driven through the real client, the real
worker processes and the real report generator:

1. nothing listening on the port, so every connect is refused at the transport
   layer;
2. a listener that accepts the connection and never writes a response byte, so
   only the client's own timeout can end the request.

What is asserted in both cases: the run terminates in bounded time, no request
is retried, every request lands in the reports as a failure, all three
request-lifecycle reports are still produced, and no worker process is left
alive holding a half-open connection.

Why it is worth asserting: nothing in the suite today checks that the client
times out at all. ``LoadConfig.request_timeout`` defaults to ``None``, which
means ``openAIModelServerClientSession`` passes ``aiohttp.helpers.sentinel``
instead of a ``ClientTimeout``, so an unresponsive server is bounded only by
aiohttp's own default. ``LoadGenerator.mp_run`` calls ``run_stage`` without a
timeout argument, so its "wait until all requests are finished" loop has no
deadline of its own: if a request never completes, the stage never ends. The
client timeout is the only thing standing between an unresponsive server and a
run that never returns, and that is what these tests pin down.

Both tests configure a small explicit ``request_timeout`` so they run fast. The
bound they assert (``RUN_BOUND_SEC``) sits far above the configured timeout and
far below aiohttp's default total timeout, so a regression that drops the
configured timeout on the floor fails the bound rather than passing slowly.

Note on scope: the hang in #469 is a barrier futex in the process machinery and
is unrelated to the client-side timeout path exercised here.

"Fake the conditions, never the oracle": the fakes supply only the failure
condition (a closed port, a silent listener). Every asserted value comes from
the production path, either the metrics the real client recorded or the reports
the real ``ReportGenerator`` produced.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import aiohttp.client
import pytest

from unresponsive_server import UnresponsiveServer, reserve_unbound_port

from inference_perf.apis import RequestLifecycleMetric
from inference_perf.client.modelserver import openai_client as openai_client_module
from inference_perf.client.modelserver.vllm_client import vLLMModelServerClient
from inference_perf.client.server_metrics.base import PerfRuntimeParameters
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.config import (
    APIConfig,
    APIType,
    Config,
    DataConfig,
    DataGenType,
    LoadConfig,
    LoadType,
    ReportConfig,
    RequestLifecycleMetricsReportConfig,
    StandardLoadStage,
)
from inference_perf.datagen import MockDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.metrics.request_collector import MultiprocessRequestMetricCollector
from inference_perf.reportgen.base import ReportGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer

# One stage of REQUESTS requests, dispatched at REQUESTS per second.
REQUESTS = 4
NUM_WORKERS = 2

# Small enough that an unresponsive server is abandoned quickly, large enough
# that a loaded CI machine does not trip it on a connect that would have
# succeeded.
REQUEST_TIMEOUT_SEC = 2.0

# The bounded-termination oracle. run_stage adds a one second dispatch offset
# and polls at one second granularity, so a correct run costs roughly
# REQUEST_TIMEOUT_SEC plus a few seconds of scheduling. Anything near
# aiohttp's five minute default total timeout, or an unbounded wait, blows
# through this.
RUN_BOUND_SEC = 60.0

# How long a worker gets to notice the stop signal and exit on its own once the
# run is over. A worker still wedged on a socket does not come back at all, so
# the exact value only has to be comfortably longer than a clean shutdown.
WORKER_EXIT_GRACE_SEC = 15.0

# Guard against the assertions passing on a run that never sent anything.
MIN_EXPECTED_ERROR_TYPES = 1

EXPECTED_REPORTS = (
    "summary_lifecycle_metrics",
    "stage_0_lifecycle_metrics",
    "per_request_lifecycle_metrics",
)


# Stands in for the HuggingFace tokenizer so building the client needs no Hub download.
# count_tokens("a b c") -> 3; nothing in these tests actually calls it.
class _StubTokenizer(CustomTokenizer):
    """Stands in for the HuggingFace-backed tokenizer the client builds in
    __init__, which would otherwise fetch from the Hub.

    Nothing here is ever called: on a pure transport failure the client never
    reaches ``process_failure`` bodies that tokenize, and
    ``CompletionAPIData.process_failure`` is a no-op. It is a module-level
    class rather than a Mock because the client is pickled into the worker
    processes under the forkserver start method.
    """

    def __init__(self) -> None:
        pass

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return len(text.split()) if text else 0


# A real vLLMModelServerClient at uri: non-streaming completion API, 2 TCP connections,
# request_timeout=2.0s, with the stub tokenizer patched in for the constructor only.
def _build_client(uri: str, collector: MultiprocessRequestMetricCollector) -> vLLMModelServerClient:
    api_config = APIConfig(type=APIType.Completion, streaming=False)
    with patch.object(openai_client_module, "CustomTokenizer", return_value=_StubTokenizer()):
        return vLLMModelServerClient(
            metrics_collector=collector,
            api_config=api_config,
            uri=uri,
            model_name="dead-server-model",
            tokenizer_config=None,
            max_tcp_connections=NUM_WORKERS,
            additional_filters=[],
            timeout=REQUEST_TIMEOUT_SEC,
        )


# The Config the ReportGenerator reads: one constant stage of 4 requests over 1s, 2 workers,
# request_timeout=2.0s, all three request-lifecycle reports switched on.
def _build_config() -> Config:
    """The config object the ReportGenerator reads. Its report section asks for
    all three request-lifecycle reports, which is what the run must still
    produce with zero successful requests."""
    return Config(
        api=APIConfig(type=APIType.Completion, streaming=False),
        data=DataConfig(type=DataGenType.Mock),
        load=LoadConfig(
            type=LoadType.CONSTANT,
            interval=0,
            num_workers=NUM_WORKERS,
            worker_max_concurrency=REQUESTS,
            request_timeout=REQUEST_TIMEOUT_SEC,
            stages=[StandardLoadStage(rate=REQUESTS, duration=1)],
        ),
        report=ReportConfig(
            request_lifecycle=RequestLifecycleMetricsReportConfig(
                summary=True,
                per_stage=True,
                per_request=True,
            )
        ),
    )


# Joins each worker against a shared 15s budget and returns their exit codes: 0 = clean,
# None = still running, negative = killed by a signal. Runs before LoadGenerator.stop() so a
# wedged worker is reported rather than terminated away.
def _await_worker_exits(load_gen: LoadGenerator) -> List[Optional[int]]:
    """Give every worker ``WORKER_EXIT_GRACE_SEC`` in total to exit by itself and
    report the exit codes.

    Deliberately done before ``LoadGenerator.stop()``, which escalates to
    ``terminate()``: a worker that only dies on SIGTERM is exactly the wedged
    worker this is looking for, and stop() would hide it.
    """
    deadline = time.monotonic() + WORKER_EXIT_GRACE_SEC
    for worker in load_gen.workers:
        worker.join(timeout=max(0.0, deadline - time.monotonic()))
    return [worker.exitcode for worker in load_gen.workers]


# Runs one full stage (4 requests, 2 workers, 2s timeout) at uri under a 60s wait_for, then
# generates the reports. Returns (elapsed seconds, the 4 recorded metrics,
# {report name: contents}, worker exit codes).
async def _run_against(uri: str) -> Tuple[float, List[RequestLifecycleMetric], Dict[str, Any], List[Optional[int]]]:
    """Drive one full stage at ``uri`` and return

    (elapsed seconds, recorded metrics, {report name: contents}, worker exit codes).

    The whole run is wrapped in ``asyncio.wait_for`` so a client that never
    gives up surfaces as a test failure instead of a hung suite.
    """
    config = _build_config()
    collector = MultiprocessRequestMetricCollector()
    client = _build_client(uri, collector)
    datagen = MockDataGenerator(config.api, config.data, None)
    load_gen = LoadGenerator(datagen, config.load)

    start = time.perf_counter()
    try:
        async with collector.start():
            await asyncio.wait_for(load_gen.mp_run(client), timeout=RUN_BOUND_SEC)
        elapsed = time.perf_counter() - start
        worker_exitcodes = _await_worker_exits(load_gen)
    finally:
        await load_gen.stop()

    runtime_parameters = PerfRuntimeParameters(
        start_time=start,
        duration=elapsed,
        model_server_metrics=BaseMetrics(),
        stages=load_gen.stage_runtime_info,
    )
    reportgen = ReportGenerator(metrics_client=None, metrics_collector=collector, config=config)
    report_files = await reportgen.generate_reports(config.report, runtime_parameters)
    reports = {report.name: report.contents for report in report_files}

    return elapsed, collector.get_metrics(), reports, worker_exitcodes


# Exactly 4 metrics, each with an error and no response_metrics. Returns the sorted distinct
# error_type names, e.g. ["ClientConnectorError"].
def _assert_every_request_failed(metrics: List[RequestLifecycleMetric]) -> List[str]:
    """Every request must carry an error, and none may look like a success.
    Returns the distinct error types seen."""
    assert len(metrics) == REQUESTS, f"expected {REQUESTS} recorded requests, got {len(metrics)}"

    error_types = set()
    for metric in metrics:
        assert metric.error is not None, "a request against a dead endpoint was recorded without an error"
        error_types.add(metric.error.error_type)
        # A failed request must not carry response metrics: those feed the
        # latency and token aggregates, which must be computed from successful
        # requests only.
        assert metric.info is None or metric.info.response_metrics is None, (
            f"failed request carries response_metrics: {metric.info}"
        )

    assert len(error_types) >= MIN_EXPECTED_ERROR_TYPES, "no error types recorded, the run sent nothing"
    return sorted(error_types)


# summary, stage_0 and per_request reports all present; per_request has 4 entries, all with
# an error; summary and stage_0 both say successes=0 and failures=4.
def _assert_reports_are_complete(reports: Dict[str, Any]) -> None:
    """All three request-lifecycle reports exist, and both summaries agree that
    nothing succeeded and everything failed."""
    for name in EXPECTED_REPORTS:
        assert name in reports, f"missing {name}, have {sorted(reports)}"

    per_request = reports["per_request_lifecycle_metrics"]
    assert len(per_request) == REQUESTS, f"per-request report has {len(per_request)} entries, expected {REQUESTS}"
    assert all(entry["error"] for entry in per_request), "per-request report shows a request without an error"

    for name in ("summary_lifecycle_metrics", "stage_0_lifecycle_metrics"):
        summary = reports[name]
        assert summary["successes"]["count"] == 0, f"{name} reports a success against a dead endpoint"
        assert summary["failures"]["count"] == REQUESTS, (
            f"{name} reports {summary['failures']['count']} failures, expected {REQUESTS}"
        )


# Exactly 2 exit codes and every one is 0: no worker was still running or had to be signalled.
def _assert_no_worker_wedged(worker_exitcodes: List[Optional[int]]) -> None:
    """Every worker exited on its own, cleanly.

    ``None`` means the process was still running when the grace period ran out,
    a negative code means it had to be signalled: both are a worker that the
    dead endpoint left holding a connection it could not let go of.
    """
    assert len(worker_exitcodes) == NUM_WORKERS, f"expected {NUM_WORKERS} workers, saw {len(worker_exitcodes)}"
    assert all(code == 0 for code in worker_exitcodes), (
        f"workers did not exit cleanly on their own, exit codes: {worker_exitcodes}"
    )


# 4 requests at a port with no listener. The run ends in under 60s (a few seconds in
# practice), all 4 fail with error_type ClientConnectorError only (never TimeoutError), the
# three reports say 0 successes / 4 failures, and both workers exit 0.
@pytest.mark.asyncio
async def test_run_terminates_when_nothing_is_listening() -> None:
    """Case (a): the endpoint's port has no listener, so every connect is
    refused. The run must end quickly with N failures and complete reports."""
    port = reserve_unbound_port()
    elapsed, metrics, reports, worker_exitcodes = await _run_against(f"http://127.0.0.1:{port}")

    assert elapsed < RUN_BOUND_SEC, f"run took {elapsed:.1f}s against a closed port"
    error_types = _assert_every_request_failed(metrics)
    # A refused connect is an aiohttp connector error, never a timeout: if the
    # client is waiting out the timeout on a connection that was refused
    # immediately, that is a bug worth catching here.
    assert error_types == ["ClientConnectorError"], f"unexpected error types on a refused connect: {error_types}"
    _assert_reports_are_complete(reports)
    _assert_no_worker_wedged(worker_exitcodes)


# No load, no server. A client built with request_timeout=2.0 must show
# session.timeout.total == 2.0; one built with no request_timeout must show
# aiohttp.client.DEFAULT_TIMEOUT (300s total).
@pytest.mark.asyncio
async def test_configured_request_timeout_reaches_the_aiohttp_session() -> None:
    """The wiring the unresponsive-server case rests on: ``request_timeout``
    from the load config has to arrive at the aiohttp session as a total
    timeout, and with nothing configured the client applies none of its own.

    The second half is not a bug report, it is the documented default: with
    ``request_timeout`` unset the only bound on a request to a server that
    never answers is aiohttp's own default, which is two orders of magnitude
    longer than any timeout a benchmark would choose. Pinning it here means a
    change to that default is visible rather than silent.
    """
    collector = MultiprocessRequestMetricCollector()

    configured = _build_client("http://127.0.0.1:1", collector)
    session = configured.new_session()
    try:
        assert isinstance(session, openai_client_module.openAIModelServerClientSession)
        assert session.session.timeout.total == REQUEST_TIMEOUT_SEC, (
            f"configured request_timeout did not reach the session: {session.session.timeout}"
        )
    finally:
        await session.close()

    with patch.object(openai_client_module, "CustomTokenizer", return_value=_StubTokenizer()):
        unconfigured = vLLMModelServerClient(
            metrics_collector=collector,
            api_config=APIConfig(type=APIType.Completion, streaming=False),
            uri="http://127.0.0.1:1",
            model_name="dead-server-model",
            tokenizer_config=None,
            max_tcp_connections=1,
            additional_filters=[],
        )
    default_session = unconfigured.new_session()
    try:
        assert isinstance(default_session, openai_client_module.openAIModelServerClientSession)
        assert default_session.session.timeout == aiohttp.client.DEFAULT_TIMEOUT, (
            f"with no request_timeout the client must fall through to aiohttp's default, got {default_session.session.timeout}"
        )
    finally:
        await default_session.close()


# 4 requests at a listener that accepts and never answers. The run ends in under 60s, all 4
# fail with error_type TimeoutError only, the fake accepted between 1 and 4 connections (no
# retries), the three reports say 0 successes / 4 failures, and both workers exit 0.
@pytest.mark.asyncio
async def test_run_terminates_when_the_server_never_responds() -> None:
    """Case (b): the endpoint accepts the connection and never answers. Only
    the configured client timeout can end these requests, so this is the test
    that fails if the timeout is not applied."""
    async with UnresponsiveServer() as server:
        elapsed, metrics, reports, worker_exitcodes = await _run_against(server.base_url)
        connections = server.connections

    assert elapsed < RUN_BOUND_SEC, f"run took {elapsed:.1f}s against an unresponsive server"
    error_types = _assert_every_request_failed(metrics)
    assert error_types == ["TimeoutError"], f"unresponsive server produced {error_types}, expected the client timeout"

    # No retry loop: each request may open at most one connection, and aiohttp
    # is free to reuse one, so more accepted connections than requests means
    # something dialed again after giving up.
    assert 0 < connections <= REQUESTS, f"fake server accepted {connections} connections for {REQUESTS} requests"

    _assert_reports_are_complete(reports)
    _assert_no_worker_wedged(worker_exitcodes)
