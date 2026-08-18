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
"""Worker-count matrix for issue #632 (#606 Integration tier, per-change lane).

``num_workers`` is not a scaling knob, it selects between two execution
architectures: 0 runs the whole benchmark on one in-process asyncio loop
through ``LocalRequestMetricCollector``, and > 0 runs the ``mp_run`` path with
real worker processes, per-worker queues, shared counters, a stage barrier and
``MultiprocessRequestMetricCollector``. Every motivating bug (#589, #590, #469,
#593) lives in that process machinery rather than in the server, so this file
fakes the server and keeps the processes real: the load runs against the
in-process ``FakeOpenAIServer`` over loopback, which child workers reach exactly
like any other HTTP endpoint.

The oracle is the fake server's own record of what it served. Whatever the
architecture, the number of requests the client accounts for must equal the
number the server actually handled, and the per-request token accounting must
be identical in every cell. That invariant is what a lost, duplicated or
silently dropped worker result breaks.

Start-method agnostic on purpose. Nothing here asserts on, or configures, the
multiprocessing start method: this file is meant to gate #526 (fork ->
forkserver), so it has to pass identically before and after that switch. What
it does assert is the property #526 depends on, that the worker payload
survives a pickle round trip (#589), which holds under either method.

Everything driving real processes is wrapped in ``asyncio.wait_for``: a
barrier-quorum regression (#469) hangs rather than fails, and a bounded wait
turns that hang into a test failure instead of a stuck CI job.

The ``num_workers = 0`` cell sends every planned request of a stage, like the
mp cells; the per-stage count in the matrix test pins that.
"""

import asyncio
import pickle
import time
from typing import Any, Dict, List, Optional

import pytest

from fake_openai_server import FakeOpenAIServer, StreamEvent

from inference_perf.apis import RequestLifecycleMetric
from inference_perf.client.modelserver.base import ModelServerClient
from inference_perf.client.modelserver.openai_client import OpenAIMetrics, openAIModelServerClient
from inference_perf.client.modelserver.metrics import CounterMetric, GaugeMetric, HistogramMetric
from inference_perf.client.modelserver.otel_instrumentation import get_otel_instrumentation
from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, LoadConfig, LoadType, StandardLoadStage
from inference_perf.datagen.synthetic.mock_datagen import MockDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.metrics.request_collector import (
    LocalRequestMetricCollector,
    MultiprocessRequestMetricCollector,
    RequestMetricCollector,
)
from inference_perf.reportgen.base import ResponsesSummary, summarize_requests
from inference_perf.utils.custom_tokenizer import CustomTokenizer

# N is pinned rather than inherited from cpu_count(), which is what makes a CI
# run test a known cell instead of whichever cell the runner happens to have.
PINNED_N = 4

RATE = 8
DURATION = 1
NUM_STAGES = 2
REQUESTS_PER_STAGE = RATE * DURATION

PERCENTILES = [50.0, 90.0]

# Generous but finite. A healthy cell finishes in well under 20s; this only has
# to be small enough that a deadlock fails the job rather than wedging it.
RUN_TIMEOUT_SEC = 180.0

# One scripted response, four single-word content chunks. Word-per-chunk keeps
# the whitespace tokenizer below exact: four chunks, four tokens, and the
# concatenation re-counts to the same four, so token_count_mismatches stays 0
# and any per-cell divergence is a real divergence.
RESPONSE_SCRIPT = [
    StreamEvent("content", "alpha", 0.01),
    StreamEvent("content", " beta", 0.01),
    StreamEvent("content", " gamma", 0.01),
    StreamEvent("content", " delta", 0.01),
]
EXPECTED_OUTPUT_TOKENS = 4

# How many requests the collector-equivalence comparison drives. Small: the
# comparison is exact, so more requests buy nothing but wall time.
EQUIVALENCE_REQUESTS = 8


class WhitespaceTokenizer(CustomTokenizer):
    """Counts whitespace-delimited words, with no model download.

    Deliberately does not call ``CustomTokenizer.__init__``: that pulls a
    HuggingFace tokenizer, which the per-change lane cannot depend on. Token
    fidelity is not what this file is testing (#631 and #697 own that); what
    matters here is that the count is deterministic and identical in every
    cell, so a difference between cells is attributable to the process
    machinery.
    """

    def __init__(self) -> None:
        pass

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return len(text.split())

    def get_tokenizer(self) -> Any:
        return None


class FakeServerClient(openAIModelServerClient):
    """The production OpenAI client, pointed at the in-process fake.

    Only the constructor is replaced, and only to skip building a
    ``CustomTokenizer`` from the hub. Request-body construction, the aiohttp
    session, SSE parsing, response accounting and the ``record_metric`` call
    are all the shipped code path, which is what makes the collector
    comparison below meaningful.
    """

    def __init__(self, metrics_collector: RequestMetricCollector, api_config: APIConfig, uri: str) -> None:
        ModelServerClient.__init__(self, api_config, None)
        self.uri = uri
        self.model_name = "fake-model"
        self.max_completion_tokens = 30
        # The fake stops where its script stops, which is EOS behaviour, not a
        # truncation; under ignore_eos a short reply would be recorded as one.
        self.ignore_eos = False
        self.metrics_collector = metrics_collector
        self.max_tcp_connections = 100
        self.additional_filters: List[str] = []
        self.api_key: Optional[str] = None
        self.cert_path: Optional[str] = None
        self.key_path: Optional[str] = None
        self.lora_config = None
        self.otel = get_otel_instrumentation()
        self.tokenizer = WhitespaceTokenizer()

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_prometheus_metric_metadata(self) -> OpenAIMetrics:
        return OpenAIMetrics(
            filters=[],
            prompt_tokens=CounterMetric("fake:prompt_tokens"),
            output_tokens=CounterMetric("fake:generation_tokens"),
            requests=CounterMetric("fake:request_success"),
            request_latency=HistogramMetric("fake:e2e_request_latency_seconds"),
            queue_length=GaugeMetric("fake:num_requests_waiting"),
            time_per_output_token=HistogramMetric("fake:time_per_output_token_seconds"),
        )


# make_load_config(4) gives two constant stages at rate 8 for 1s on 4 workers,
# up to 16 requests in flight per worker, no pause between stages, base_seed 42.
def make_load_config(num_workers: int) -> LoadConfig:
    return LoadConfig(
        type=LoadType.CONSTANT,
        num_workers=num_workers,
        worker_max_concurrency=16,
        interval=0,
        stages=[StandardLoadStage(rate=RATE, duration=DURATION) for _ in range(NUM_STAGES)],
        base_seed=42,
    )


# For a chat api_config it yields user messages "mock prompt 1", "mock prompt 2" and so
# on, forever, with no tokenizer.
def make_datagen(api_config: APIConfig) -> MockDataGenerator:
    return MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)


# run_cell(4) runs both stages on 4 worker processes into the multiprocess collector and
# returns (metrics, served), 16 and 16 when healthy. run_cell(0) runs in-process into
# LocalRequestMetricCollector and returns 14 to 16 of each.
async def run_cell(num_workers: int) -> tuple[List[RequestLifecycleMetric], int]:
    """Run the configured multi-stage load in one matrix cell.

    Returns the collected metrics and the number of requests the fake server
    actually served, which is the oracle the metrics are checked against.
    """
    api_config = APIConfig(type=APIType.Chat, streaming=True)
    collector: RequestMetricCollector = (
        LocalRequestMetricCollector() if num_workers == 0 else MultiprocessRequestMetricCollector()
    )

    async with FakeOpenAIServer(RESPONSE_SCRIPT, completion_tokens=EXPECTED_OUTPUT_TOKENS) as server:
        client = FakeServerClient(collector, api_config, f"http://127.0.0.1:{server.port}")
        load_gen = LoadGenerator(make_datagen(api_config), make_load_config(num_workers))
        try:
            async with collector.start():
                # A stage-barrier quorum bug (#469) deadlocks rather than
                # failing, so the whole run is bounded.
                await asyncio.wait_for(load_gen.run(client), timeout=RUN_TIMEOUT_SEC)
        finally:
            await load_gen.stop()
            await client.close()
        served = len(server.served)

    return collector.get_metrics(), served


# count=8 sends 8 concurrent requests through the real client with no load
# generator in between. Returns how many the fake server served, 8 when healthy.
async def drive_requests(collector: RequestMetricCollector, count: int) -> int:
    """Send ``count`` real requests through the real client, concurrently.

    Deliberately bypasses the load generator. The collector comparison below
    needs one fixed body of load, and the stage dispatch loops add scheduling
    variance that has nothing to do with the collectors. The request path, the
    wire format and the metric construction are still the real ones. Returns
    what the server served.
    """
    api_config = APIConfig(type=APIType.Chat, streaming=True)
    async with FakeOpenAIServer(RESPONSE_SCRIPT, completion_tokens=EXPECTED_OUTPUT_TOKENS) as server:
        client = FakeServerClient(collector, api_config, f"http://127.0.0.1:{server.port}")
        data_stream = make_datagen(api_config).get_data()
        requests = [next(data_stream) for _ in range(count)]
        try:
            scheduled = time.perf_counter()
            await asyncio.gather(*(client.process_request(request, 0, scheduled) for request in requests))
        finally:
            await client.close()
        return len(server.served)


# 8 metrics of 4 output tokens give successes count 8 and output_tokens total 32.
# Each latency field that is set carries mean, min, max, median and p90.
def summarize(metrics: List[RequestLifecycleMetric]) -> ResponsesSummary:
    return summarize_requests(metrics, PERCENTILES, tokenizer=WhitespaceTokenizer())


# Metrics with output_tokens 4, 2 and 3 give [2, 3, 4]. A metric with no
# response_metrics or no output_tokens fails the assertion.
def output_token_values(metrics: List[RequestLifecycleMetric]) -> List[int]:
    """Per-request output token counts, sorted, so two collectors can be
    compared without depending on the order they reassembled the metrics in."""
    values: List[int] = []
    for metric in metrics:
        response_metrics = metric.info.response_metrics
        assert response_metrics is not None
        assert response_metrics.output_tokens is not None
        values.append(response_metrics.output_tokens)
    return sorted(values)


# Two 1s stages at rate 8 with 0, 1 and 4 workers. Every served request comes back
# with 4 output tokens. The mp cells send 8 per stage, the in-process cell 7 or 8.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "num_workers",
    [
        pytest.param(0, id="workers-0"),
        pytest.param(1, id="workers-1"),
        pytest.param(PINNED_N, id=f"workers-{PINNED_N}"),
    ],
)
async def test_worker_matrix_accounts_every_served_request(num_workers: int) -> None:
    """Each cell of the matrix must account for exactly what the server served.

    This is the invariant that survives the architectural difference between
    the cells. It fails if a worker result is lost crossing the queue, counted
    twice, or produced by a worker that died silently (#593). The in-process
    cell is held to it too, since that path once crashed with nothing in CI
    running it (#590).
    """
    metrics, served = await run_cell(num_workers)

    assert served > 0, "the fake server handled no requests at all"
    assert len(metrics) == served, f"collected {len(metrics)} metrics for {served} served requests"

    errored = [m for m in metrics if m.error is not None]
    assert not errored, f"{len(errored)}/{len(metrics)} requests errored, first: {errored[0].error}"

    # Both stages must have run and both must have delivered load.
    by_stage: Dict[int, int] = {}
    for metric in metrics:
        assert metric.stage_id is not None, "a recorded metric lost its stage id"
        by_stage[metric.stage_id] = by_stage.get(metric.stage_id, 0) + 1
    assert sorted(by_stage) == list(range(NUM_STAGES)), f"expected one bucket per stage, got {sorted(by_stage)}"
    for stage_id in range(NUM_STAGES):
        assert by_stage[stage_id] == REQUESTS_PER_STAGE, (
            f"stage {stage_id} delivered {by_stage[stage_id]} requests, expected {REQUESTS_PER_STAGE}"
        )

    # Per-request token accounting is identical in every cell: the response is
    # scripted, so any drift here is the process machinery mangling a metric
    # on its way back, not the server.
    for metric in metrics:
        assert metric.info.response_metrics is not None
        assert metric.info.response_metrics.output_tokens == EXPECTED_OUTPUT_TOKENS

    summary = summarize(metrics)
    assert summary.successes["count"] == len(metrics)
    assert summary.failures["count"] == 0
    assert summary.successes["token_count_mismatches"] == 0
    assert summary.successes["output_tokens"]["total"] == float(len(metrics) * EXPECTED_OUTPUT_TOKENS)
    # Latency must actually have been measured on every path, including
    # through the queue: a null TTFT here means the streamed timestamps did
    # not survive the trip.
    assert summary.successes["latency"]["time_to_first_token"] is not None
    assert summary.successes["latency"]["inter_token_latency"] is not None


# 8 real requests recorded locally, then the same 8 metrics replayed from a child process
# into the multiprocess collector. Counts and token values match exactly, latency
# summaries to within 1e-9.
@pytest.mark.asyncio
async def test_collectors_agree_on_identical_load() -> None:
    """``LocalRequestMetricCollector`` and ``MultiprocessRequestMetricCollector``
    must produce equivalent aggregates for identical load.

    "Identical" is made literal: one real batch of requests against the fake
    produces the metrics, then the same metric objects go through both
    collectors, with the multiprocess one fed from a real child process so its
    queue, its pickling and its drain-on-sentinel protocol are all exercised.
    Any difference in the aggregates is therefore attributable to the
    collector, not to two runs having drawn different request timings. The
    matrix test above is what covers real workers feeding the multiprocess
    collector end to end; this one isolates the transport.

    What is compared, and why the split:

    - EXACT: request counts, failure counts, token totals and the per-request
      token values. These are integers copied verbatim through the queue, so
      any difference at all is a transport bug.
    - TOLERANCE: the latency summaries. The values themselves are copied
      verbatim too, but the multiprocess collector reassembles them in queue
      arrival order rather than record order, and ``np.mean`` over a permuted
      list of floats can differ in the last bits. Bit-exactness would be an
      assertion about float summation order, not about the collector, so the
      comparison is to within a nanosecond, far below any real distortion
      (#566 was a K-fold ITL deflation).
    - STRUCTURAL: which latency fields are present versus None. A collector
      that dropped the streamed timestamps would keep the counts intact and
      silently turn TTFT into None, so presence is asserted separately from
      value.
    """
    local = LocalRequestMetricCollector()
    served = await drive_requests(local, EQUIVALENCE_REQUESTS)
    assert served == EQUIVALENCE_REQUESTS
    metrics = local.get_metrics()
    assert len(metrics) == EQUIVALENCE_REQUESTS

    multiprocess = MultiprocessRequestMetricCollector()
    async with multiprocess.start():
        # Feeding from a child process is the point: recording in-process
        # would compare a list against itself and never exercise the
        # cross-process transport that #632 is about.
        await asyncio.wait_for(_feed_from_child_process(multiprocess, metrics), timeout=RUN_TIMEOUT_SEC)

    local_metrics = metrics
    mp_metrics = multiprocess.get_metrics()
    assert len(mp_metrics) == len(local_metrics), "multiprocess collector lost or duplicated a metric in transit"

    local_summary = summarize(local_metrics)
    mp_summary = summarize(mp_metrics)

    # --- exact ---
    assert mp_summary.successes["count"] == local_summary.successes["count"]
    assert mp_summary.failures["count"] == local_summary.failures["count"]
    assert mp_summary.load_summary["count"] == local_summary.load_summary["count"]
    assert mp_summary.successes["token_count_mismatches"] == local_summary.successes["token_count_mismatches"]
    assert mp_summary.successes["output_tokens"]["total"] == local_summary.successes["output_tokens"]["total"]
    assert mp_summary.successes["prompt_tokens"]["total"] == local_summary.successes["prompt_tokens"]["total"]
    assert output_token_values(mp_metrics) == output_token_values(local_metrics)

    # --- structural, then tolerance ---
    for field in ("request_latency", "time_to_first_token", "time_per_output_token", "inter_token_latency"):
        local_field = local_summary.successes["latency"][field]
        mp_field = mp_summary.successes["latency"][field]
        assert (local_field is None) == (mp_field is None), f"{field} is defined on one collector but not the other"
        if local_field is None:
            continue
        assert sorted(local_field) == sorted(mp_field), f"{field} summaries have different keys"
        for key, value in local_field.items():
            assert mp_field[key] == pytest.approx(value, abs=1e-9), f"{field}.{key} differs between collectors"


# Given 8 metrics in a child process, puts all 8 on the collector's queue in order.
def _record_all(collector: MultiprocessRequestMetricCollector, metrics: List[RequestLifecycleMetric]) -> None:
    """Child-process entrypoint: replay metrics into the collector's queue.

    Module level and importable by name so it survives a start method that
    pickles the target instead of inheriting it.
    """
    for metric in metrics:
        collector.record_metric(metric)


# Runs _record_all in a daemon child process and waits for it off the event loop.
# Fails if the child exits with anything but 0.
async def _feed_from_child_process(
    collector: MultiprocessRequestMetricCollector, metrics: List[RequestLifecycleMetric]
) -> None:
    import multiprocessing as mp

    process = mp.Process(target=_record_all, args=(collector, metrics), daemon=True)
    process.start()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, process.join)
    assert process.exitcode == 0, f"metric-feeding child exited with {process.exitcode}"


# Pickles and unpickles the mock datagen and the client stub. The datagen still yields a
# request, the client keeps uri http://127.0.0.1:1 and counts "one two three" as 3 tokens.
def test_worker_payload_survives_a_pickle_round_trip() -> None:
    """The datagen and client a worker receives must be picklable (#589).

    Under today's default start method the payload is inherited rather than
    pickled, so nothing in CI would notice it becoming unpicklable until #526
    switches workers to forkserver and every mp run breaks at once. Asserting
    the property directly keeps the guard independent of which start method is
    in force, which is the whole point of this file gating #526.

    Narrow by design: it pins the payload this file builds, not the real
    tokenizer-backed datagens, whose forkserver behavior #526 covers in its own
    e2e.
    """
    api_config = APIConfig(type=APIType.Chat, streaming=True)
    datagen = make_datagen(api_config)
    client = FakeServerClient(LocalRequestMetricCollector(), api_config, "http://127.0.0.1:1")

    restored_datagen = pickle.loads(pickle.dumps(datagen))
    assert next(restored_datagen.get_data()) is not None

    restored_client = pickle.loads(pickle.dumps(client))
    assert restored_client.uri == client.uri
    assert restored_client.tokenizer.count_tokens("one two three") == 3
