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
"""Integration test for issue #726 (#606 Integration tier, per-change lane).

This release has token-count oracles (#631, #627, #580) and unit coverage of the latency
arithmetic (`tests/required/reportgen/test_summarize_requests.py`), but nothing that
injects a known latency profile and asserts the numbers that come back out equal it.
#564 was exactly that gap: a latency metric silently wrong for roughly two months.

These tests script a stream with a known first-token delay and known per-chunk gaps,
serve it over real HTTP, and drive the real path end to end: the real
`openAIModelServerClient` session, the real aiohttp streaming parser, the real
`LocalRequestMetricCollector`, and `ReportGenerator.generate_reports`. The reported
TTFT, ITL, TPOT and request latency are then checked against the injected profile.

What is faked and what is the oracle:

* Faked: the model server. `FakeOpenAIServer` (added by #694) serves a scripted SSE
  sequence with controlled pauses.
* Oracle: the server's own `perf_counter` send stamps. They are taken in the same
  process on the same clock as the client's receive stamps, so they bound the reported
  values directly and do not inherit `asyncio.sleep` imprecision. The configured script
  is the second, looser oracle: it says the pauses under test were really injected.

Deliberately out of scope: coordinated-omission semantics, that is, whether reported
latency should be corrected by `schedule_delay` under offered load. That is a
metric-definition decision, it is tracked on #726, and it does not gate v0.7.0.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

import pytest

from fake_openai_server import FakeOpenAIServer, ServedStream, StreamEvent

from inference_perf.apis import RequestLifecycleMetric
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.client.modelserver import openai_client as openai_client_module
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.client.modelserver.openai_client import OpenAIMetrics, openAIModelServerClient
from inference_perf.client.server_metrics.base import PerfRuntimeParameters, StageRuntimeInfo, StageStatus
from inference_perf.config import (
    APIConfig,
    APIType,
    Config,
    CustomTokenizerConfig,
    ReportConfig,
    RequestLifecycleMetricsReportConfig,
)
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector
from inference_perf.reportgen.base import ReportGenerator

# --- Tolerances -------------------------------------------------------------------
#
# The fake server and the client run in one process on one `perf_counter` clock, so the
# client cannot observe a chunk before the server's send stamp for it. CLOCK_EPS absorbs
# float comparison noise on that lower bound and nothing else.
CLOCK_EPS = 1e-6

# How much later than a send stamp the client may observe the chunk: one loopback write
# plus one SSE line parse. That is sub-millisecond when measured locally, so 50 ms is
# roughly two orders of magnitude of headroom for a contended CI runner. Every gap these
# tests distinguish is separated by at least 300 ms, so the bound stays far from the
# values it is asserting about.
DELIVERY_TOLERANCE = 0.05

# `asyncio.sleep` guarantees a lower bound, not a duration, so the server's own send
# stamps drift later than the configured script. This is the tolerance for the separate,
# looser check that the injected pauses actually happened.
SCHEDULING_TOLERANCE = 0.10

# Combined bound for a reported value checked straight against the configured script.
INJECTION_TOLERANCE = SCHEDULING_TOLERANCE + DELIVERY_TOLERANCE

# --- Injected profile -------------------------------------------------------------
#
# One chunk per token: the fake tokenizer counts whitespace-separated words, and every
# chunk below carries exactly one, so the arithmetic is checkable by hand.
FIRST_TOKEN_DELAY = 0.25
# Alternating slow and fast gaps, so ITL min/max/median describe the distribution rather
# than collapsing onto the mean. An implementation that reported only the average gap
# would pass on `mean` and fail on `min` and `max`.
CHUNK_GAPS = [0.40, 0.10, 0.40, 0.10]
CHUNK_TEXTS = ["alpha", " beta", " gamma", " delta", " epsilon"]
# "alpha beta gamma delta epsilon" is five whitespace-separated words.
EXPECTED_OUTPUT_TOKENS = 5

SCRIPT = [
    StreamEvent("content", text, delay) for text, delay in zip(CHUNK_TEXTS, [FIRST_TOKEN_DELAY] + CHUNK_GAPS, strict=True)
]


class _ConcreteOpenAIClient(openAIModelServerClient):
    """`openAIModelServerClient` is abstract only in the two methods below, and neither
    participates in the streaming path under test."""

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat, APIType.Completion]

    def get_prometheus_metric_metadata(self) -> OpenAIMetrics:
        raise NotImplementedError("no server metrics are scraped in the integration tier")


def make_tokenizer() -> MagicMock:
    """Counts whitespace-separated words. Real enough for chunk accounting, and exact,
    so no expected value below depends on a model vocabulary."""
    tokenizer = MagicMock()
    tokenizer.count_tokens = MagicMock(side_effect=lambda text, **kwargs: len(text.split()))
    return tokenizer


async def run_scripted_request(
    script: List[StreamEvent],
    *,
    completion_tokens: Optional[int] = None,
    tokenize_in_reportgen: bool = False,
) -> Tuple[Dict[str, Any], ServedStream, RequestLifecycleMetric]:
    """Serve `script` once and return (summary report contents, what the server did, the
    metric the collector recorded).

    Everything between the socket and the report is the shipped code path: the real
    client session, the real streaming parser, the real collector, and the real report
    generator. Only the tokenizer is substituted, and only to keep token counts exact.

    `tokenize_in_reportgen` decides whether `summarize_requests` re-derives
    `output_token_times` from the raw chunks (the #564 code path) or passes the API
    layer's per-chunk arrival times straight through.
    """
    collector = LocalRequestMetricCollector()

    async with FakeOpenAIServer(script, completion_tokens=completion_tokens) as server:
        # The client would otherwise build a CustomTokenizer by fetching from the Hub.
        with patch.object(openai_client_module, "CustomTokenizer", return_value=make_tokenizer()):
            client = _ConcreteOpenAIClient(
                metrics_collector=collector,
                api_config=APIConfig(type=APIType.Chat, streaming=True),
                uri=f"http://127.0.0.1:{server.port}",
                model_name="fake-model",
                tokenizer_config=None,
                max_tcp_connections=1,
                additional_filters=[],
            )
        try:
            await client.process_request(
                ChatCompletionAPIData(messages=[ChatMessage(role="user", content="prompt")], max_tokens=64),
                stage_id=0,
                scheduled_time=0.0,
            )
        finally:
            await client.close()

    metrics = collector.get_metrics()
    assert len(metrics) == 1, "one request must produce exactly one collected metric"
    metric = metrics[0]
    assert metric.error is None, f"the scripted stream must succeed, got {metric.error}"

    config = Config()
    if tokenize_in_reportgen:
        config.tokenizer = CustomTokenizerConfig(pretrained_model_name_or_path="fake-model")
    generator = ReportGenerator(metrics_client=None, metrics_collector=collector, config=config)
    runtime_parameters = PerfRuntimeParameters(
        start_time=0.0,
        duration=1.0,
        model_server_metrics=BaseMetrics(),
        stages={0: StageRuntimeInfo(stage_id=0, rate=1.0, start_time=0.0, end_time=1.0, status=StageStatus.COMPLETED)},
    )
    report_config = ReportConfig(request_lifecycle=RequestLifecycleMetricsReportConfig(summary=True))

    with patch("inference_perf.utils.custom_tokenizer.CustomTokenizer", return_value=make_tokenizer()):
        reports = await generator.generate_reports(report_config, runtime_parameters)

    summaries = [report for report in reports if report.name == "summary_lifecycle_metrics"]
    assert len(summaries) == 1, "the summary report must be emitted"
    return summaries[0].contents, server.served[-1], metric


def offsets_from(send_times: Sequence[float], start: float) -> List[float]:
    """Server send stamps expressed against the client's request start, which is the
    zero point every reported latency is measured from."""
    return [t - start for t in send_times]


def gaps(values: Sequence[float]) -> List[float]:
    return [b - a for a, b in zip(values, values[1:], strict=False)]


def assert_matches_send_stamp(reported: float, expected: float, label: str) -> None:
    """A value the client derived from chunk arrivals, against the server's send stamp
    for the same chunk. The client cannot see a chunk early, so this is one-sided."""
    assert reported >= expected - CLOCK_EPS, f"{label}: {reported} precedes the server's send stamp {expected}"
    assert reported <= expected + DELIVERY_TOLERANCE, (
        f"{label}: {reported} exceeds send stamp {expected} by more than delivery"
    )


def assert_matches_injection(reported: float, configured: float, label: str) -> None:
    """A reported value straight against the configured script. Looser, because
    `asyncio.sleep` only promises to sleep at least as long as asked."""
    assert reported >= configured - DELIVERY_TOLERANCE, f"{label}: {reported} is below the injected {configured}"
    assert reported <= configured + INJECTION_TOLERANCE, f"{label}: {reported} is above the injected {configured}"


@pytest.mark.asyncio
async def test_injected_pauses_actually_happen_on_the_wire() -> None:
    """Guards the oracle before anything is asserted against it.

    If the fake did not really pause, every accuracy assertion below would still pass
    while measuring nothing. This checks the server's own send stamps against the
    configured script: each gap must be at least what was asked for, and not wildly
    more.
    """
    _, served, metric = await run_scripted_request(SCRIPT, completion_tokens=EXPECTED_OUTPUT_TOKENS)

    sent = offsets_from(served.content_send_times, metric.start_time)
    assert len(sent) == len(CHUNK_TEXTS)
    assert sent[0] >= FIRST_TOKEN_DELAY - CLOCK_EPS
    assert sent[0] <= FIRST_TOKEN_DELAY + SCHEDULING_TOLERANCE

    for observed, configured in zip(gaps(sent), CHUNK_GAPS, strict=True):
        assert observed >= configured - CLOCK_EPS, f"gap {observed} is shorter than the injected {configured}"
        assert observed <= configured + SCHEDULING_TOLERANCE, f"gap {observed} overshot the injected {configured}"


@pytest.mark.asyncio
async def test_reported_ttft_matches_the_injected_first_token_delay() -> None:
    """TTFT must equal the time the first content chunk was put on the wire.

    Checked twice: against the server's send stamp for that chunk (tight) and against
    the 0.25s the script asked for (loose). A TTFT anchored to the wrong event, for
    instance to response headers or to stream completion, fails both.
    """
    summary, served, metric = await run_scripted_request(SCRIPT, completion_tokens=EXPECTED_OUTPUT_TOKENS)

    ttft = summary["successes"]["latency"]["time_to_first_token"]
    assert ttft is not None, "a streaming response must report a TTFT"
    # One request, so mean, min, median and max are all the same single observation.
    for key in ("mean", "min", "median", "max"):
        assert ttft[key] == pytest.approx(ttft["mean"]), key

    first_chunk_offset = offsets_from(served.content_send_times, metric.start_time)[0]
    assert_matches_send_stamp(ttft["mean"], first_chunk_offset, "ttft")
    assert_matches_injection(ttft["mean"], FIRST_TOKEN_DELAY, "ttft")


@pytest.mark.asyncio
async def test_reported_ttft_tracks_a_change_in_the_injected_delay() -> None:
    """Two runs differing only in the first-token pause must differ by that pause.

    A single-script test can pass on a TTFT that happens to be near-constant, for
    instance one measured from the wrong anchor on a short stream. Changing only the
    injected delay isolates it: 0.55s more waiting must show up as 0.55s more TTFT.
    """
    fast_delay, slow_delay = 0.05, 0.60
    tail = [StreamEvent("content", " beta", 0.05)]

    fast_summary, _, _ = await run_scripted_request([StreamEvent("content", "alpha", fast_delay)] + tail)
    slow_summary, _, _ = await run_scripted_request([StreamEvent("content", "alpha", slow_delay)] + tail)

    fast_ttft = fast_summary["successes"]["latency"]["time_to_first_token"]["mean"]
    slow_ttft = slow_summary["successes"]["latency"]["time_to_first_token"]["mean"]

    assert_matches_injection(fast_ttft, fast_delay, "fast ttft")
    assert_matches_injection(slow_ttft, slow_delay, "slow ttft")
    # The difference cancels the fixed connect and parse overhead common to both runs,
    # so it is bounded by scheduling jitter alone.
    assert slow_ttft - fast_ttft == pytest.approx(slow_delay - fast_delay, abs=INJECTION_TOLERANCE)


@pytest.mark.asyncio
async def test_reported_itl_matches_the_injected_per_chunk_gaps() -> None:
    """ITL must reproduce the gap distribution, not just its average.

    The script alternates 0.40s and 0.10s pauses, so the reported summary has to show
    min near 0.10, max near 0.40 and mean near 0.25. #564 is the reason this asserts the
    distribution: that bug halved ITL by inventing extra tokens at already-recorded
    timestamps, which moves the mean while leaving the metric present and plausible.
    """
    summary, served, metric = await run_scripted_request(SCRIPT, completion_tokens=EXPECTED_OUTPUT_TOKENS)

    itl = summary["successes"]["latency"]["inter_token_latency"]
    assert itl is not None

    observed_gaps = gaps(offsets_from(served.content_send_times, metric.start_time))
    assert len(observed_gaps) == len(CHUNK_GAPS)

    # Tight: against what the server actually did.
    assert itl["min"] == pytest.approx(min(observed_gaps), abs=DELIVERY_TOLERANCE)
    assert itl["max"] == pytest.approx(max(observed_gaps), abs=DELIVERY_TOLERANCE)
    assert itl["mean"] == pytest.approx(sum(observed_gaps) / len(observed_gaps), abs=DELIVERY_TOLERANCE)

    # Loose: against the script, which is what a reader of the report cares about.
    assert_matches_injection(itl["min"], min(CHUNK_GAPS), "itl min")
    assert_matches_injection(itl["max"], max(CHUNK_GAPS), "itl max")
    assert_matches_injection(itl["mean"], sum(CHUNK_GAPS) / len(CHUNK_GAPS), "itl mean")
    # Two slow and two fast gaps put the median midway between them: (0.10 + 0.40) / 2.
    assert_matches_injection(itl["median"], 0.25, "itl median")


@pytest.mark.asyncio
async def test_reported_tpot_matches_the_injected_decode_span() -> None:
    """TPOT is the decode span divided by output tokens minus one, and must exclude the
    prefill wait.

    Injected: 5 tokens, first at 0.25s, last at 1.25s, so the span is 1.00s over 4
    intervals and TPOT is 0.25s. Note it is not `request_latency / tokens`: that is
    `normalized_time_per_output_token`, which the same report emits separately and which
    is checked below to be strictly larger because it does include the 0.25s prefill.
    """
    summary, served, metric = await run_scripted_request(SCRIPT, completion_tokens=EXPECTED_OUTPUT_TOKENS)

    assert summary["successes"]["output_len"]["mean"] == EXPECTED_OUTPUT_TOKENS

    tpot = summary["successes"]["latency"]["time_per_output_token"]
    assert tpot is not None

    sent = offsets_from(served.content_send_times, metric.start_time)
    expected_tpot = (sent[-1] - sent[0]) / (EXPECTED_OUTPUT_TOKENS - 1)
    assert tpot["mean"] == pytest.approx(expected_tpot, abs=DELIVERY_TOLERANCE)
    assert_matches_injection(tpot["mean"], sum(CHUNK_GAPS) / (EXPECTED_OUTPUT_TOKENS - 1), "tpot")

    ntpot = summary["successes"]["latency"]["normalized_time_per_output_token"]
    assert ntpot is not None
    assert ntpot["mean"] > tpot["mean"], "NTPOT must include the prefill wait that TPOT excludes"


@pytest.mark.asyncio
async def test_reported_request_latency_matches_the_injected_total() -> None:
    """Request latency must cover the whole injected profile and nothing beyond it.

    The script spends 0.25s before the first token and 1.00s streaming the rest, so the
    injected total is 1.25s. The reported value has to be at least that, at least as
    large as the last send stamp, and within one delivery window of it: anything much
    larger means the client is charging the request for teardown it did not spend.
    """
    summary, served, metric = await run_scripted_request(SCRIPT, completion_tokens=EXPECTED_OUTPUT_TOKENS)

    latency = summary["successes"]["latency"]["request_latency"]
    assert latency is not None

    injected_total = FIRST_TOKEN_DELAY + sum(CHUNK_GAPS)
    assert_matches_injection(latency["mean"], injected_total, "request_latency")

    last_send_offset = offsets_from(served.content_send_times, metric.start_time)[-1]
    assert latency["mean"] >= last_send_offset - CLOCK_EPS, "latency cannot end before the last chunk was sent"
    assert latency["mean"] <= last_send_offset + DELIVERY_TOLERANCE

    # The report's own consistency: the request cannot finish before its first token.
    ttft = summary["successes"]["latency"]["time_to_first_token"]
    assert ttft is not None
    assert latency["mean"] > ttft["mean"]


@pytest.mark.asyncio
async def test_itl_is_not_deflated_when_reportgen_retokenizes_chunks() -> None:
    """The #564 shape, through the path that produced it.

    With a tokenizer configured, `summarize_requests` rebuilds `output_token_times` from
    the raw chunks rather than using the API layer's arrival times. #564 was a
    per-chunk BOS in that rebuild: every one-token chunk counted as two, which duplicated
    each timestamp, inserted a zero-length interval between the copies, and roughly
    halved reported ITL while leaving the metric present and plausible.

    Three chunks of one word each, 0.35s apart, must therefore report ITL near 0.35s.
    Under the #564 behavior the same stream reports near 0.175s.
    """
    gap = 0.35
    script = [
        StreamEvent("content", "alpha", 0.10),
        StreamEvent("content", " beta", gap),
        StreamEvent("content", " gamma", gap),
    ]

    summary, served, metric = await run_scripted_request(script, completion_tokens=3, tokenize_in_reportgen=True)

    itl = summary["successes"]["latency"]["inter_token_latency"]
    assert itl is not None
    observed_gaps = gaps(offsets_from(served.content_send_times, metric.start_time))
    assert itl["mean"] == pytest.approx(sum(observed_gaps) / len(observed_gaps), abs=DELIVERY_TOLERANCE)
    assert_matches_injection(itl["mean"], gap, "itl mean via reportgen tokenizer")
    # No zero-length interval, which is the fingerprint of a duplicated timestamp.
    assert itl["min"] > 0.0, "a zero ITL sample means two tokens were recorded at one arrival time"

    # The client's own accounting agrees with the server's completion_tokens, so the
    # mismatch detector added for #564 stays quiet on a correct stream.
    assert summary["successes"]["token_count_mismatches"] == 0
