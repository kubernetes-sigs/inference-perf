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
"""Integration test for issue #531 (#606 Integration tier, per-change lane).

#530 fixed the bug: ``parse_sse_stream`` now wraps a mid-stream failure in
``StreamInterruptedError`` carrying the bytes read so far, and
``process_request`` recovers them into ``response_data`` so the per-request
report still shows what the server sent. Nothing gates that recovery, and an
ungated fix is how #410 reached the #564 postmortem.

These tests drive the real client against a fake that returns 200, writes
valid SSE frames, then ends the connection short of its declared
``Content-Length``. That is the reproduction described in #531, and it raises a
genuine ``aiohttp.ClientPayloadError`` inside the client, so the recovery
branch is exercised through the real exception path rather than a patched-in
one.
"""

import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from fake_truncating_server import TruncatingSSEServer

from inference_perf.apis import RequestLifecycleMetric
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.client.modelserver import openai_client as openai_client_module
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.client.modelserver.openai_client import OpenAIMetrics, openAIModelServerClient
from inference_perf.client.server_metrics.base import PerfRuntimeParameters, StageRuntimeInfo, StageStatus
from inference_perf.config import APIConfig, APIType, ReportConfig, RequestLifecycleMetricsReportConfig
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector
from inference_perf.reportgen.base import ReportGenerator

# Two well-formed frames, so a break after them leaves a body that is both
# non-empty and recognizably SSE.
EVENTS = [
    '{"choices":[{"text":"Hello "}]}',
    '{"choices":[{"text":"world "}]}',
]


class _ConcreteOpenAIClient(openAIModelServerClient):
    """openAIModelServerClient is abstract only in the two methods below, and
    neither participates in the streaming path under test."""

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat, APIType.Completion]

    def get_prometheus_metric_metadata(self) -> OpenAIMetrics:
        raise NotImplementedError("no server metrics are scraped in the integration tier")


def make_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.count_tokens = MagicMock(side_effect=lambda text, **kwargs: len(text.split()))
    return tokenizer


async def run_request_against(server: TruncatingSSEServer) -> RequestLifecycleMetric:
    """One streaming completion through the real client, returning the metric
    the client recorded for it."""
    collector = LocalRequestMetricCollector()
    # The client builds a CustomTokenizer, which would otherwise fetch from the
    # Hub; token counts are irrelevant to this test, only the response body is.
    with patch.object(openai_client_module, "CustomTokenizer", return_value=make_tokenizer()):
        client = _ConcreteOpenAIClient(
            metrics_collector=collector,
            api_config=APIConfig(type=APIType.Completion, streaming=True),
            uri=server.base_url,
            model_name="fake-model",
            tokenizer_config=None,
            max_tcp_connections=1,
            additional_filters=[],
        )

    session = client.new_session()
    try:
        await session.process_request(
            CompletionAPIData(prompt="the quick brown fox", max_tokens=16),
            stage_id=0,
            scheduled_time=time.perf_counter(),
        )
    finally:
        await session.close()

    metrics = collector.get_metrics()
    assert len(metrics) == 1, "the client must record exactly one metric for one request"
    return metrics[0]


@pytest.mark.asyncio
async def test_partial_body_is_preserved_when_the_stream_breaks() -> None:
    """The #531 regression: bytes received before the break must reach
    response_data, byte for byte."""
    async with TruncatingSSEServer(EVENTS) as server:
        metric = await run_request_against(server)

    assert metric.response_data == server.sent_body, "the partial body must be preserved exactly as sent"
    # Guards against a future "fix" that stores the parsed text instead: the
    # value has to be the raw wire bytes, framing included, or it is useless
    # for diagnosing a malformed stream.
    assert "data: " in (metric.response_data or ""), "response_data must hold raw SSE framing, not reassembled output"
    assert "Hello " in (metric.response_data or "")


@pytest.mark.asyncio
async def test_underlying_exception_is_reported_not_the_wrapper() -> None:
    """StreamInterruptedError is plumbing. The report must name the error the
    client actually hit, or the bucket in build_error_counts is useless."""
    async with TruncatingSSEServer(EVENTS) as server:
        metric = await run_request_against(server)

    assert metric.error is not None, "a broken stream must be recorded as a failure"
    assert metric.error.error_type == "ClientPayloadError"
    assert metric.error.error_type != "StreamInterruptedError"


@pytest.mark.asyncio
async def test_break_before_any_body_byte_leaves_the_response_empty() -> None:
    """The empty-raw_content branch: with nothing received there is nothing to
    preserve, and the 200 path must not fall through to the placeholder text
    the non-200 path writes."""
    async with TruncatingSSEServer([]) as server:
        metric = await run_request_against(server)

    assert metric.response_data == ""
    assert metric.error is not None
    assert metric.error.error_type == "ClientPayloadError"
    assert "Failed to read response text" not in (metric.response_data or "")


@pytest.mark.asyncio
async def test_per_request_report_carries_the_partial_body() -> None:
    """End of the chain: the preserved bytes have to survive into
    per_request_lifecycle_metrics, which is the escape hatch #531 is about."""
    async with TruncatingSSEServer(EVENTS) as server:
        metric = await run_request_against(server)
        sent_body = server.sent_body

    config = MagicMock()
    config.tokenizer = None
    config.model_dump = MagicMock(return_value={})
    generator = ReportGenerator(
        metrics_client=None,
        metrics_collector=MagicMock(get_metrics=MagicMock(return_value=[metric])),
        config=config,
    )
    runtime_parameters = PerfRuntimeParameters(
        start_time=0.0,
        duration=1.0,
        model_server_metrics=BaseMetrics(),
        stages={0: StageRuntimeInfo(stage_id=0, rate=1.0, start_time=0.0, end_time=1.0, status=StageStatus.COMPLETED)},
    )
    report_config = ReportConfig(request_lifecycle=RequestLifecycleMetricsReportConfig(per_request=True))

    reports = await generator.generate_reports(report_config, runtime_parameters)

    per_request = [report for report in reports if report.name == "per_request_lifecycle_metrics"]
    assert len(per_request) == 1, "per_request: true must emit the per-request report"
    entries: List[Dict[str, Any]] = per_request[0].contents
    assert len(entries) == 1
    assert entries[0]["response"] == sent_body, "the per-request entry must carry the partial body, not an empty string"
    assert entries[0]["error"]["error_type"] == "ClientPayloadError"
