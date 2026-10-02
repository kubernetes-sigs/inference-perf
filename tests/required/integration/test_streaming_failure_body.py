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

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from fake_truncating_server import TruncatingSSEServer
from openai_client_harness import run_request_against

from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.client.server_metrics.base import PerfRuntimeParameters, StageRuntimeInfo, StageStatus
from inference_perf.config import ReportConfig, RequestLifecycleMetricsReportConfig
from inference_perf.reportgen.base import ReportGenerator

# Two well-formed frames, so a break after them leaves a body that is both
# non-empty and recognizably SSE.
EVENTS = [
    '{"choices":[{"text":"Hello "}]}',
    '{"choices":[{"text":"world "}]}',
]


@pytest.mark.asyncio
async def test_partial_body_is_preserved_when_the_stream_breaks() -> None:
    """The #531 regression: bytes received before the break must reach
    response_data, byte for byte."""
    async with TruncatingSSEServer(EVENTS) as server:
        metric = await run_request_against(server.base_url)

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
        metric = await run_request_against(server.base_url)

    assert metric.error is not None, "a broken stream must be recorded as a failure"
    assert metric.error.error_type == "ClientPayloadError"
    assert metric.error.error_type != "StreamInterruptedError"


@pytest.mark.asyncio
async def test_break_before_any_body_byte_leaves_the_response_empty() -> None:
    """The empty-raw_content branch: with nothing received there is nothing to
    preserve, and the 200 path must not fall through to the placeholder text
    the non-200 path writes."""
    async with TruncatingSSEServer([]) as server:
        metric = await run_request_against(server.base_url)

    assert metric.response_data == ""
    assert metric.error is not None
    assert metric.error.error_type == "ClientPayloadError"
    assert "Failed to read response text" not in (metric.response_data or "")


@pytest.mark.asyncio
async def test_per_request_report_carries_the_partial_body() -> None:
    """End of the chain: the preserved bytes have to survive into
    per_request_lifecycle_metrics, which is the escape hatch #531 is about."""
    async with TruncatingSSEServer(EVENTS) as server:
        metric = await run_request_against(server.base_url)
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
