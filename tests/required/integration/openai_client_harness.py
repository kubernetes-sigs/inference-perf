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
"""Drives one request through the real ``openAIModelServerClient`` against a
fake server and hands back the ``RequestLifecycleMetric`` it recorded.

Shared by the integration tests that fake a misbehaving server (#531, #655,
#713): they differ in what the server sends, not in how the client is driven.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

from inference_perf.apis import InferenceAPIData, RequestLifecycleMetric
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.client.modelserver import openai_client as openai_client_module
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.client.modelserver.openai_client import OpenAIMetrics, openAIModelServerClient
from inference_perf.client.server_metrics.base import PerfRuntimeParameters, StageRuntimeInfo, StageStatus
from inference_perf.config import APIConfig, APIType, ReportConfig, RequestLifecycleMetricsReportConfig
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector
from inference_perf.reportgen.base import ReportGenerator


class ConcreteOpenAIClient(openAIModelServerClient):
    """openAIModelServerClient is abstract only in the two methods below, and
    neither participates in the request path under test."""

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat, APIType.Completion]

    def get_prometheus_metric_metadata(self) -> OpenAIMetrics:
        raise NotImplementedError("no server metrics are scraped in the integration tier")


# A tokenizer stand-in that counts whitespace-separated words, so no Hub download.
def make_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.count_tokens = MagicMock(side_effect=lambda text, **kwargs: len(text.split()))
    return tokenizer


# Sends one request (a 16-token completion by default) through the real client to
# `base_url` with the given API config and ignore_eos setting (on by default, as
# in the config), and returns the single metric the client recorded for it.
# Fails if the client recorded zero or several metrics.
async def run_request_against(
    base_url: str,
    api_config: Optional[APIConfig] = None,
    data: Optional[InferenceAPIData] = None,
    ignore_eos: bool = True,
) -> RequestLifecycleMetric:
    """One request through the real client, returning the metric it recorded."""
    collector = LocalRequestMetricCollector()
    # The client builds a CustomTokenizer, which would otherwise fetch from the
    # Hub; token counts are irrelevant to these tests, only the response body is.
    with patch.object(openai_client_module, "CustomTokenizer", return_value=make_tokenizer()):
        client = ConcreteOpenAIClient(
            metrics_collector=collector,
            api_config=api_config or APIConfig(type=APIType.Completion, streaming=True),
            uri=base_url,
            model_name="fake-model",
            tokenizer_config=None,
            max_tcp_connections=1,
            additional_filters=[],
            ignore_eos=ignore_eos,
        )

    session = client.new_session()
    try:
        await session.process_request(
            data or CompletionAPIData(prompt="the quick brown fox", max_tokens=16),
            stage_id=0,
            scheduled_time=time.perf_counter(),
        )
    finally:
        await session.close()

    metrics = collector.get_metrics()
    assert len(metrics) == 1, "the client must record exactly one metric for one request"
    return metrics[0]


# Runs the real report generator over the given recorded metrics (one stage,
# summary + per-request enabled) and returns (summary contents, per-request
# entries). Fails unless exactly one of each report was produced.
async def generate_reports(metrics: List[RequestLifecycleMetric]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    config = MagicMock()
    config.tokenizer = None
    config.model_dump = MagicMock(return_value={})
    generator = ReportGenerator(
        metrics_client=None,
        metrics_collector=MagicMock(get_metrics=MagicMock(return_value=list(metrics))),
        config=config,
    )
    runtime_parameters = PerfRuntimeParameters(
        start_time=0.0,
        duration=1.0,
        model_server_metrics=BaseMetrics(),
        stages={0: StageRuntimeInfo(stage_id=0, rate=1.0, start_time=0.0, end_time=1.0, status=StageStatus.COMPLETED)},
    )
    report_config = ReportConfig(request_lifecycle=RequestLifecycleMetricsReportConfig(summary=True, per_request=True))

    reports = await generator.generate_reports(report_config, runtime_parameters)

    summary = [r for r in reports if r.name == "summary_lifecycle_metrics"]
    per_request = [r for r in reports if r.name == "per_request_lifecycle_metrics"]
    assert len(summary) == 1 and len(per_request) == 1
    return summary[0].contents, per_request[0].contents
