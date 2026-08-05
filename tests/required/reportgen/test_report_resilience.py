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
"""generate_reports must not lose the lifecycle reports of a completed run when the
Prometheus section fails: the load has already been sent by then, so a failure there
may only cost the Prometheus reports."""

import logging
from unittest.mock import Mock, patch

import pytest

from inference_perf.apis.base import InferenceInfo, RequestLifecycleMetric
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.client.server_metrics.base import PerfRuntimeParameters, StageRuntimeInfo, StageStatus
from inference_perf.config.reportgen.config import ReportConfig
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import ReportGenerator


def _request_metric(stage_id: int = 0) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=1.0,
        request_data="prompt",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=10))),
        error=None,
    )


async def test_generate_reports_survives_prometheus_report_failure(caplog: pytest.LogCaptureFixture) -> None:
    """A raise out of generate_prometheus_metrics_report costs only the Prometheus section:
    the lifecycle reports are still returned and the failure is logged."""
    config = Mock()
    config.tokenizer = None
    config.model_dump = Mock(return_value={})
    generator = ReportGenerator(
        metrics_client=None,
        metrics_collector=Mock(get_metrics=Mock(return_value=[_request_metric()])),
        config=config,
    )
    runtime_parameters = PerfRuntimeParameters(
        start_time=0.0,
        duration=1.0,
        model_server_metrics=BaseMetrics(),
        stages={0: StageRuntimeInfo(stage_id=0, rate=1.0, start_time=0.0, end_time=1.0, status=StageStatus.COMPLETED)},
    )

    with patch.object(ReportGenerator, "generate_prometheus_metrics_report", side_effect=RuntimeError("boom")):
        with caplog.at_level(logging.ERROR, logger="inference_perf.reportgen.base"):
            reports = await generator.generate_reports(ReportConfig(), runtime_parameters)

    names = [report.name for report in reports]
    assert "summary_lifecycle_metrics" in names
    assert "config" in names
    assert not any("prometheus" in name for name in names)
    assert "Prometheus metrics report generation failed" in caplog.text
