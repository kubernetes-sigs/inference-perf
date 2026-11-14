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
"""The capacity probe's measurements must reach the report set, qualified.

r_sat on its own cannot be acted on: a ladder stopped by the concurrency cap or
by client saturation reports a lower bound rather than a measured plateau, so
the report has to carry the stop reason and the rung curve that produced it.
"""

from typing import Optional
from unittest.mock import Mock

from inference_perf.apis.base import InferenceInfo, RequestLifecycleMetric
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.client.server_metrics.base import PerfRuntimeParameters, StageRuntimeInfo, StageStatus
from inference_perf.config.reportgen.config import ReportConfig
from inference_perf.loadgen.probe import (
    BoundConstant,
    ConfidenceInterval,
    ProbeResult,
    ProbeStopReason,
    RungResult,
)
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


def _runtime_parameters(probe_result: Optional[ProbeResult]) -> PerfRuntimeParameters:
    return PerfRuntimeParameters(
        start_time=0.0,
        duration=1.0,
        model_server_metrics=BaseMetrics(),
        stages={0: StageRuntimeInfo(stage_id=0, rate=1.0, start_time=0.0, end_time=1.0, status=StageStatus.COMPLETED)},
        probe_result=probe_result,
    )


def _generator() -> ReportGenerator:
    config = Mock()
    config.tokenizer = None
    config.model_dump = Mock(return_value={})
    return ReportGenerator(
        metrics_client=None,
        metrics_collector=Mock(get_metrics=Mock(return_value=[_request_metric()])),
        config=config,
    )


async def test_generate_reports_writes_the_probe_as_its_own_report() -> None:
    """A probe run emits sweep_probe with the stop reason, the censored flag and
    the full rung curve, so a lower bound is never read as a measured plateau."""
    probe_result = ProbeResult(
        rungs=(
            RungResult(concurrency=1, throughput=10.0),
            RungResult(concurrency=2, throughput=20.0, error_rate=0.25),
        ),
        constants={
            "r_sat": BoundConstant(name="r_sat", value=20.0, ci=ConfidenceInterval(18.0, 22.0)),
            "n_knee": BoundConstant(name="n_knee", value=2.0, ci=ConfidenceInterval(1.0, 2.0)),
        },
        stop_reason=ProbeStopReason.CLIENT_BOUND,
        usable_concurrencies=(1,),
    )

    reports = await _generator().generate_reports(ReportConfig(), _runtime_parameters(probe_result))

    probe_reports = [report for report in reports if report.name == "sweep_probe"]
    assert len(probe_reports) == 1
    contents = probe_reports[0].contents
    assert contents["stop_reason"] == "client_bound"
    assert contents["censored"] is True
    assert contents["constants"]["r_sat"]["value"] == 20.0
    assert [rung["concurrency"] for rung in contents["rungs"]] == [1, 2]
    assert [rung["usable"] for rung in contents["rungs"]] == [True, False]


async def test_generate_reports_omits_the_probe_report_without_a_probe() -> None:
    """A run that did not measure its sweep with the probe gets no sweep_probe
    file, rather than an empty one that implies a measurement happened."""
    reports = await _generator().generate_reports(ReportConfig(), _runtime_parameters(None))

    assert not any(report.name == "sweep_probe" for report in reports)
