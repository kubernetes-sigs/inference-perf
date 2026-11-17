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
"""A stage that sent nothing because its rate was zero still gets a stage report.

No request metric names such a stage, so it has to be found from what the run
recorded about the stage, and the summary has to hold with no request in it.
"""

from unittest.mock import Mock

from inference_perf.apis.base import InferenceInfo, RequestLifecycleMetric
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.client.server_metrics.base import PerfRuntimeParameters, StageRuntimeInfo, StageStatus
from inference_perf.config.reportgen.config import ReportConfig
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import ReportGenerator, summarize_requests
from inference_perf.utils import ReportFile


# One successful request in the given stage, sent at `start` and answered 0.5s later.
def _request_metric(stage_id: int, start: float) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=start,
        start_time=start,
        end_time=start + 0.5,
        request_data="prompt",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=10))),
        error=None,
    )


# What a run recorded about one stage: its rate, a 3s window, and for a concurrent or
# session stage its concurrency level.
def _stage_info(stage_id: int, rate: float, concurrency_level: int | None = None) -> StageRuntimeInfo:
    return StageRuntimeInfo(
        stage_id=stage_id,
        rate=rate,
        start_time=3.0 * stage_id,
        end_time=3.0 * stage_id + 3.0,
        status=StageStatus.COMPLETED,
        concurrency_level=concurrency_level,
    )


# Generates the default report set for the given stages, with two requests one second apart
# in each stage listed in `stages_with_requests`, and returns the reports by name.
async def _reports(stages: dict[int, StageRuntimeInfo], stages_with_requests: list[int]) -> dict[str, ReportFile]:
    metrics = [_request_metric(stage, start) for stage in stages_with_requests for start in (0.0, 1.0)]
    config = Mock()
    config.tokenizer = None
    config.model_dump = Mock(return_value={})
    generator = ReportGenerator(
        metrics_client=None, metrics_collector=Mock(get_metrics=Mock(return_value=metrics)), config=config
    )
    runtime_parameters = PerfRuntimeParameters(start_time=0.0, duration=9.0, model_server_metrics=BaseMetrics(), stages=stages)
    reports = await generator.generate_reports(ReportConfig(), runtime_parameters)
    return {report.name: report for report in reports}


# No requests at a requested rate of 0: the summary holds instead of raising on an empty list.
# Pins count=0, send_duration=0.0, achieved_rate=0.0, requested_rate=0.0, benchmark time 0,
# no schedule_delay, and 0 successes and failures.
def test_summary_of_no_requests() -> None:
    summary = summarize_requests([], [50, 90], stage_rate=0.0).model_dump()
    assert summary["load_summary"] == {
        "count": 0,
        "schedule_delay": None,
        "send_duration": 0.0,
        "requested_rate": 0.0,
        "achieved_rate": 0.0,
    }
    assert summary["benchmark_time_seconds"] == 0.0
    assert (summary["successes"]["count"], summary["failures"]["count"]) == (0, 0)
    assert summary["successes"]["throughput"]["requests_per_sec"] == 0.0


# Stages at rate 4, 0, 4 with two requests in each of stages 0 and 2: all three stage reports
# are written, with counts 2, 0, 2, the run summary counts 4, and validation has no errors and
# no warning about a stage without a report.
async def test_zero_rate_stage_reports_a_count_of_zero() -> None:
    reports = await _reports({0: _stage_info(0, 4.0), 1: _stage_info(1, 0.0), 2: _stage_info(2, 4.0)}, [0, 2])
    counts = [reports[f"stage_{stage}_lifecycle_metrics"].get_contents()["load_summary"]["count"] for stage in range(3)]
    assert counts == [2, 0, 2]
    idle = reports["stage_1_lifecycle_metrics"].get_contents()["load_summary"]
    assert (idle["requested_rate"], idle["achieved_rate"], idle["send_duration"]) == (0.0, 0.0, 0.0)
    assert reports["summary_lifecycle_metrics"].get_contents()["load_summary"]["count"] == 4

    validation = reports["validation"].get_contents()
    assert validation["global"]["errors"] == []
    assert [group["errors"] for group in validation["reports"].values() if group["errors"]] == []
    assert "stage_1_lifecycle_metrics.json" in validation["reports"]
    assert not any(warning["check"] == "cross_report.stage_indices" for warning in validation["global"]["warnings"])


# Only a zero rate earns the empty report. With requests in stage 0 alone, none is written for
# stage 1 at rate 4 with no requests, stage 2 at rate 0 with a concurrency level of 2 (a session
# or concurrent stage records its rate that way), or stage -1 at rate 0 (the sweep's probe).
async def test_other_stages_without_requests_get_no_report() -> None:
    stages = {
        -1: _stage_info(-1, 0.0),
        0: _stage_info(0, 4.0),
        1: _stage_info(1, 4.0),
        2: _stage_info(2, 0.0, concurrency_level=2),
    }
    reports = await _reports(stages, [0])
    stage_reports = sorted(name for name in reports if name.startswith("stage_"))
    assert stage_reports == ["stage_0_lifecycle_metrics"]


# A run whose only stage has a zero rate sent nothing at all: the stage report is written with
# count 0, and no run summary is, as for any run with no requests.
async def test_run_of_only_a_zero_rate_stage() -> None:
    reports = await _reports({0: _stage_info(0, 0.0)}, [])
    assert reports["stage_0_lifecycle_metrics"].get_contents()["load_summary"]["count"] == 0
    assert "summary_lifecycle_metrics" not in reports
