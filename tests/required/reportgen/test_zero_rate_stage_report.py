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


# What a run recorded about one stage: its rate or its request_interval, a 3s window, and for
# a concurrent or session stage its concurrency level.
def _stage_info(
    stage_id: int, rate: float | None, concurrency_level: int | None = None, request_interval: str | None = None
) -> StageRuntimeInfo:
    return StageRuntimeInfo(
        stage_id=stage_id,
        rate=rate,
        request_interval=request_interval,
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


# Any rate stage that sent nothing reports, whatever its rate. With requests in stage 0 alone, stage 1
# at rate 4 (it ended before its first send) reports count 0. None is written for stage 2 with a
# concurrency level of 2 (a session or concurrent stage) or for stage -1 (the sweep's probe).
async def test_which_stages_without_requests_get_a_report() -> None:
    stages = {
        -1: _stage_info(-1, 0.0),
        0: _stage_info(0, 4.0),
        1: _stage_info(1, 4.0),
        2: _stage_info(2, 0.0, concurrency_level=2),
    }
    reports = await _reports(stages, [0])
    stage_reports = sorted(name for name in reports if name.startswith("stage_"))
    assert stage_reports == ["stage_0_lifecycle_metrics", "stage_1_lifecycle_metrics"]
    assert reports["stage_1_lifecycle_metrics"].get_contents()["load_summary"]["count"] == 0
    partials = sorted(name for name in reports if name.startswith("inference-perf.partial"))
    assert partials == ["inference-perf.partial.stage_0", "inference-perf.partial.stage_1"]


# A rate too low to reach one request reports like a zero rate. Stages at rate 4, 0.1, 4 with two
# requests in each of stages 0 and 2: stage 1 reports count 0 with requested_rate 0.1 and
# achieved_rate 0, and validation has no errors.
async def test_stage_that_rounds_down_to_zero_requests_reports_a_count_of_zero() -> None:
    reports = await _reports({0: _stage_info(0, 4.0), 1: _stage_info(1, 0.1), 2: _stage_info(2, 4.0)}, [0, 2])
    idle = reports["stage_1_lifecycle_metrics"].get_contents()["load_summary"]
    assert (idle["count"], idle["requested_rate"], idle["achieved_rate"]) == (0, 0.1, 0.0)
    validation = reports["validation"].get_contents()
    assert validation["global"]["errors"] == []
    assert [group["errors"] for group in validation["reports"].values() if group["errors"]] == []


# A stage that sent nothing gets a BR0.2 partial too. For stages at rate 4, 0, 4 the partial of
# stage 1 carries its 3s window (3s to 6s) and empty results, it is validated as a BR0.2 document
# with no errors, and the other two partials still carry request results.
async def test_stage_that_sent_nothing_gets_a_partial() -> None:
    reports = await _reports({0: _stage_info(0, 4.0), 1: _stage_info(1, 0.0), 2: _stage_info(2, 4.0)}, [0, 2])
    partial = reports["inference-perf.partial.stage_1"].get_contents()
    assert partial["results"] == {}
    assert partial["run"]["time"]["duration"] == "PT3.000S"
    assert "request_performance" in reports["inference-perf.partial.stage_0"].get_contents()["results"]
    validation = reports["validation"].get_contents()
    assert validation["reports"]["inference-perf.partial.stage_1.yaml"] == {"warnings": [], "errors": []}


# The warning about a rate lower than the stage before it says when the lower stage is a pause.
# For rates 4, 0, 4 it reads "stage 1 requested_rate 0 < stage 0 requested_rate 4. Stage 1 sent no
# requests, so it is a pause in the load". For rates 4, 2 with requests in both it does not.
async def test_rate_order_warning_names_a_pause() -> None:
    paused = await _reports({0: _stage_info(0, 4.0), 1: _stage_info(1, 0.0), 2: _stage_info(2, 4.0)}, [0, 2])
    messages = [w["message"] for w in paused["validation"].get_contents()["global"]["warnings"]]
    assert messages == [
        "stage requested rates are not monotonically increasing: stage 1 requested_rate 0 < stage 0 "
        "requested_rate 4. Stage 1 sent no requests, so it is a pause in the load"
    ]
    slower = await _reports({0: _stage_info(0, 4.0), 1: _stage_info(1, 2.0)}, [0, 1])
    messages = [w["message"] for w in slower["validation"].get_contents()["global"]["warnings"]]
    assert messages == [
        "stage requested rates are not monotonically increasing: stage 1 requested_rate 2 < stage 0 requested_rate 4"
    ]


# A run whose only stage has a zero rate sent nothing at all: the stage report is written with
# count 0 and so is its partial, and no run summary is, as for any run with no requests.
async def test_run_of_only_a_zero_rate_stage() -> None:
    reports = await _reports({0: _stage_info(0, 0.0)}, [])
    assert reports["stage_0_lifecycle_metrics"].get_contents()["load_summary"]["count"] == 0
    assert "summary_lifecycle_metrics" not in reports
    assert reports["inference-perf.partial.stage_0"].get_contents()["results"] == {}


# A request_interval stage that sent nothing reports too. Stages at rate 4, request_interval "oo", rate 4
# with two requests in each of stages 0 and 2: stage 1 reports count 0 and carries its request_interval
# in place of a requested rate.
async def test_request_interval_stage_that_sent_nothing_reports_a_count_of_zero() -> None:
    stages = {0: _stage_info(0, 4.0), 1: _stage_info(1, None, request_interval="oo"), 2: _stage_info(2, 4.0)}
    reports = await _reports(stages, [0, 2])
    idle = reports["stage_1_lifecycle_metrics"].get_contents()["load_summary"]
    assert (idle["count"], idle["request_interval"], idle["achieved_rate"]) == (0, "oo", 0.0)
    assert "requested_rate" not in idle
    assert reports["validation"].get_contents()["global"]["errors"] == []
