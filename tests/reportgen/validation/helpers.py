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
"""Shared fixtures for report-validation tests.

``make_report_set`` drives the *real* generators (``summarize_requests``,
``build_partial_report``) over synthesized request metrics, so a valid fixture
stays valid when the generators change — and validators are tested against
what the tool actually emits, not against a hand-maintained copy of it.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from inference_perf.apis import ErrorResponseInfo, InferenceInfo, RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import summarize_requests
from inference_perf.reportgen.br.v0_2 import build_partial_report
from inference_perf.utils import ReportFile

PERCENTILES = [50.0, 90.0]

# Request timestamps in these fixtures are small monotonic-clock values (0.0,
# 1.5, 10.0, ...). run.time needs a real epoch window, so the stage window is
# these same offsets shifted onto a fixed wall-clock base. Shifting preserves
# the span, so run.time.duration still equals the stage's benchmark window.
STAGE_EPOCH_BASE = 1750000000.0

SUMMARY_NAME = "summary_lifecycle_metrics"
PER_REQUEST_NAME = "per_request_lifecycle_metrics"


def make_metric(
    stage_id: int,
    start: float,
    end: float,
    *,
    input_tokens: int = 100,
    output_tokens: int = 8,
    failed: bool = False,
) -> RequestLifecycleMetric:
    """One request metric with evenly spaced streamed token timestamps."""
    token_times = [start + (i + 1) * (end - start) / (output_tokens + 1) for i in range(output_tokens)]
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=start - 0.001,
        start_time=start,
        end_time=end,
        request_data="{}",
        response_data=None if failed else "ok",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=input_tokens)),
            response_metrics=None
            if failed
            else StreamedResponseMetrics(
                response_chunks=[],
                chunk_times=list(token_times),
                output_tokens=output_tokens,
                output_token_times=list(token_times),
                server_usage={"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
            ),
        ),
        error=ErrorResponseInfo(error_type="HTTP Error 500", error_msg="boom") if failed else None,
    )


def make_stage_metrics() -> Dict[int, List[RequestLifecycleMetric]]:
    """Two stages: stage 0 has 3 successes + 1 failure, stage 1 has 2 successes."""
    return {
        0: [
            make_metric(0, 0.0, 1.0, input_tokens=100, output_tokens=8),
            make_metric(0, 0.5, 1.8, input_tokens=120, output_tokens=12),
            make_metric(0, 1.0, 2.2, input_tokens=90, output_tokens=6),
            make_metric(0, 1.5, 2.0, failed=True),
        ],
        1: [
            make_metric(1, 10.0, 11.0, input_tokens=100, output_tokens=8),
            make_metric(1, 10.5, 12.0, input_tokens=110, output_tokens=10),
        ],
    }


def per_request_record(metric: RequestLifecycleMetric) -> Dict[str, Any]:
    """Mirror of the record shape emitted by ``generate_reports``."""
    return {
        "start_time": metric.start_time,
        "end_time": metric.end_time,
        "request": metric.request_data,
        "response": metric.response_data,
        "info": metric.info.model_dump() if metric.info else None,
        "error": metric.error.model_dump() if metric.error else None,
    }


def make_report_set(
    stage_metrics: Optional[Dict[int, List[RequestLifecycleMetric]]] = None,
    *,
    with_br_partials: bool = True,
) -> List[ReportFile]:
    """A fully consistent report set, built through the real generators."""
    stages = make_stage_metrics() if stage_metrics is None else stage_metrics
    all_metrics = [m for metrics in stages.values() for m in metrics]

    reports = [
        ReportFile(
            name=SUMMARY_NAME,
            contents=summarize_requests(all_metrics, PERCENTILES).model_dump(),
        ),
        ReportFile(
            name=PER_REQUEST_NAME,
            contents=[per_request_record(m) for m in all_metrics],
        ),
        ReportFile(
            name="config",
            file_type="yaml",
            contents={"load": {"stages": [{"rate": 1, "duration": 5} for _ in stages]}},
        ),
    ]
    for stage_id, metrics in stages.items():
        reports.append(
            ReportFile(
                name=f"stage_{stage_id}_lifecycle_metrics",
                contents=summarize_requests(metrics, PERCENTILES, stage_rate=1.0).model_dump(),
            )
        )
        if with_br_partials:
            reports.append(
                ReportFile(
                    name=f"inference-perf.partial.stage_{stage_id}",
                    file_type="yaml",
                    contents=build_partial_report(
                        metrics,
                        tokenizer=None,
                        run_uid=f"test-uid-{stage_id}",
                        stage_start=STAGE_EPOCH_BASE + min(m.start_time for m in metrics),
                        stage_end=STAGE_EPOCH_BASE + max(m.end_time for m in metrics),
                    ),
                )
            )
    return reports


def replace_contents(reports: List[ReportFile], filename: str, contents: Any) -> List[ReportFile]:
    """A copy of the report set with one file's contents replaced."""
    replaced = []
    for report in reports:
        if report.get_filename() == filename:
            replaced.append(ReportFile(name=report.name, contents=contents, file_type=report.file_type))
        else:
            replaced.append(report)
    return replaced


def tampered(reports: List[ReportFile], filename: str) -> tuple[List[ReportFile], Any]:
    """A copy of the report set plus a deep-copied contents dict to corrupt.

    The returned contents object is already wired into the returned set, so
    tests mutate it in place and validate the set.
    """
    original = next(r for r in reports if r.get_filename() == filename)
    contents = copy.deepcopy(original.get_contents())
    return replace_contents(reports, filename, contents), contents
