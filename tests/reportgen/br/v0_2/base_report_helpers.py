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
"""Shared fixtures for the base-report tests: a two-stage run through the real ``generate_reports``."""

import asyncio
from typing import Dict, List, Union
from unittest.mock import Mock

from inference_perf.apis import ErrorResponseInfo, InferenceInfo, RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.client.server_metrics import PerfRuntimeParameters
from inference_perf.client.server_metrics.base import StageRuntimeInfo, StageStatus
from inference_perf.config import Config
from inference_perf.config.reportgen.config import ReportConfig
from inference_perf.metrics.request_collector import RequestMetricCollector
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen import ReportGenerator

# Request timestamps are small offsets. The stage window needs epoch seconds, so
# it is the same offsets on a fixed wall-clock base.
STAGE_EPOCH_BASE = 1750000000.0


# A request from start to end with 8 streamed tokens evenly spaced inside it,
# or a failed request with an HTTP 500 when failed=True.
def make_metric(stage_id: int, start: float, end: float, *, failed: bool = False) -> RequestLifecycleMetric:
    token_times = [start + (i + 1) * (end - start) / 9 for i in range(8)]
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=start - 0.001,
        start_time=start,
        end_time=end,
        request_data="{}",
        response_data=None if failed else "ok",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=100)),
            response_metrics=None
            if failed
            else StreamedResponseMetrics(
                response_chunks=[],
                chunk_times=list(token_times),
                output_tokens=8,
                output_token_times=list(token_times),
            ),
        ),
        error=ErrorResponseInfo(error_type="HTTP Error 500", error_msg="boom") if failed else None,
    )


# Stage 0 has 3 successes and 1 failure at 0-2.2s. Stage 1 has 2 successes at 10-12s.
def make_stage_metrics() -> Dict[int, List[RequestLifecycleMetric]]:
    return {
        0: [
            make_metric(0, 0.0, 1.0),
            make_metric(0, 0.5, 1.8),
            make_metric(0, 1.0, 2.2),
            make_metric(0, 1.5, 2.0, failed=True),
        ],
        1: [make_metric(1, 10.0, 11.0), make_metric(1, 10.5, 12.0)],
    }


# Runs the real generate_reports over make_stage_metrics() with report_config and
# returns {filename: contents}, e.g. "inference-perf.partial.stage_0.yaml" and "validation.json".
def generate_reports(report_config: ReportConfig) -> Dict[str, object]:
    stage_metrics = make_stage_metrics()
    collector = Mock(spec=RequestMetricCollector)
    collector.get_metrics.return_value = [m for metrics in stage_metrics.values() for m in metrics]
    config = Mock(spec=Config)
    config.tokenizer = None
    config.model_dump.return_value = {"load": {"stages": [{"rate": 1.0}, {"rate": 1.0}]}}
    generator = ReportGenerator(metrics_client=None, metrics_collector=collector, config=config)
    stages = {
        stage_id: StageRuntimeInfo(
            stage_id=stage_id,
            rate=1.0,
            start_time=STAGE_EPOCH_BASE + min(m.start_time for m in metrics),
            end_time=STAGE_EPOCH_BASE + max(m.end_time for m in metrics),
            status=StageStatus.COMPLETED,
        )
        for stage_id, metrics in stage_metrics.items()
    }
    runtime_parameters = PerfRuntimeParameters(start_time=0.0, duration=12.0, model_server_metrics=Mock(), stages=stages)
    reports = asyncio.run(generator.generate_reports(report_config, runtime_parameters))
    return {report.get_filename(): report.get_contents() for report in reports}


# The partial files out of generate_reports' output, e.g. {0: {...}, 1: {...}}.
def partials(reports: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    prefix = "inference-perf.partial.stage_"
    found: Dict[int, Dict[str, object]] = {}
    for name, contents in reports.items():
        if name.startswith(prefix):
            assert isinstance(contents, dict), name
            found[int(name[len(prefix) : -len(".yaml")])] = contents
    return found


# Every error validation.json recorded against a partial file. A clean run gives [].
def partial_validation_errors(reports: Dict[str, object]) -> List[object]:
    groups = at(reports, "validation.json", "reports")
    assert isinstance(groups, dict)
    errors: List[object] = []
    for name in groups:
        if name.startswith("inference-perf.partial."):
            found = at(groups, name, "errors")
            assert isinstance(found, list)
            errors.extend(found)
    return errors


# The value under a path of keys and list indexes, asserting each step exists.
# at({"pods": [{"name": "a"}]}, "pods", 0, "name") == "a".
def at(document: object, *path: Union[str, int]) -> object:
    node = document
    for step in path:
        if isinstance(step, int):
            assert isinstance(node, list) and step < len(node), f"no [{step}] in {node!r}"
            node = node[step]
        else:
            assert isinstance(node, dict) and step in node, f"no {step!r} in {node!r}"
            node = node[step]
    return node
