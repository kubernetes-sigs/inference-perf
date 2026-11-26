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
"""Unit tests for the lifecycle reports sliced by request labels (``group_by_labels``)."""

from typing import Dict, List, Optional
from unittest.mock import MagicMock

from inference_perf.apis import ErrorResponseInfo, InferenceInfo, RequestLifecycleMetric
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.client.server_metrics.base import PerfRuntimeParameters, StageRuntimeInfo, StageStatus
from inference_perf.config import ReportConfig, RequestLifecycleMetricsReportConfig
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import (
    MISSING_LABEL_VALUE,
    ReportGenerator,
    group_metrics_by_labels,
    label_slice_report_name,
)
from inference_perf.reportgen.validation import VALIDATION_REPORT_NAME
from inference_perf.utils import ReportFile
from inference_perf.utils.cli_summary import is_label_slice_report


# Builds one 1s request in stage 0 carrying the given labels; it failed with HTTP 500 when failed=True.
def _req(labels: Optional[Dict[str, str]] = None, *, failed: bool = False) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=1.0,
        request_data="prompt",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=10)), labels=labels or {}),
        error=ErrorResponseInfo(error_type="HTTP Error 500", error_msg="Internal Server Error") if failed else None,
    )


# Runs generate_reports over the given requests with the given group_by_labels and returns every report.
async def _generate(metrics: List[RequestLifecycleMetric], group_by_labels: Optional[List[List[str]]]) -> List[ReportFile]:
    config = MagicMock()
    config.tokenizer = None
    config.model_dump = MagicMock(return_value={})
    generator = ReportGenerator(
        metrics_client=None,
        metrics_collector=MagicMock(get_metrics=MagicMock(return_value=metrics)),
        config=config,
    )
    runtime_parameters = PerfRuntimeParameters(
        start_time=0.0,
        duration=1.0,
        model_server_metrics=BaseMetrics(),
        stages={0: StageRuntimeInfo(stage_id=0, rate=1.0, start_time=0.0, end_time=1.0, status=StageStatus.COMPLETED)},
    )
    report_config = ReportConfig(request_lifecycle=RequestLifecycleMetricsReportConfig(group_by_labels=group_by_labels))
    return await generator.generate_reports(report_config, runtime_parameters)


# Keeps only the label-slice reports, keyed by report name.
def _slices(reports: List[ReportFile]) -> Dict[str, ReportFile]:
    return {r.name: r for r in reports if is_label_slice_report(r.name)}


# Three requests with tenant a, a, b grouped by ["tenant"] give two slices: ("a",) with 2 requests, ("b",) with 1.
def test_group_by_one_label() -> None:
    metrics = [_req({"tenant": "a"}), _req({"tenant": "a"}), _req({"tenant": "b"})]
    groups = group_metrics_by_labels(metrics, ["tenant"])
    assert {key: len(value) for key, value in groups.items()} == {("a",): 2, ("b",): 1}


# Grouping by ["tenant", "priority"] keys each slice by the value pair in that order, one slice per pair seen.
def test_group_by_two_labels_is_joint() -> None:
    metrics = [
        _req({"tenant": "a", "priority": "high"}),
        _req({"tenant": "a", "priority": "low"}),
        _req({"tenant": "b", "priority": "high"}),
        _req({"tenant": "a", "priority": "high"}),
    ]
    groups = group_metrics_by_labels(metrics, ["tenant", "priority"])
    assert {key: len(value) for key, value in groups.items()} == {("a", "high"): 2, ("a", "low"): 1, ("b", "high"): 1}


# A request with no "tenant" label lands in the MISSING_LABEL_VALUE slice instead of being dropped.
def test_missing_label_uses_fallback_value() -> None:
    metrics = [_req({"tenant": "a"}), _req({"priority": "high"}), _req()]
    groups = group_metrics_by_labels(metrics, ["tenant"])
    assert {key: len(value) for key, value in groups.items()} == {("a",): 1, (MISSING_LABEL_VALUE,): 2}


# {"tenant": "a", "priority": "high"} is named labels_tenant=a_priority=high_lifecycle_metrics.
def test_report_name_carries_keys_and_values() -> None:
    assert label_slice_report_name({"tenant": "a", "priority": "high"}) == "labels_tenant=a_priority=high_lifecycle_metrics"


# The same value under two different keys gives two different names, so neither report overwrites the other.
def test_report_name_differs_by_key() -> None:
    assert label_slice_report_name({"tenant": "default"}) != label_slice_report_name({"priority": "default"})


# A value of "../a/b" is percent-encoded: the name has no path separator and "a/b" and "a_b" stay distinct.
def test_report_name_encodes_path_characters() -> None:
    name = label_slice_report_name({"tenant": "../a/b"})
    assert "/" not in name
    assert name == "labels_tenant=..%2Fa%2Fb_lifecycle_metrics"
    assert name != label_slice_report_name({"tenant": ".._a_b"})


# Without group_by_labels the run emits no label-slice report.
async def test_no_slices_by_default() -> None:
    reports = await _generate([_req({"tenant": "a"})], None)
    assert _slices(reports) == {}


# Tenants a (2 ok) and b (1 ok, 1 failed) grouped by ["tenant"]: one report per tenant, each with its own
# request, success and failure counts and a "labels" block naming the slice.
async def test_generate_reports_emits_one_report_per_slice() -> None:
    metrics = [_req({"tenant": "a"}), _req({"tenant": "a"}), _req({"tenant": "b"}), _req({"tenant": "b"}, failed=True)]
    slices = _slices(await _generate(metrics, [["tenant"]]))
    assert set(slices) == {"labels_tenant=a_lifecycle_metrics", "labels_tenant=b_lifecycle_metrics"}

    a = slices["labels_tenant=a_lifecycle_metrics"].contents
    assert a["labels"] == {"tenant": "a"}
    assert a["load_summary"]["count"] == 2
    assert a["successes"]["count"] == 2
    assert a["failures"]["count"] == 0

    b = slices["labels_tenant=b_lifecycle_metrics"].contents
    assert b["labels"] == {"tenant": "b"}
    assert b["load_summary"]["count"] == 2
    assert b["successes"]["count"] == 1
    assert b["failures"]["count"] == 1


# Two groups, ["tenant"] and ["tenant", "priority"], over one request give one report for each group.
async def test_generate_reports_emits_every_group() -> None:
    slices = _slices(await _generate([_req({"tenant": "a", "priority": "high"})], [["tenant"], ["tenant", "priority"]]))
    assert set(slices) == {"labels_tenant=a_lifecycle_metrics", "labels_tenant=a_priority=high_lifecycle_metrics"}


# Collects every entry of every "errors" list at any depth of a validation.json document.
def _validation_errors(node: object) -> List[object]:
    found: List[object] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "errors" and isinstance(value, list):
                found.extend(value)
            else:
                found.extend(_validation_errors(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_validation_errors(item))
    return found


# Adding label slices leaves validation.json with no error findings.
async def test_slices_do_not_fail_report_validation() -> None:
    reports = await _generate([_req({"tenant": "a"}), _req({"tenant": "b"})], [["tenant"]])
    validation = next(r for r in reports if r.name == VALIDATION_REPORT_NAME)
    assert _validation_errors(validation.contents) == []
