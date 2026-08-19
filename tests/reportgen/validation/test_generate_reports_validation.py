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
"""End-to-end assembly test: drive the real ``generate_reports`` over
synthesized request metrics and assert the emitted set validates cleanly.

This is the assembly-level seam #661 asks for: the report set users read is
produced by the real code path, and ``validation.json`` is the assertion
interface over it.
"""

from __future__ import annotations

import asyncio
from typing import List, cast
from unittest.mock import Mock

from inference_perf.client.server_metrics import PerfRuntimeParameters
from inference_perf.client.server_metrics.base import StageRuntimeInfo, StageStatus
from inference_perf.config import Config
from inference_perf.config.reportgen.config import ReportConfig, RequestLifecycleMetricsReportConfig
from inference_perf.metrics.request_collector import RequestMetricCollector
from inference_perf.reportgen import ReportGenerator
from inference_perf.utils import ReportFile

from .helpers import make_stage_metrics


def _generate_reports() -> List[ReportFile]:
    stage_metrics = make_stage_metrics()
    all_metrics = [m for metrics in stage_metrics.values() for m in metrics]

    collector = Mock(spec=RequestMetricCollector)
    collector.get_metrics.return_value = all_metrics

    config = Mock(spec=Config)
    config.tokenizer = None
    config.model_dump.return_value = {"load": {"stages": [{"rate": 1.0}, {"rate": 1.0}]}}

    generator = ReportGenerator(metrics_client=None, metrics_collector=collector, config=cast(Config, config))

    stages = {
        stage_id: StageRuntimeInfo(
            stage_id=stage_id,
            rate=1.0,
            start_time=min(m.start_time for m in metrics),
            end_time=max(m.end_time for m in metrics),
            status=StageStatus.COMPLETED,
        )
        for stage_id, metrics in stage_metrics.items()
    }
    runtime_parameters = PerfRuntimeParameters(start_time=0.0, duration=12.0, model_server_metrics=Mock(), stages=stages)
    report_config = ReportConfig(
        request_lifecycle=RequestLifecycleMetricsReportConfig(per_request=True),
        prometheus=None,
    )
    return asyncio.run(generator.generate_reports(report_config, runtime_parameters))


def test_generate_reports_emits_validation_json_and_it_is_clean() -> None:
    reports = _generate_reports()
    filenames = {r.get_filename() for r in reports}

    assert "validation.json" in filenames
    validation = next(r for r in reports if r.get_filename() == "validation.json").get_contents()

    assert validation["global"]["errors"] == [], validation["global"]["errors"]
    all_errors = [e for group in validation["reports"].values() for e in group["errors"]]
    assert all_errors == [], all_errors
    all_warnings = [w for group in validation["reports"].values() for w in group["warnings"]]
    assert all_warnings == [], all_warnings


def test_generate_reports_validation_covers_the_emitted_files() -> None:
    reports = _generate_reports()
    validation = next(r for r in reports if r.get_filename() == "validation.json").get_contents()

    covered = set(validation["reports"].keys())
    assert "summary_lifecycle_metrics.json" in covered
    assert "stage_0_lifecycle_metrics.json" in covered
    assert "stage_1_lifecycle_metrics.json" in covered
    assert "per_request_lifecycle_metrics.json" in covered
    assert "inference-perf.partial.stage_0.yaml" in covered
    assert "inference-perf.partial.stage_1.yaml" in covered
