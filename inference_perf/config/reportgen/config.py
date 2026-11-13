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
from typing import Dict, List, Optional

from inference_perf.config.common import StrictBaseModel
from pydantic import Field, field_validator


class PerRequestFieldsConfig(StrictBaseModel):
    request: bool = Field(default=True, description="Include the raw request payload in per-request report entries.")
    response: bool = Field(default=True, description="Include the raw response payload in per-request report entries.")
    info: bool = Field(default=True, description="Include response metadata in per-request report entries.")
    response_chunks: bool = Field(
        default=True,
        description="Include raw streamed chunks, including text, reasoning and tool arguments, in per-request metadata.",
    )
    computed_metrics: bool = Field(
        default=False, description="Include computed per-request latency and token metrics in report entries."
    )


class RequestLifecycleMetricsReportConfig(StrictBaseModel):
    summary: Optional[bool] = Field(default=True, description="Generate a summary report across the whole run.")
    per_stage: Optional[bool] = Field(default=True, description="Generate a report for each load stage.")
    per_request: Optional[bool] = Field(default=False, description="Generate a report with per-request details.")
    per_request_fields: PerRequestFieldsConfig = Field(
        default_factory=PerRequestFieldsConfig, description="Select raw and computed fields included in per-request entries."
    )
    per_adapter: Optional[bool] = Field(default=True, description="Generate a report for each LoRA adapter.")
    per_adapter_stage: Optional[bool] = Field(
        default=False, description="Generate a report for each LoRA adapter within each load stage."
    )
    percentiles: List[float] = Field(
        default=[0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9], description="Percentiles reported for each metric."
    )
    use_server_output_tokens: bool = Field(
        default=False,
        description="Use the server-reported output token counts in metrics instead of tokenizing the response text.",
    )
    max_error_messages: int = Field(
        default=100,
        description="Cap on the number of distinct example error messages retained per error label in the failure "
        "report, and per substitution entry.",
    )


class PrometheusMetricsReportConfig(StrictBaseModel):
    summary: Optional[bool] = Field(default=True, description="Generate a summary report across the whole run.")
    per_stage: Optional[bool] = Field(default=False, description="Generate a report for each load stage.")


class SessionLifecycleReportConfig(StrictBaseModel):
    summary: Optional[bool] = Field(default=True, description="Generate a summary report across the whole run.")
    per_stage: Optional[bool] = Field(default=True, description="Generate a report for each load stage.")
    per_session: Optional[bool] = Field(default=False, description="Generate a report with per-session details.")


class GoodputConfig(StrictBaseModel):
    constraints: Dict[str, float] = Field(
        default={},
        description="SLO thresholds in seconds that a request must meet to count as good."
        " Keys: 'ttft', 'tpot', 'itl', 'ntpot', 'request_latency'.",
    )


class BenchmarkReportConfig(StrictBaseModel):
    base_file: Optional[str] = Field(
        default=None,
        description="Path to a partial BR0.2 report (YAML) that every inference-perf.partial.stage_<n>.yaml is built on."
        " It may set fields inference-perf does not measure, such as run.description or scenario.stack."
        " Setting a field inference-perf writes is an error at config load.",
    )

    @field_validator("base_file")
    @classmethod
    def _check_base_file(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            # Imported here: the BR package imports reportgen.base, which imports this config package.
            from inference_perf.reportgen.br.v0_2.base_report import load_base_report

            load_base_report(value)
        return value


class ReportConfig(StrictBaseModel):
    request_lifecycle: RequestLifecycleMetricsReportConfig = Field(
        default=RequestLifecycleMetricsReportConfig(),
        description="Reports on client-observed request metrics (latency, throughput, tokens).",
    )
    prometheus: Optional[PrometheusMetricsReportConfig] = Field(
        default=PrometheusMetricsReportConfig(), description="Reports on server-side metrics collected via Prometheus."
    )
    session_lifecycle: SessionLifecycleReportConfig = Field(
        default=SessionLifecycleReportConfig(), description="Reports on session metrics for multi-turn benchmarks."
    )
    goodput: Optional[GoodputConfig] = Field(
        default=None, description="Goodput reporting: the share of requests meeting the configured SLOs."
    )
    benchmark_report: BenchmarkReportConfig = Field(
        default=BenchmarkReportConfig(),
        description="The BR0.2 partial report written per stage (inference-perf.partial.stage_<n>.yaml).",
    )
