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

from pydantic import BaseModel, Field


class RequestLifecycleMetricsReportConfig(BaseModel):
    summary: Optional[bool] = Field(default=True, description="Generate a summary report across the whole run.")
    per_stage: Optional[bool] = Field(default=True, description="Generate a report for each load stage.")
    per_request: Optional[bool] = Field(default=False, description="Generate a report with per-request details.")
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


class PrometheusMetricsReportConfig(BaseModel):
    summary: Optional[bool] = Field(default=True, description="Generate a summary report across the whole run.")
    per_stage: Optional[bool] = Field(default=False, description="Generate a report for each load stage.")


class SessionLifecycleReportConfig(BaseModel):
    summary: Optional[bool] = Field(default=True, description="Generate a summary report across the whole run.")
    per_stage: Optional[bool] = Field(default=True, description="Generate a report for each load stage.")
    per_session: Optional[bool] = Field(default=False, description="Generate a report with per-session details.")


class GoodputConfig(BaseModel):
    constraints: Dict[str, float] = Field(
        default={},
        description="SLO thresholds in seconds that a request must meet to count as good."
        " Keys: 'ttft', 'tpot', 'itl', 'ntpot', 'request_latency'.",
    )


class ReportConfig(BaseModel):
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
