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


class PerRequestFieldsConfig(BaseModel):
    request: bool = True
    response: bool = True
    info: bool = True
    response_chunks: bool = True


class RequestLifecycleMetricsReportConfig(BaseModel):
    summary: Optional[bool] = True
    per_stage: Optional[bool] = True
    per_request: Optional[bool] = False
    per_request_fields: PerRequestFieldsConfig = Field(default_factory=PerRequestFieldsConfig)
    per_adapter: Optional[bool] = True
    per_adapter_stage: Optional[bool] = False
    percentiles: List[float] = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    # Normalize the per-output-token latency metrics (TPOT, NTPOT) by the server's
    # reported usage.completion_tokens instead of the client-side re-tokenized count.
    use_server_output_tokens: bool = False


class PrometheusMetricsReportConfig(BaseModel):
    summary: Optional[bool] = True
    per_stage: Optional[bool] = False


class SessionLifecycleReportConfig(BaseModel):
    summary: Optional[bool] = True
    per_stage: Optional[bool] = True
    per_session: Optional[bool] = False


class GoodputConfig(BaseModel):
    constraints: Dict[str, float] = {}


class ReportConfig(BaseModel):
    request_lifecycle: RequestLifecycleMetricsReportConfig = RequestLifecycleMetricsReportConfig()
    prometheus: Optional[PrometheusMetricsReportConfig] = PrometheusMetricsReportConfig()
    session_lifecycle: SessionLifecycleReportConfig = SessionLifecycleReportConfig()
    goodput: Optional[GoodputConfig] = None
