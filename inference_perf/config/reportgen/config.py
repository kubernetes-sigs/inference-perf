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


class RequestLifecycleMetricsReportConfig(StrictBaseModel):
    summary: Optional[bool] = True
    per_stage: Optional[bool] = True
    per_request: Optional[bool] = False
    per_adapter: Optional[bool] = True
    per_adapter_stage: Optional[bool] = False
    percentiles: List[float] = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    use_server_output_tokens: bool = False
    # Cap on the number of per-error example messages retained per error label
    # (in failures.by_label) and per substitution entry
    max_error_messages: int = 100


class PrometheusMetricsReportConfig(StrictBaseModel):
    summary: Optional[bool] = True
    per_stage: Optional[bool] = False


class SessionLifecycleReportConfig(StrictBaseModel):
    summary: Optional[bool] = True
    per_stage: Optional[bool] = True
    per_session: Optional[bool] = False


class GoodputConfig(StrictBaseModel):
    constraints: Dict[str, float] = {}


class ReportConfig(StrictBaseModel):
    request_lifecycle: RequestLifecycleMetricsReportConfig = RequestLifecycleMetricsReportConfig()
    prometheus: Optional[PrometheusMetricsReportConfig] = PrometheusMetricsReportConfig()
    session_lifecycle: SessionLifecycleReportConfig = SessionLifecycleReportConfig()
    goodput: Optional[GoodputConfig] = None
