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
from typing import Optional

from inference_perf.config.common import StrictBaseModel
from pydantic import Field


class RuntimeMetricsConfig(StrictBaseModel):
    """Exposition of inference-perf's own runtime metrics (not the model server's)."""

    port: int = Field(
        # Keep in sync with inference_perf.observability.metrics.prometheus.DEFAULT_PORT
        # (not imported here to avoid a config <-> observability import cycle; a test pins them).
        default=9464,
        ge=0,
        le=65535,
        description=(
            "Port the runtime metrics endpoint listens on. Setting it turns the endpoint on. "
            "0 picks an ephemeral port, logged at startup."
        ),
    )


class ObservabilityConfig(StrictBaseModel):
    metrics: Optional[RuntimeMetricsConfig] = Field(
        default=None,
        description=(
            "Serve inference-perf's own runtime metrics (stage state, request counts, latencies) over HTTP "
            "/metrics on every interface, for Prometheus to scrape by pod or host IP. Off unless set. Metrics "
            "are always collected in-process for the progress bars. This only starts the endpoint."
        ),
    )
