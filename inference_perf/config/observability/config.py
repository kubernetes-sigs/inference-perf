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
from inference_perf.config.common import StrictBaseModel
from pydantic import Field


class RuntimeMetricsConfig(StrictBaseModel):
    """Exposition of inference-perf's own runtime metrics (not the model server's)."""

    enabled: bool = Field(
        default=False,
        description=(
            "Serve inference-perf's own runtime metrics (stage state, request counts, latencies) over an "
            "HTTP /metrics endpoint for Prometheus to scrape. Metrics are always collected in-process; this "
            "only controls whether the HTTP endpoint is started."
        ),
    )
    host: str = Field(default="0.0.0.0", description="Address the runtime metrics endpoint binds to.")
    port: int = Field(
        # Keep in sync with inference_perf.observability.metrics.prometheus.DEFAULT_PORT
        # (not imported here to avoid a config <-> observability import cycle; a test pins them).
        default=9464,
        ge=0,
        le=65535,
        description="Port the runtime metrics endpoint listens on. 0 picks an ephemeral port, logged at startup.",
    )


class ObservabilityConfig(StrictBaseModel):
    metrics: RuntimeMetricsConfig = Field(
        default=RuntimeMetricsConfig(),
        description="Runtime metrics inference-perf exports about the benchmark run itself.",
    )
