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
import os

from inference_perf.config.common import StrictBaseModel
from pydantic import Field


def default_metrics_host() -> str:
    # kubelet sets this in every container. Outside hostNetwork a pod has only lo
    # and eth0, so binding every interface there exposes just the pod IP a scraper dials.
    return "0.0.0.0" if "KUBERNETES_SERVICE_HOST" in os.environ else "127.0.0.1"


class RuntimeMetricsConfig(StrictBaseModel):
    """Exposition of inference-perf's own runtime metrics (not the model server's)."""

    enabled: bool = Field(
        default=True,
        description=(
            "Serve inference-perf's own runtime metrics (stage state, request counts, latencies) over an "
            "HTTP /metrics endpoint for Prometheus to scrape. Metrics are always collected in-process; this "
            "only controls whether the HTTP endpoint is started. On by default, so a run is observable "
            "without being configured for it; set false to keep the port free."
        ),
    )
    host: str = Field(
        default_factory=default_metrics_host,
        description=(
            "Address the runtime metrics endpoint binds to. Defaults to 0.0.0.0 inside a Kubernetes pod, where a "
            "scraper reaches the run by pod IP, and to 127.0.0.1 anywhere else, so a laptop run is not reachable "
            "from the network. Set 0.0.0.0 to scrape a container published with docker run -p."
        ),
    )
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
        default_factory=RuntimeMetricsConfig,
        description="Runtime metrics inference-perf exports about the benchmark run itself.",
    )
