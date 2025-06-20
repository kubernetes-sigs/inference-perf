# Copyright 2025 The Kubernetes Authors.
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
import logging
from typing import Any
from pydantic import HttpUrl
from inference_perf.client.metricsclient.prometheus_client.base import PrometheusMetricsClient
from inference_perf.config import PrometheusClientConfig
import google.auth

logger = logging.getLogger(__name__)


class GoogleManagedPrometheusMetricsClient(PrometheusMetricsClient):
    def __init__(self, config: PrometheusClientConfig) -> None:
        if config.google_managed is None:
            raise Exception("google managed prometheus config missing")
        self.google_managed_prometheus_config = config.google_managed
        # Creates a credentials object from the default service account file
        # Assumes that script has appropriate default credentials set up, ref:
        # https://googleapis.dev/python/google-auth/latest/user-guide.html#application-default-credentials
        credentials, project_id = google.auth.default()  # type: ignore[no-untyped-call]
        self.credentials = credentials
        self.project_id = project_id
        config.url = HttpUrl(f"https://monitoring.googleapis.com/v1/projects/{self.project_id}/location/global/prometheus")
        super().__init__(config)

    def get_headers(self) -> dict[str, Any]:
        # Prepare an authentication request - helps format the request auth token
        auth_req = google.auth.transport.requests.Request()

        self.credentials.refresh(auth_req)
        return {"Authorization": "Bearer " + self.credentials.token}
