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
"""Google Managed Prometheus client: endpoint construction and auth headers.

Runs without credentials: google.auth is patched, so these tests pin the
monitoring.googleapis.com endpoint shape and the Bearer-token handling
(refresh on every header build, failure when no token comes back).
"""

from unittest.mock import MagicMock, patch

import pytest

from inference_perf.client.server_metrics.prometheus_client.google_managed_prometheus_client import (
    GoogleManagedPrometheusMetricsClient,
)
from inference_perf.config import PrometheusClientConfig

GMP_MODULE = "inference_perf.client.server_metrics.prometheus_client.google_managed_prometheus_client"


def make_client(credentials: MagicMock) -> GoogleManagedPrometheusMetricsClient:
    with patch(f"{GMP_MODULE}.google.auth.default", return_value=(credentials, "proj-1")):
        return GoogleManagedPrometheusMetricsClient(PrometheusClientConfig(google_managed=True, scrape_interval=15))


def test_init_builds_monitoring_endpoint_from_project() -> None:
    client = make_client(MagicMock())
    assert client.query_url == "https://monitoring.googleapis.com/v1/projects/proj-1/location/global/prometheus/api/v1/query"
    assert client.project_id == "proj-1"
    assert client.scrape_interval == 15


def test_get_headers_refreshes_credentials_and_returns_bearer_token() -> None:
    credentials = MagicMock()
    credentials.token = "tok-123"
    client = make_client(credentials)

    with patch(f"{GMP_MODULE}.google.auth.transport.requests.Request") as mock_request_cls:
        headers = client.get_headers()

    assert headers == {"Authorization": "Bearer tok-123"}
    credentials.refresh.assert_called_once_with(mock_request_cls.return_value)


def test_get_headers_fails_without_token() -> None:
    credentials = MagicMock()
    credentials.token = None
    client = make_client(credentials)

    with patch(f"{GMP_MODULE}.google.auth.transport.requests.Request"):
        with pytest.raises(Exception, match="Failed to get credentials token"):
            client.get_headers()
