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
"""Validity rules for ``inference_perf.config.client.server_metrics``."""

import pytest

from inference_perf.config import PrometheusClientConfig


def test_url_only_is_valid() -> None:
    config = PrometheusClientConfig(url="http://localhost:9090")
    # Pydantic's HttpUrl normalizes by appending a trailing slash.
    assert str(config.url) == "http://localhost:9090/"


def test_google_managed_only_is_valid() -> None:
    config = PrometheusClientConfig(google_managed=True)
    assert config.google_managed is True


def test_both_url_and_google_managed_is_error() -> None:
    with pytest.raises(ValueError, match="Exactly one of 'url' or 'google_managed' must be set"):
        PrometheusClientConfig(url="http://localhost:9090", google_managed=True)


def test_neither_url_nor_google_managed_is_error() -> None:
    with pytest.raises(ValueError, match="Exactly one of 'url' or 'google_managed' must be set"):
        PrometheusClientConfig(google_managed=False)


def test_auth_defaults_preserve_existing_behavior() -> None:
    """Regression: no token, TLS verified, no extra headers unless configured."""
    config = PrometheusClientConfig(url="http://localhost:9090")
    assert config.bearer_token is None
    assert config.verify_ssl is True
    assert config.headers is None


def test_bearer_token_accepted() -> None:
    config = PrometheusClientConfig(url="http://localhost:9090", bearer_token="prom-token")
    assert config.bearer_token is not None
    assert config.bearer_token.get_secret_value() == "prom-token"


def test_verify_ssl_can_be_disabled() -> None:
    config = PrometheusClientConfig(url="http://localhost:9090", verify_ssl=False)
    assert config.verify_ssl is False


def test_custom_headers_accepted() -> None:
    config = PrometheusClientConfig(url="http://localhost:9090", headers={"X-Scope-OrgID": "team-a"})
    assert config.headers == {"X-Scope-OrgID": "team-a"}
