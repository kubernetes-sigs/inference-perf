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
"""Validity rules for ``inference_perf.config.client.filestorage``."""

import pytest
from pydantic import ValidationError

from inference_perf.config import SimpleStorageServiceConfig


def test_defaults() -> None:
    config = SimpleStorageServiceConfig(bucket_name="my-bucket")
    assert config.bucket_name == "my-bucket"
    assert config.endpoint_url is None
    assert config.region_name is None
    assert config.addressing_style is None


def test_accepts_addressing_overrides() -> None:
    config = SimpleStorageServiceConfig(
        bucket_name="my-bucket",
        endpoint_url="https://s3.example.com",
        region_name="us-east-1",
        addressing_style="virtual",
    )
    assert config.endpoint_url == "https://s3.example.com"
    assert config.region_name == "us-east-1"
    assert config.addressing_style == "virtual"


def test_accepts_path_addressing_style() -> None:
    config = SimpleStorageServiceConfig(bucket_name="my-bucket", addressing_style="path")
    assert config.addressing_style == "path"


def test_rejects_invalid_addressing_style() -> None:
    with pytest.raises(ValidationError):
        SimpleStorageServiceConfig(bucket_name="my-bucket", addressing_style="hosted")
