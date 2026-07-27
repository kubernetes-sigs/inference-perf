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
"""Top-level Config assembly: read_config, deep_merge, and cross-field validators.

Per-surface schema rules live in the sibling modules that mirror the
``inference_perf.config`` package (``loadgen/``, ``datagen/``, ``client/``,
etc.). This file covers only the glue: loading YAML into a ``Config``,
merging overrides, timestamp substitution, and the whole-config validators
defined on ``Config`` itself.
"""

import os
import tempfile

import pytest
import yaml

from inference_perf.config import (
    APIType,
    Config,
    DataGenType,
    LoadType,
    MetricsClientType,
    deep_merge,
    read_config,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_read_config() -> None:
    config = read_config(os.path.join(REPO_ROOT, "config.yml"))

    assert isinstance(config, Config)
    assert config.api.type == APIType.Completion
    assert config.data.type == DataGenType.ShareGPT
    assert config.load.type == LoadType.CONSTANT
    if config.metrics:
        assert config.metrics.type == MetricsClientType.PROMETHEUS
    assert config.report.request_lifecycle.summary is True


def test_read_config_empty_yaml_uses_defaults() -> None:
    """An empty config file is valid; the resulting Config is all defaults."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write("")
        tmp_path = tmp.name
    try:
        config = read_config(tmp_path)
        assert config == Config()
    finally:
        os.remove(tmp_path)


def test_read_config_no_file_returns_defaults() -> None:
    assert read_config() == Config()


def test_deep_merge() -> None:
    base = {
        "api": APIType.Chat,
        "data": {"type": DataGenType.ShareGPT},
        "load": {"type": LoadType.CONSTANT},
        "metrics": {"type": MetricsClientType.PROMETHEUS},
    }
    override = {
        "data": {"type": DataGenType.Mock},
        "load": {"type": LoadType.POISSON},
    }
    merged = deep_merge(base, override)

    assert merged["api"] == APIType.Chat
    assert merged["data"]["type"] == DataGenType.Mock
    assert merged["load"]["type"] == LoadType.POISSON
    assert merged["metrics"]["type"] == MetricsClientType.PROMETHEUS


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"data": {"type": "mock", "path": "keep"}}
    override = {"data": {"type": "synthetic"}}

    merged = deep_merge(base, override)

    assert merged["data"] == {"type": "synthetic", "path": "keep"}
    # Originals untouched.
    assert base == {"data": {"type": "mock", "path": "keep"}}
    assert override == {"data": {"type": "synthetic"}}


def test_read_config_cli_overrides_win() -> None:
    config_content = {"data": {"type": "shareGPT"}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config_content, tmp)
        tmp_path = tmp.name
    try:
        config = read_config(tmp_path, cli_overrides={"data": {"type": "mock"}})
        assert config.data.type == DataGenType.Mock
    finally:
        os.remove(tmp_path)


def test_read_config_timestamp_substitution() -> None:
    # Create a minimalistic config with {timestamp} in the storage path
    config_content = {
        "storage": {
            "local_storage": {"path": "reports-{timestamp}"},
            "google_cloud_storage": {"bucket_name": "my-bucket", "path": "gcs-reports-{timestamp}"},
            "simple_storage_service": {"bucket_name": "my-bucket", "path": "s3-reports-{timestamp}"},
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config_content, tmp)
        tmp_path = tmp.name

    try:
        config = read_config(tmp_path)
        # Verify substitution happened
        assert config.storage is not None
        assert "{timestamp}" not in config.storage.local_storage.path
        assert config.storage.local_storage.path.startswith("reports-")

        assert config.storage.google_cloud_storage is not None
        assert "{timestamp}" not in config.storage.google_cloud_storage.path
        assert config.storage.google_cloud_storage.path.startswith("gcs-reports-")

        assert config.storage.simple_storage_service is not None
        assert "{timestamp}" not in config.storage.simple_storage_service.path
        assert config.storage.simple_storage_service.path.startswith("s3-reports-")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_otel_trace_replay_requires_trace_session_replay_load() -> None:
    """Config-level validator: otel_trace_replay data demands session-replay load."""
    with pytest.raises(ValueError, match="requires load.type 'trace_session_replay'"):
        Config.model_validate(
            {
                "data": {"type": "otel_trace_replay"},
                "load": {"type": "constant"},
            }
        )


def test_otel_trace_replay_with_session_replay_load_ok() -> None:
    config = Config.model_validate(
        {
            "data": {"type": "otel_trace_replay"},
            "load": {"type": "trace_session_replay"},
        }
    )
    assert config.data.type == DataGenType.OTelTraceReplay
    assert config.load.type == LoadType.TRACE_SESSION_REPLAY
