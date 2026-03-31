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
from inference_perf.config import (
    APIType,
    Config,
    DataGenType,
    LoadType,
    MetricsClientType,
    deep_merge,
    read_config,
    sanitize_config,
)
import os
import tempfile
import yaml
from pathlib import Path
from typing import Any


def test_read_config() -> None:
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yml"))
    config = read_config(config_path)

    assert isinstance(config, Config)
    assert config.api.type == APIType.COMPLETION
    assert config.data.type == DataGenType.SHAREGPT
    assert config.load.type == LoadType.CONSTANT
    if config.metrics:
        assert config.metrics.type == MetricsClientType.PROMETHEUS
    assert config.report.request_lifecycle.summary is True


def test_deep_merge() -> None:
    base = {
        "api": APIType.CHAT,
        "data": {"type": DataGenType.SHAREGPT},
        "load": {"type": LoadType.CONSTANT},
        "metrics": {"type": MetricsClientType.PROMETHEUS},
    }
    override = {
        "data": {"type": DataGenType.MOCK},
        "load": {"type": LoadType.POISSON},
    }
    merged = deep_merge(base, override)

    assert merged["api"] == APIType.CHAT
    assert merged["data"]["type"] == DataGenType.MOCK
    assert merged["load"]["type"] == LoadType.POISSON
    assert merged["metrics"]["type"] == MetricsClientType.PROMETHEUS
    assert merged["metrics"]["type"] == MetricsClientType.PROMETHEUS


def test_read_config_timestamp_substitution() -> None:
    # Create a minimalistic config with {timestamp} in the storage path
    config_content = {"storage": {"local_storage": {"path": "reports-{timestamp}"}}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config_content, tmp)
        tmp_path = tmp.name

    try:
        config = read_config(tmp_path)
        # Verify substitution happened
        assert config.storage is not None
        assert "{timestamp}" not in config.storage.local_storage.path
        assert config.storage.local_storage.path.startswith("reports-")
        # Basic check for timestamp format (YYYYMMDD...) which implies it's roughly length 8+
        assert len(config.storage.local_storage.path) > len("reports-")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_shared_prefix_aliases() -> None:
    # Test using the short names (field names)
    config_short = Config.model_validate(
        {
            "data": {
                "type": DataGenType.SHARED_PREFIX,
                "shared_prefix": {"num_groups": 5, "num_prompts_per_group": 20},
            }
        }
    )
    assert config_short.data.shared_prefix is not None
    assert config_short.data.shared_prefix.num_groups == 5
    assert config_short.data.shared_prefix.num_prompts_per_group == 20

    # Test using the long names (aliases)
    raw_cfg = {
        "data": {
            "type": DataGenType.SHARED_PREFIX,
            "shared_prefix": {"num_unique_system_prompts": 7, "num_users_per_system_prompt": 15},
        }
    }
    config_long = Config.model_validate(sanitize_config(raw_cfg))
    assert config_long.data.shared_prefix is not None
    assert config_long.data.shared_prefix.num_groups == 7
    assert config_long.data.shared_prefix.num_prompts_per_group == 15


def test_sanitize_config() -> None:
    # Test DataGenType.SHAREGPT
    cfg: dict[str, Any] = {"data": {"type": "shareGPT"}}
    sanitized = sanitize_config(cfg)
    assert sanitized["data"]["type"] == 2

    # Test TraceFormat.AZURE_PUBLIC_DATASET in data.trace
    cfg = {"data": {"trace": {"format": "AzurePublicDataset"}}}
    sanitized = sanitize_config(cfg)
    assert sanitized["data"]["trace"]["format"] == 1

    # Test TraceFormat.AZURE_PUBLIC_DATASET in load.trace
    cfg = {"load": {"trace": {"format": "AzurePublicDataset"}}}
    sanitized = sanitize_config(cfg)
    assert sanitized["load"]["trace"]["format"] == 1

    # Test ModelServerType.MOCK_SERVER
    cfg = {"server": {"type": "mock"}}
    sanitized = sanitize_config(cfg)
    assert sanitized["server"]["type"] == 4


def test_sanitize_config_presubmit() -> None:
    import inspect

    # Get the source code of sanitize_config
    source = inspect.getsource(sanitize_config)

    # Count non-empty lines
    lines = [line.strip() for line in source.split("\n") if line.strip()]
    assert len(lines) <= 40, (
        f"sanitize_config has grown too large ({len(lines)} lines). Please do not add non-generated APIs here."
    )


def test_config_validation() -> None:
    from pydantic import ValidationError
    from inference_perf.config import SweepConfig

    # Test SweepConfig validation
    try:
        SweepConfig(num_requests=0)
        raise AssertionError("SweepConfig should have failed with num_requests=0")
    except ValidationError as e:
        assert "num_requests" in str(e)

    try:
        SweepConfig(saturation_percentile=-1.0)
        raise AssertionError("SweepConfig should have failed with saturation_percentile=-1.0")
    except ValidationError as e:
        assert "saturation_percentile" in str(e)

    try:
        SweepConfig(saturation_percentile=101.0)
        raise AssertionError("SweepConfig should have failed with saturation_percentile=101.0")
    except ValidationError as e:
        assert "saturation_percentile" in str(e)

    # Valid config should pass
    sc = SweepConfig(num_requests=1, timeout=1.0, num_stages=1, stage_duration=1, saturation_percentile=50.0)
    assert sc.num_requests == 1


def test_cel_validation(tmp_path: Path) -> None:
    from inference_perf.config import read_config
    import yaml

    # Create a temp config file
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "invalid_config.yml"

    # Config with INFINITY_INSTRUCT but no path
    cfg_data = {"api": {"type": "completion"}, "data": {"type": "infinity_instruct"}, "load": {"type": "constant"}}

    p.write_text(yaml.dump(cfg_data))

    try:
        read_config(str(p))
        raise AssertionError("Should have failed with CEL validation error")
    except Exception as e:
        assert "invalid Config" in str(e)
