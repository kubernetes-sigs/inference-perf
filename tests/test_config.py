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
    Distribution,
    DistributionType,
    LoadType,
    MetricsClientType,
    deep_merge,
    read_config,
    ResponseFormat,
    ResponseFormatType,
    StandardLoadStage,
    ConcurrentLoadStage,
    LoadConfig,
    PrometheusClientConfig,
    MultiLoRAConfig,
)
import os
import tempfile

import pytest
import yaml


def test_read_config() -> None:
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yml"))
    config = read_config(config_path)

    assert isinstance(config, Config)
    assert config.api.type == APIType.Completion
    assert config.data.type == DataGenType.ShareGPT
    assert config.load.type == LoadType.CONSTANT
    if config.metrics:
        assert config.metrics.type == MetricsClientType.PROMETHEUS
    assert config.report.request_lifecycle.summary is True


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
    assert merged["metrics"]["type"] == MetricsClientType.PROMETHEUS


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


def test_shared_prefix_aliases() -> None:
    # Test using the short names (field names)
    config_short = Config.model_validate(
        {
            "data": {
                "type": DataGenType.SharedPrefix,
                "shared_prefix": {"num_groups": 5, "num_prompts_per_group": 20},
            }
        }
    )
    assert config_short.data.shared_prefix is not None
    assert config_short.data.shared_prefix.num_groups == 5
    assert config_short.data.shared_prefix.num_prompts_per_group == 20

    # Test using the long names (aliases)
    config_long = Config.model_validate(
        {
            "data": {
                "type": DataGenType.SharedPrefix,
                "shared_prefix": {"num_unique_system_prompts": 7, "num_users_per_system_prompt": 15},
            }
        }
    )
    assert config_long.data.shared_prefix is not None
    assert config_long.data.shared_prefix.num_groups == 7
    assert config_long.data.shared_prefix.num_prompts_per_group == 15

    # Test serialization
    dumped = config_long.model_dump(mode="json", by_alias=True)
    shared_prefix_dump = dumped["data"]["shared_prefix"]
    assert shared_prefix_dump["num_unique_system_prompts"] == 7
    assert shared_prefix_dump["num_users_per_system_prompt"] == 15


def test_response_format_to_api_format() -> None:
    # Test JSON_SCHEMA (default)
    fmt = ResponseFormat(json_schema={"type": "object"})
    assert fmt.to_api_format() == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": {"type": "object"},
        },
    }

    # Test JSON_OBJECT
    fmt2 = ResponseFormat(type=ResponseFormatType.JSON_OBJECT)
    assert fmt2.to_api_format() == {"type": "json_object"}


def test_standard_load_stage_validation() -> None:
    import pytest

    # valid
    StandardLoadStage(rate=10, duration=60)

    # invalid num_requests
    with pytest.raises(ValueError, match="num_requests should not be set"):
        StandardLoadStage(rate=10, duration=60, num_requests=100)

    # invalid concurrency_level
    with pytest.raises(ValueError, match="concurrency_level should not be set"):
        StandardLoadStage(rate=10, duration=60, concurrency_level=5)


def test_concurrent_load_stage() -> None:
    # Just verify we can create it and it hits the validator returning self
    stage = ConcurrentLoadStage(num_requests=100, concurrency_level=10)
    assert stage.num_requests == 100
    assert stage.concurrency_level == 10


def test_load_config_validation() -> None:
    import pytest
    from inference_perf.config import SweepConfig, StageGenType

    # Sweep with CONCURRENT
    with pytest.raises(ValueError, match="Cannot have sweep config with CONCURRENT"):
        LoadConfig(
            type=LoadType.CONCURRENT,
            sweep=SweepConfig(type=StageGenType.GEOM),
            stages=[ConcurrentLoadStage(num_requests=10, concurrency_level=1)],
        )

    # CONCURRENT with non-ConcurrentLoadStage
    with pytest.raises(ValueError, match="CONCURRENT load type requires ConcurrentLoadStage"):
        LoadConfig(
            type=LoadType.CONCURRENT,
            stages=[StandardLoadStage(rate=10, duration=60)],
        )

    # CONSTANT with non-StandardLoadStage
    with pytest.raises(ValueError, match="CONSTANT load type requires StandardLoadStage"):
        LoadConfig(
            type=LoadType.CONSTANT,
            stages=[ConcurrentLoadStage(num_requests=10, concurrency_level=1)],
        )

    # MultiLoRA traffic split not adding up to 1.0
    with pytest.raises(ValueError, match="MultiLoRA traffic split.*does not add up to 1.0"):
        LoadConfig(
            lora_traffic_split=[
                MultiLoRAConfig(name="a", split=0.5),
                MultiLoRAConfig(name="b", split=0.4),
            ]
        )


def test_prometheus_client_config_validation() -> None:
    import pytest

    # Both set
    with pytest.raises(ValueError, match="Exactly one of 'url' or 'google_managed' must be set"):
        PrometheusClientConfig(url="http://localhost:9090", google_managed=True)

    # Neither set
    with pytest.raises(ValueError, match="Exactly one of 'url' or 'google_managed' must be set"):
        PrometheusClientConfig(google_managed=False)

def test_shared_prefix_inline_distribution() -> None:
    config = Config.model_validate(
        {
            "data": {
                "type": DataGenType.SharedPrefix,
                "shared_prefix": {
                    "num_groups": 5,
                    "question_len": {
                        "type": "skew_normal",
                        "mean": 200,
                        "std_dev": 80,
                        "skew": 2.5,
                        "min": 10,
                        "max": 2000,
                    },
                },
            }
        }
    )
    sp = config.data.shared_prefix
    assert sp is not None
    assert isinstance(sp.question_len, Distribution)
    assert sp.question_len.type == DistributionType.SKEW_NORMAL
    assert sp.question_len.mean == 200
    assert sp.question_len.skew == 2.5


def test_shared_prefix_fixed_int_unchanged() -> None:
    config = Config.model_validate(
        {
            "data": {
                "type": DataGenType.SharedPrefix,
                "shared_prefix": {"question_len": 75},
            }
        }
    )
    sp = config.data.shared_prefix
    assert sp is not None
    assert sp.question_len == 75
    assert isinstance(sp.question_len, int)


def test_shared_prefix_ambiguous_distribution_error() -> None:
    with pytest.raises(Exception, match="Cannot specify both"):
        Config.model_validate(
            {
                "data": {
                    "type": DataGenType.SharedPrefix,
                    "shared_prefix": {
                        "question_len": {
                            "type": "normal",
                            "mean": 200,
                            "min": 10,
                            "max": 2000,
                            "std_dev": 50,
                        },
                        "question_distribution": {
                            "min": 10,
                            "max": 1024,
                            "mean": 512,
                            "std_dev": 200,
                        },
                    },
                }
            }
        )


def test_shared_prefix_legacy_distribution_compat() -> None:
    config = Config.model_validate(
        {
            "data": {
                "type": DataGenType.SharedPrefix,
                "shared_prefix": {
                    "question_len": 50,
                    "question_distribution": {
                        "min": 10,
                        "max": 1024,
                        "mean": 512,
                        "std_dev": 200,
                    },
                },
            }
        }
    )
    sp = config.data.shared_prefix
    assert sp is not None
    assert sp.question_len == 50
    assert sp.question_distribution is not None
    assert sp.question_distribution.mean == 512


def test_shared_prefix_seed_field() -> None:
    config = Config.model_validate(
        {
            "data": {
                "type": DataGenType.SharedPrefix,
                "shared_prefix": {"seed": 42},
            }
        }
    )
    assert config.data.shared_prefix is not None
    assert config.data.shared_prefix.seed == 42


def test_distribution_variance_conversion() -> None:
    d = Distribution(type=DistributionType.NORMAL, mean=100.0, variance=6400.0, std_dev=0.0)
    assert abs(d.std_dev - 80.0) < 1e-6


def test_distribution_both_variance_and_std_dev_error() -> None:
    with pytest.raises(Exception, match="Specify either"):
        Distribution(type=DistributionType.NORMAL, mean=100.0, std_dev=10.0, variance=100.0)
