from inference_perf.config import (
    APIType,
    Config,
    DataGenType,
    LoadType,
    MetricsClientType,
    deep_merge,
    read_config,
)
import os
import tempfile
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
            "local_storage": {
                "path": "reports-{timestamp}"
            }
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
