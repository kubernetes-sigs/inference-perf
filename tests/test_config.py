from inference_perf.config import read_config, deep_merge, Config, APIType, DataGenType, LoadType, MetricsClientType
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
