from inference_perf.config import read_config, Config, APIType
import os


def test_read_config() -> None:
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yml"))
    config = read_config(["-c", config_path])
    assert isinstance(config, Config)
    assert config.api == APIType.Chat
