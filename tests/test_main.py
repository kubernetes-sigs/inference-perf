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

import sys
from unittest.mock import MagicMock, patch

import pytest

from inference_perf.main import main_cli


def test_main_cli_with_url_flag() -> None:
    """Test that main_cli with --url flag sets up config correctly."""
    # Mock sys.argv
    test_args = ["inference-perf", "--url", "http://dummy-server"]

    # Mock dependencies to avoid real runs and network calls
    with (
        patch.object(sys, "argv", test_args),
        patch("inference_perf.detector.detector.autofill_config") as mock_autofill,
        patch("inference_perf.main.InferencePerfRunner") as mock_runner_class,
        patch("inference_perf.main.vLLMModelServerClient"),
        patch("inference_perf.main.MultiprocessRequestDataCollector"),
        patch("inference_perf.main.ReportGenerator"),
    ):
        from inference_perf.config import Config, LoadConfig, ModelServerClientConfig, ModelServerType, StandardLoadStage

        real_config = Config(
            server=ModelServerClientConfig(type=ModelServerType.VLLM, base_url="http://dummy"),
            load=LoadConfig(stages=[StandardLoadStage(rate=1.0, duration=10)]),
        )
        mock_autofill.return_value = real_config
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        # Run main_cli
        main_cli()

        # Verify autofill_config was called with the URL
        mock_autofill.assert_called_once_with("http://dummy-server", base_config=None)

        # Verify runner was instantiated
        assert mock_runner_class.called

        # Verify runner.run was called
        assert mock_runner.run.called


def test_main_cli_with_url_and_config_file() -> None:
    """Test that main_cli with both -c and --url patches the config."""
    test_args = ["inference-perf", "-c", "dummy_config.yml", "--url", "http://overridden-server"]

    with (
        patch.object(sys, "argv", test_args),
        patch("os.path.exists") as mock_exists,
        patch("inference_perf.main.read_config") as mock_read_config,
        patch("inference_perf.detector.detector.autofill_config") as mock_autofill,
        patch("inference_perf.main.InferencePerfRunner") as mock_runner_class,
        patch("inference_perf.main.vLLMModelServerClient"),
        patch("inference_perf.main.MultiprocessRequestDataCollector"),
        patch("inference_perf.main.ReportGenerator"),
    ):
        mock_exists.return_value = True  # File exists

        from inference_perf.config import Config, LoadConfig, ModelServerClientConfig, ModelServerType, StandardLoadStage

        base_config = Config(
            server=ModelServerClientConfig(type=ModelServerType.VLLM, base_url="http://file-server"),
            load=LoadConfig(stages=[StandardLoadStage(rate=2.0, duration=20)]),
        )
        mock_read_config.return_value = base_config

        patched_config = Config(
            server=ModelServerClientConfig(type=ModelServerType.VLLM, base_url="http://overridden-server"),
            load=LoadConfig(stages=[StandardLoadStage(rate=2.0, duration=20)]),
        )
        mock_autofill.return_value = patched_config

        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        main_cli()

        mock_read_config.assert_called_once_with("dummy_config.yml")
        mock_autofill.assert_called_once_with("http://overridden-server", base_config=base_config)


def test_main_cli_without_args_fails() -> None:
    """Test that main_cli fails if neither -c nor --url is provided."""
    test_args = ["inference-perf"]

    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
        main_cli()
