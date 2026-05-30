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

from unittest.mock import MagicMock, patch

from inference_perf.config import ModelServerType
from inference_perf.detector.detector import EnvironmentType, detect_server_type


def test_detect_vllm_via_metrics() -> None:
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "vllm:num_requests_waiting 0"
        mock_get.return_value = mock_response

        assert detect_server_type("http://dummy") == ModelServerType.VLLM


def test_detect_sglang_via_metrics() -> None:
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "sglang:num_queue_reqs 0"
        mock_get.return_value = mock_response

        assert detect_server_type("http://dummy") == ModelServerType.SGLANG


def test_detect_tgi_via_metrics() -> None:
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "tgi_queue_size 0"
        mock_get.return_value = mock_response

        assert detect_server_type("http://dummy") == ModelServerType.TGI


def test_detect_vllm_via_v1_models() -> None:
    with patch("requests.get") as mock_get:
        # /metrics returns empty text
        mock_metrics = MagicMock()
        mock_metrics.status_code = 200
        mock_metrics.text = ""

        # /v1/models returns 200
        mock_models = MagicMock()
        mock_models.status_code = 200

        mock_get.side_effect = [mock_metrics, mock_models]

        assert detect_server_type("http://dummy") == ModelServerType.VLLM


def test_detect_tgi_via_generate() -> None:
    with patch("requests.get") as mock_get:
        # /metrics fails
        mock_metrics = MagicMock()
        mock_metrics.status_code = 404

        # /v1/models fails
        mock_models = MagicMock()
        mock_models.status_code = 404

        # /generate returns 405 Method Not Allowed
        mock_generate = MagicMock()
        mock_generate.status_code = 405

        mock_get.side_effect = [mock_metrics, mock_models, mock_generate]

        assert detect_server_type("http://dummy") == ModelServerType.TGI


def test_detect_fallback_to_none() -> None:
    with patch("requests.get") as mock_get:
        # All probes raise exception
        mock_get.side_effect = Exception("Connection error")

        assert detect_server_type("http://dummy") is None


def test_detect_environment_gke() -> None:
    with patch("requests.get") as mock_get, patch.dict("os.environ", {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        from inference_perf.detector.detector import detect_environment

        assert detect_environment() == EnvironmentType.GKE


def test_detect_environment_eks() -> None:
    with patch("requests.get") as mock_get, patch.dict("os.environ", {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}):
        # GCP fails
        mock_gcp = MagicMock()
        mock_gcp.status_code = 404
        # AWS succeeds
        mock_aws = MagicMock()
        mock_aws.status_code = 200

        mock_get.side_effect = [mock_gcp, mock_aws]

        from inference_perf.detector.detector import detect_environment

        assert detect_environment() == EnvironmentType.EKS


def test_detect_environment_local() -> None:
    # Use empty dict to ensure KUBERNETES_SERVICE_HOST is not present
    with patch("requests.get") as mock_get, patch.dict("os.environ", {}, clear=True):
        mock_get.side_effect = Exception("Connection error")

        from inference_perf.detector.detector import detect_environment

        assert detect_environment() == EnvironmentType.LOCAL


def test_autofill_metrics_config_gke() -> None:
    from inference_perf.config import MetricsClientType
    from inference_perf.detector.detector import autofill_metrics_config

    cfg = autofill_metrics_config(EnvironmentType.GKE)
    assert cfg is not None
    assert cfg.type == MetricsClientType.PROMETHEUS
    assert cfg.prometheus is not None
    assert cfg.prometheus.google_managed is True


def test_autofill_metrics_config_local() -> None:
    from inference_perf.config import MetricsClientType
    from inference_perf.detector.detector import autofill_metrics_config

    cfg = autofill_metrics_config(EnvironmentType.LOCAL)
    assert cfg is not None
    assert cfg.type == MetricsClientType.PROMETHEUS
    assert cfg.prometheus is not None
    assert cfg.prometheus.google_managed is False
    assert str(cfg.prometheus.url) == "http://localhost:9090/"


def test_autofill_config_with_base_config() -> None:
    from inference_perf.config import Config, LoadConfig, ModelServerClientConfig, ModelServerType, StandardLoadStage
    from inference_perf.detector.detector import autofill_config

    base_config = Config(
        server=ModelServerClientConfig(type=ModelServerType.VLLM, base_url="http://old-server"),
        load=LoadConfig(stages=[StandardLoadStage(rate=5.0, duration=50)]),
    )

    with (
        patch("inference_perf.detector.detector.detect_server_type") as mock_detect_type,
        patch("inference_perf.detector.detector.detect_environment") as mock_detect_env,
    ):
        mock_detect_type.return_value = ModelServerType.TGI
        mock_detect_env.return_value = EnvironmentType.LOCAL

        patched = autofill_config("http://new-server", base_config=base_config)

        assert patched.server is not None  # For mypy
        assert patched.server.type == ModelServerType.TGI
        assert patched.server.base_url == "http://new-server"
        stage = patched.load.stages[0]
        assert isinstance(stage, StandardLoadStage)
        assert stage.rate == 5.0  # Preserved!


def test_detect_api_type_chat() -> None:
    from inference_perf.config import APIType
    from inference_perf.detector.detector import detect_api_type

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"paths": {"/v1/chat/completions": {}}}
        mock_get.return_value = mock_response

        assert detect_api_type("http://dummy") == APIType.Chat


def test_detect_api_type_completion() -> None:
    from inference_perf.config import APIType
    from inference_perf.detector.detector import detect_api_type

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"paths": {"/v1/completions": {}}}
        mock_get.return_value = mock_response

        assert detect_api_type("http://dummy") == APIType.Completion


def test_detect_api_type_both_prefers_chat() -> None:
    from inference_perf.config import APIType
    from inference_perf.detector.detector import detect_api_type

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"paths": {"/v1/chat/completions": {}, "/v1/completions": {}}}
        mock_get.return_value = mock_response

        assert detect_api_type("http://dummy") == APIType.Chat


def test_detect_api_type_fallback() -> None:
    from inference_perf.config import APIType
    from inference_perf.detector.detector import detect_api_type

    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("Connection error")

        assert detect_api_type("http://dummy") == APIType.Completion


def test_detect_model_name_success() -> None:
    from inference_perf.detector.detector import detect_model_name

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "data": [{"id": "test-model", "object": "model"}],
        }
        mock_get.return_value = mock_response

        assert detect_model_name("http://dummy") == "test-model"


def test_detect_model_name_empty_list() -> None:
    from inference_perf.detector.detector import detect_model_name

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"object": "list", "data": []}
        mock_get.return_value = mock_response

        assert detect_model_name("http://dummy") is None


def test_detect_model_name_fallback() -> None:
    from inference_perf.detector.detector import detect_model_name

    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("Connection error")

        assert detect_model_name("http://dummy") is None
