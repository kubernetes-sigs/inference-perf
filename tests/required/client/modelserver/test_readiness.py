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
"""Server readiness wait before the run contacts the server (#979).

The vLLM profile names ``/health`` as its readiness route. With
``server.readiness_timeout`` set, client construction polls it until it answers
200, before model auto-detection, and fails at the deadline with the last status
it saw. No network: ``requests.get`` and the clock are faked.
"""

from typing import Any, List, Optional, Type
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import ValidationError

from backend_client_suite import BASE_URI, MODEL, StubTokenizer
from inference_perf.client.modelserver.openai_client import (
    _READINESS_PROBE_TIMEOUT_SEC,
    openAIModelServerClient,
)
from inference_perf.client.modelserver.sglang_client import SGlangModelServerClient
from inference_perf.client.modelserver.tgi_client import TGImodelServerClient
from inference_perf.client.modelserver.vllm_client import vLLMModelServerClient
from inference_perf.config import APIConfig, APIType
from inference_perf.config.client.modelserver.config import ModelServerClientConfig

HEALTH_URL = f"{BASE_URI}/health"


# A clock that only moves when the code under test sleeps: sleep(1.0) at t=0 -> monotonic() == 1.0.
class FakeClock:
    # Starts at t=0 with no sleeps recorded.
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: List[float] = []

    # Returns the current fake time, e.g. 0.0 before any sleep.
    def monotonic(self) -> float:
        return self.now

    # sleep(1.0) at t=2.0 -> t=3.0, and 1.0 appended to sleeps.
    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# status 200 -> a response whose status_code is 200 and whose json() is `body`.
def _response(status: int, body: Optional[Any] = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = body
    return response


# vLLM client, model_name=MODEL, readiness_timeout=10 -> constructed with requests.get and the clock faked.
def _build(
    mock_get: MagicMock,
    clock: FakeClock,
    client_cls: Type[openAIModelServerClient] = vLLMModelServerClient,
    model_name: Optional[str] = MODEL,
    readiness_timeout: Optional[float] = 10.0,
    **kwargs: Any,
) -> openAIModelServerClient:
    with (
        patch("inference_perf.client.modelserver.openai_client.CustomTokenizer", StubTokenizer),
        patch("inference_perf.client.modelserver.openai_client.requests.get", mock_get),
        patch("inference_perf.client.modelserver.openai_client.time", clock),
    ):
        return client_cls(
            metrics_collector=MagicMock(),
            api_config=APIConfig(type=APIType.Completion, streaming=False),
            uri=BASE_URI,
            model_name=model_name,
            tokenizer_config=None,
            max_tcp_connections=4,
            additional_filters=[],
            readiness_timeout=readiness_timeout,
            **kwargs,
        )


# vLLM profile -> "/health"; SGLang and TGI profiles -> None until their own issues (#991, #997) set one.
def test_readiness_route_per_profile() -> None:
    assert vLLMModelServerClient.readiness_route == "/health"
    assert SGlangModelServerClient.readiness_route is None
    assert TGImodelServerClient.readiness_route is None


# readiness_timeout unset with a pinned model name -> no HTTP request at all during construction.
def test_unset_timeout_sends_no_probe() -> None:
    mock_get = MagicMock()
    _build(mock_get, FakeClock(), readiness_timeout=None)
    mock_get.assert_not_called()


# /health answers 503, then refuses the connection, then 200 -> three probes, two 1 s sleeps, construction succeeds.
def test_polls_until_health_answers_200() -> None:
    mock_get = MagicMock(
        side_effect=[_response(503), requests.ConnectionError("connection refused"), _response(200)],
    )
    clock = FakeClock()
    _build(mock_get, clock)
    assert mock_get.call_count == 3
    assert clock.sleeps == [1.0, 1.0]
    for call in mock_get.call_args_list:
        assert call.args == (HEALTH_URL,)
        assert call.kwargs["headers"] == {"Content-Type": "application/json"}
        assert call.kwargs["timeout"] == _READINESS_PROBE_TIMEOUT_SEC
        assert call.kwargs["cert"] is None


# api_key "k", cert and key paths set -> the probe carries "Authorization: Bearer k" and the client certificate.
def test_probe_sends_auth_header_and_client_certificate() -> None:
    mock_get = MagicMock(return_value=_response(200))
    _build(mock_get, FakeClock(), api_key="k", cert_path="/tmp/c.crt", key_path="/tmp/c.key")
    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer k"
    assert mock_get.call_args.kwargs["cert"] == ("/tmp/c.crt", "/tmp/c.key")


# /health answers 503 forever, readiness_timeout=3 -> raises at t=3 naming the URL and "HTTP 503".
def test_deadline_raises_with_last_status() -> None:
    mock_get = MagicMock(return_value=_response(503))
    clock = FakeClock()
    with pytest.raises(Exception, match=rf"{HEALTH_URL} did not answer 200 within 3.0s \(last: HTTP 503\)"):
        _build(mock_get, clock, readiness_timeout=3.0)
    assert clock.now == 3.0
    assert mock_get.call_count == 3


# readiness_timeout=2.5 -> no probe waits past the deadline: the third probe gets timeout 0.5, the last sleep 0.5.
def test_probe_and_sleep_never_overrun_the_deadline() -> None:
    mock_get = MagicMock(side_effect=requests.ConnectionError("connection refused"))
    clock = FakeClock()
    with pytest.raises(Exception, match=r"last: ConnectionError: connection refused"):
        _build(mock_get, clock, readiness_timeout=2.5)
    assert [call.kwargs["timeout"] for call in mock_get.call_args_list] == [2.5, 1.5, 0.5]
    assert clock.sleeps == [1.0, 1.0, 0.5]


# model_name unset -> /health is polled to 200 first, then /v1/models is probed for the model id.
def test_waits_before_model_auto_detection() -> None:
    mock_get = MagicMock(
        side_effect=[_response(503), _response(200), _response(200, {"data": [{"id": MODEL}]})],
    )
    client = _build(mock_get, FakeClock(), model_name=None)
    assert [call.args[0] for call in mock_get.call_args_list] == [HEALTH_URL, HEALTH_URL, f"{BASE_URI}/v1/models"]
    assert client.model_name == MODEL


# SGLang (no readiness route) with readiness_timeout=10 -> a warning, no probe, construction succeeds.
def test_profile_without_route_warns_and_skips(caplog: pytest.LogCaptureFixture) -> None:
    mock_get = MagicMock()
    _build(mock_get, FakeClock(), client_cls=SGlangModelServerClient)
    mock_get.assert_not_called()
    assert "SGlangModelServerClient has no readiness route" in caplog.text


# server config without readiness_timeout -> None; 0 or a negative number -> rejected at config load.
@pytest.mark.parametrize("bad", [0, -1.0])
def test_config_readiness_timeout_must_be_positive(bad: float) -> None:
    assert ModelServerClientConfig(base_url=BASE_URI).readiness_timeout is None
    assert ModelServerClientConfig(base_url=BASE_URI, readiness_timeout=600).readiness_timeout == 600
    with pytest.raises(ValidationError):
        ModelServerClientConfig(base_url=BASE_URI, readiness_timeout=bad)
