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

Client construction always polls a readiness route until it answers 200, before model
auto-detection: the profile's own route when it names one (vLLM: ``/health``), otherwise
``/v1/models``. A 404 means the route is not served, so the run carries on; anything else
is polled until ``readiness_timeout_sec`` runs out. No network: ``requests.get`` and the
clock are faked.
"""

import socket
from typing import Any, List, Optional, Type
from unittest.mock import MagicMock, patch

import pytest
import requests

from backend_client_suite import BASE_URI, MODEL, StubTokenizer
from inference_perf.client.modelserver.openai_client import (
    _READINESS_PROBE_TIMEOUT_SEC,
    openAIModelServerClient,
)
from inference_perf.client.modelserver.sglang_client import SGlangModelServerClient
from inference_perf.client.modelserver.tgi_client import TGImodelServerClient
from inference_perf.client.modelserver.vllm_client import vLLMModelServerClient
from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig, MultiLoRAConfig, WaitFor

HEALTH_URL = f"{BASE_URI}/health"
MODELS_URL = f"{BASE_URI}/v1/models"


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


# vLLM client, model_name=MODEL, readiness budget 10 s -> constructed with requests.get and the clock faked.
def _build(
    mock_get: MagicMock,
    clock: FakeClock,
    client_cls: Type[openAIModelServerClient] = vLLMModelServerClient,
    model_name: Optional[str] = MODEL,
    timeout_sec: float = 10.0,
    **kwargs: Any,
) -> openAIModelServerClient:
    with (
        patch("inference_perf.client.modelserver.openai_client.CustomTokenizer", StubTokenizer),
        patch("inference_perf.client.modelserver.openai_client.requests.get", mock_get),
        patch("inference_perf.client.modelserver.openai_client.time", clock),
        patch.object(client_cls, "readiness_timeout_sec", timeout_sec),
    ):
        return client_cls(
            metrics_collector=MagicMock(),
            api_config=APIConfig(type=APIType.Completion, streaming=False),
            uri=BASE_URI,
            model_name=model_name,
            tokenizer_config=None,
            max_tcp_connections=4,
            additional_filters=[],
            **kwargs,
        )


# vLLM profile -> "/health"; SGLang and TGI profiles -> None, so they probe "/v1/models".
def test_readiness_route_per_profile() -> None:
    assert vLLMModelServerClient.readiness_route == "/health"
    assert SGlangModelServerClient.readiness_route is None
    assert TGImodelServerClient.readiness_route is None


# Every profile inherits the 600 s budget; a profile that boots slower overrides the class attribute.
def test_readiness_budget_is_a_class_attribute() -> None:
    assert openAIModelServerClient.readiness_timeout_sec == 600.0
    assert vLLMModelServerClient.readiness_timeout_sec == 600.0
    assert SGlangModelServerClient.readiness_timeout_sec == 600.0


# A pinned model name no longer skips the wait: construction still probes /health once.
def test_pinned_model_name_still_waits() -> None:
    mock_get = MagicMock(return_value=_response(200))
    _build(mock_get, FakeClock())
    mock_get.assert_called_once()
    assert mock_get.call_args.args == (HEALTH_URL,)


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


# /health answers 503 forever, budget 3 s -> raises at t=3 naming the URL and "HTTP 503".
def test_deadline_raises_with_last_status() -> None:
    mock_get = MagicMock(return_value=_response(503))
    clock = FakeClock()
    with pytest.raises(Exception, match=rf"{HEALTH_URL} did not answer 200 within 3.0s \(last: HTTP 503\)"):
        _build(mock_get, clock, timeout_sec=3.0)
    assert clock.now == 3.0
    assert mock_get.call_count == 3


# Budget 2.5 s -> no probe waits past the deadline: the third probe gets timeout 0.5, the last sleep 0.5.
def test_probe_and_sleep_never_overrun_the_deadline() -> None:
    mock_get = MagicMock(side_effect=requests.ConnectionError("connection refused"))
    clock = FakeClock()
    with pytest.raises(Exception, match=r"last: ConnectionError: connection refused"):
        _build(mock_get, clock, timeout_sec=2.5)
    assert [call.kwargs["timeout"] for call in mock_get.call_args_list] == [2.5, 1.5, 0.5]
    assert clock.sleeps == [1.0, 1.0, 0.5]


# model_name unset -> /health is polled to 200 first, then /v1/models is probed for the model id.
def test_waits_before_model_auto_detection() -> None:
    mock_get = MagicMock(
        side_effect=[_response(503), _response(200), _response(200, {"data": [{"id": MODEL}]})],
    )
    client = _build(mock_get, FakeClock(), model_name=None)
    assert [call.args[0] for call in mock_get.call_args_list] == [HEALTH_URL, HEALTH_URL, MODELS_URL]
    assert client.model_name == MODEL


# SGLang (no readiness route) with /v1/models answering 200 -> one probe to /v1/models, construction succeeds.
def test_profile_without_route_probes_models(caplog: pytest.LogCaptureFixture) -> None:
    mock_get = MagicMock(return_value=_response(200))
    _build(mock_get, FakeClock(), client_cls=SGlangModelServerClient)
    mock_get.assert_called_once()
    assert mock_get.call_args.args == (MODELS_URL,)
    assert "does not serve" not in caplog.text


# vLLM's /health answers 404 -> the route is not served here, the wait ends after one probe and the run starts.
def test_404_route_is_not_waited_for() -> None:
    mock_get = MagicMock(return_value=_response(404))
    clock = FakeClock()
    _build(mock_get, clock)
    mock_get.assert_called_once()
    assert clock.sleeps == []


# /health answers 404 after one 503 -> the wait ends on the 404 instead of polling to the budget.
def test_404_ends_the_wait_after_earlier_failures() -> None:
    mock_get = MagicMock(side_effect=[_response(503), _response(404)])
    clock = FakeClock()
    _build(mock_get, clock)
    assert mock_get.call_count == 2
    assert clock.sleeps == [1.0]


# --- server.wait_for (#1065) ---------------------------------------------
# reachable waits for any HTTP response from the same route; none does not wait. Neither asks the
# server anything else before the run. The last two tests probe a local socket, not a faked requests.get.


# wait_for=reachable and /health answers 503, as a gateway does for a pool scaled to zero -> one probe, no sleep,
# and no other request (no /v1/models lookup for the model, tokenizer or LoRA adapters).
def test_reachable_any_status_ends_the_wait() -> None:
    mock_get = MagicMock(return_value=_response(503))
    clock = FakeClock()
    client = _build(mock_get, clock, wait_for=WaitFor.REACHABLE)
    mock_get.assert_called_once()
    assert mock_get.call_args.args == (HEALTH_URL,)
    assert clock.sleeps == []
    assert client.model_name == MODEL


# wait_for=reachable: connection refused, then a connect timeout, then HTTP 503 -> three probes and two 1 s sleeps.
def test_reachable_polls_through_connection_errors() -> None:
    mock_get = MagicMock(
        side_effect=[
            requests.ConnectionError("connection refused"),
            requests.ConnectTimeout("connect timed out"),
            _response(503),
        ],
    )
    clock = FakeClock()
    _build(mock_get, clock, wait_for=WaitFor.REACHABLE)
    assert mock_get.call_count == 3
    assert clock.sleeps == [1.0, 1.0]


# wait_for=reachable and the probe connects but gets no response in time (a gateway holding it) -> reachable, one probe.
def test_reachable_counts_a_read_timeout() -> None:
    mock_get = MagicMock(side_effect=requests.ReadTimeout("read timed out"))
    clock = FakeClock()
    _build(mock_get, clock, wait_for=WaitFor.REACHABLE)
    mock_get.assert_called_once()
    assert clock.sleeps == []


# wait_for=reachable, api_key "k", cert and key paths set -> the probe carries "Authorization: Bearer k" and the cert.
def test_reachable_probe_sends_auth_header_and_client_certificate() -> None:
    mock_get = MagicMock(return_value=_response(401))
    _build(mock_get, FakeClock(), wait_for=WaitFor.REACHABLE, api_key="k", cert_path="/tmp/c.crt", key_path="/tmp/c.key")
    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer k"
    assert mock_get.call_args.kwargs["cert"] == ("/tmp/c.crt", "/tmp/c.key")


# wait_for=reachable, connection refused forever, budget 2.5 s -> probe timeouts 2.5, 1.5, 0.5, sleeps 1, 1, 0.5,
# then an error naming the URL and the last error.
def test_reachable_deadline_raises_with_last_error() -> None:
    mock_get = MagicMock(side_effect=requests.ConnectionError("connection refused"))
    clock = FakeClock()
    with pytest.raises(
        Exception,
        match=rf"{HEALTH_URL} gave no HTTP response within 2.5s \(last: ConnectionError: connection refused\)",
    ):
        _build(mock_get, clock, wait_for=WaitFor.REACHABLE, timeout_sec=2.5)
    assert [call.kwargs["timeout"] for call in mock_get.call_args_list] == [2.5, 1.5, 0.5]
    assert clock.sleeps == [1.0, 1.0, 0.5]
    assert clock.now == 2.5


# wait_for=reachable with a budget of 0 s -> raises before any probe, with "last: no response".
def test_reachable_zero_budget_raises_without_probing() -> None:
    mock_get = MagicMock()
    with pytest.raises(Exception, match=r"gave no HTTP response within 0.0s \(last: no response\)"):
        _build(mock_get, FakeClock(), wait_for=WaitFor.REACHABLE, timeout_sec=0.0)
    mock_get.assert_not_called()


# SGLang (no readiness route) with wait_for=reachable -> probes /v1/models, the same route the ready wait uses.
def test_reachable_profile_without_route_probes_models() -> None:
    mock_get = MagicMock(return_value=_response(503))
    _build(mock_get, FakeClock(), client_cls=SGlangModelServerClient, wait_for=WaitFor.REACHABLE)
    mock_get.assert_called_once()
    assert mock_get.call_args.args == (MODELS_URL,)


# wait_for=none -> no request at all before the run, and no sleep.
def test_none_sends_no_request() -> None:
    mock_get = MagicMock()
    clock = FakeClock()
    client = _build(mock_get, clock, wait_for=WaitFor.NONE)
    mock_get.assert_not_called()
    assert clock.sleeps == []
    assert client.model_name == MODEL


# wait_for=none without model_name -> ValueError naming model_name, before any request (the name would come from
# /v1/models).
def test_none_without_model_name_raises() -> None:
    mock_get = MagicMock()
    with pytest.raises(ValueError, match="wait_for none requires model_name"):
        _build(mock_get, FakeClock(), model_name=None, wait_for=WaitFor.NONE)
    mock_get.assert_not_called()


# wait_for=none with a LoRA adapter the server may not list yet -> no /v1/models check, and a log line saying so.
def test_none_skips_the_lora_adapter_check(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    mock_get = MagicMock()
    _build(mock_get, FakeClock(), wait_for=WaitFor.NONE, lora_config=[MultiLoRAConfig(name="adapter-a", split=1.0)])
    mock_get.assert_not_called()
    assert "LoRA adapter names are not checked against /v1/models" in caplog.text


# wait_for=none, no tokenizer path, and model_name "pinned-alias" has no tokenizer files -> the tokenizer error names
# tokenizer.pretrained_model_name_or_path, with no /v1/models probe for the entry's root.
def test_none_does_not_probe_models_for_the_tokenizer_root() -> None:
    # Loads only MODEL; any other path raises, as an alias with no tokenizer files does.
    class AliasRejectingTokenizer:
        # config.pretrained_model_name_or_path "pinned-alias" -> RuntimeError.
        def __init__(self, config: Optional[CustomTokenizerConfig] = None) -> None:
            if config is None or config.pretrained_model_name_or_path != MODEL:
                raise RuntimeError("no tokenizer files")

    mock_get = MagicMock()
    with (
        patch("inference_perf.client.modelserver.openai_client.CustomTokenizer", AliasRejectingTokenizer),
        patch("inference_perf.client.modelserver.openai_client.requests.get", mock_get),
        patch("inference_perf.client.modelserver.openai_client.time", FakeClock()),
        pytest.raises(Exception, match="Set 'tokenizer.pretrained_model_name_or_path' explicitly"),
    ):
        vLLMModelServerClient(
            metrics_collector=MagicMock(),
            api_config=APIConfig(type=APIType.Completion, streaming=False),
            uri=BASE_URI,
            model_name="pinned-alias",
            tokenizer_config=None,
            max_tcp_connections=4,
            additional_filters=[],
            wait_for=WaitFor.NONE,
        )
    mock_get.assert_not_called()


# vLLM client against a real local address, wait_for=reachable, probe timeout 0.2 s, readiness clock faked.
def _build_against(port: int, clock: FakeClock) -> openAIModelServerClient:
    with (
        patch("inference_perf.client.modelserver.openai_client.CustomTokenizer", StubTokenizer),
        patch("inference_perf.client.modelserver.openai_client.time", clock),
        patch("inference_perf.client.modelserver.openai_client._READINESS_PROBE_TIMEOUT_SEC", 0.2),
        patch.object(vLLMModelServerClient, "readiness_timeout_sec", 3.0),
    ):
        return vLLMModelServerClient(
            metrics_collector=MagicMock(),
            api_config=APIConfig(type=APIType.Completion, streaming=False),
            uri=f"http://127.0.0.1:{port}",
            model_name=MODEL,
            tokenizer_config=None,
            max_tcp_connections=4,
            additional_filters=[],
            wait_for=WaitFor.REACHABLE,
        )


# A socket that accepts connections and never answers, as a gateway holding requests for an empty pool ->
# the first probe times out reading and the wait ends without a sleep.
def test_reachable_against_a_socket_that_holds_the_request() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        clock = FakeClock()
        _build_against(listener.getsockname()[1], clock)
    assert clock.sleeps == []


# A port nothing listens on, budget 3 s -> every probe is refused, three sleeps, then the deadline error.
def test_reachable_against_a_closed_port_waits_out_the_budget() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    clock = FakeClock()
    with pytest.raises(Exception, match=r"gave no HTTP response within 3.0s \(last: ConnectionError"):
        _build_against(port, clock)
    assert clock.sleeps == [1.0, 1.0, 1.0]
