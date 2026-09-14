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
"""A run renders its config to the startup log and into the report bundle. Neither
copy may carry the credentials the run was given.

The leaks pinned here are the ones a benchmark actually hits: server.api_key,
tokenizer.token and an api.headers authentication header printed verbatim by
read_config, and the same three written into the config.yaml that is uploaded to
object storage.

It also pins that loading a saved config fails until its credentials are supplied
again.
"""

import logging
import os
import tempfile
from typing import Annotated, Any, Callable, Iterator, Optional

import pytest
import yaml
from pydantic import BaseModel, Field, SecretStr
from unittest.mock import Mock

from inference_perf.config import Config, read_config
from inference_perf.config.redaction import REDACTED, _rules, _shapes, redact, redacted_credentials
from inference_perf.reportgen.base import ReportGenerator

API_KEY = "sk-api-key-that-must-not-be-printed"
HF_TOKEN = "hf_token_that_must_not_be_printed"
AUTH_HEADER = "Bearer gateway-token-that-must-not-be-printed"

CREDENTIALS = (API_KEY, HF_TOKEN, AUTH_HEADER)


def _config_dict(with_credentials: bool = True) -> dict[str, Any]:
    config: dict[str, Any] = {
        "api": {"type": "chat", "headers": {"x-routing-strategy": "round-robin"}},
        "data": {"type": "random"},
        "load": {"type": "constant", "stages": [{"rate": 1, "duration": 5}]},
        "server": {"type": "vllm", "base_url": "http://localhost:8000"},
        "tokenizer": {"pretrained_model_name_or_path": "gpt2"},
    }
    if with_credentials:
        config["api"]["headers"]["Authorization"] = AUTH_HEADER
        config["server"]["api_key"] = API_KEY
        config["tokenizer"]["token"] = HF_TOKEN
    return config


def _read(config: dict[str, Any], cli_overrides: Optional[dict[str, Any]] = None) -> Config:
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    try:
        return read_config(path, cli_overrides)
    finally:
        os.unlink(path)


def _saved_config_report(config: Config) -> str:
    generator = ReportGenerator(metrics_client=None, metrics_collector=Mock(), config=config)
    return yaml.dump(generator.generate_config_report().get_contents())


def _saved_config(config: dict[str, Any]) -> dict[str, Any]:
    saved: dict[str, Any] = yaml.safe_load(_saved_config_report(_read(config)))
    return saved


def _fields(
    model: type[BaseModel],
    path: tuple[str, ...] = (),
    chain: tuple[type[BaseModel], ...] = (),
) -> Iterator[tuple[type[BaseModel], str, tuple[str, ...]]]:
    """Every field reachable from a config model, with the model that declares it."""
    if model in chain:
        return
    for name, field in model.model_fields.items():
        yield model, name, path + (name,)
        for shape, _ in _shapes(field.annotation):
            if isinstance(shape, type) and issubclass(shape, BaseModel):
                yield from _fields(shape, path + (name,), chain + (model,))


def _sweep(model: type[BaseModel]) -> tuple[set[tuple[type[BaseModel], str]], set[tuple[type[BaseModel], str]]]:
    """The fields named like a credential, and the fields the walk treats as secrets.

    The vocabulary is deliberately narrow: over the whole schema it matches the two
    credentials and nothing else, so a field it does match is one whose name says it
    holds a credential.
    """
    exact = {"token", "key", "password", "secret"}
    substrings = ("api_key", "apikey", "secret", "password", "passwd", "credential", "access_token", "auth_token", "bearer")

    owners: set[type[BaseModel]] = set()
    named: set[tuple[type[BaseModel], str]] = set()
    for owner, name, _ in _fields(model):
        owners.add(owner)
        if name in exact or any(hint in name for hint in substrings):
            named.add((owner, name))
    secrets = {(owner, path[0]) for owner in owners for path in _rules(owner).secrets}
    return named, secrets


class _Upstream(BaseModel):
    api_key: Optional[Annotated[SecretStr, Field(min_length=1)]] = None
    token: SecretStr | None = None
    headers: Optional[dict[str, str]] = None
    fallback: Optional["_Upstream"] = None


class _Gateway(BaseModel):
    routes: dict[str, _Upstream] = {}
    pool: list[_Upstream] = []
    api_keys: dict[str, SecretStr] = {}


def test_logged_config_hides_credentials(caplog: pytest.LogCaptureFixture) -> None:
    """The echoed config is the copy the bug report template asks people to paste."""
    with caplog.at_level(logging.INFO, logger="inference_perf.config.config"):
        _read(_config_dict())

    logged = caplog.text
    assert "Benchmarking with the following config" in logged
    for credential in CREDENTIALS:
        assert credential not in logged
    assert logged.count(REDACTED) == len(CREDENTIALS)


def test_saved_config_report_hides_credentials() -> None:
    """The report bundle is uploaded to GCS or S3 and passed around."""
    saved = _saved_config_report(_read(_config_dict()))

    for credential in CREDENTIALS:
        assert credential not in saved
    assert saved.count(REDACTED) == len(CREDENTIALS)


def test_redaction_keeps_the_rest_of_the_headers() -> None:
    """Routing headers are part of reproducing a run, so only the credential goes."""
    redacted = redact(_read(_config_dict()).model_dump(mode="json"), Config)

    assert redacted["api"]["headers"] == {"x-routing-strategy": "round-robin", "Authorization": REDACTED}


def test_headers_named_like_a_credential_are_masked() -> None:
    """Gateways name their key headers differently, so the match is on name fragments."""
    headers = {"apikey": "kong", "X-Auth-Token": "token", "Ocp-Apim-Subscription-Key": "azure", "x-routing-strategy": "rr"}

    assert redact({"api": {"headers": headers}}, Config)["api"]["headers"] == {
        "apikey": REDACTED,
        "X-Auth-Token": REDACTED,
        "Ocp-Apim-Subscription-Key": REDACTED,
        "x-routing-strategy": "rr",
    }


def test_credentials_reach_the_run_intact() -> None:
    """Masking is for the rendered copies. The run still gets the real values."""
    config = _read(_config_dict())

    assert config.server is not None and config.server.api_key is not None
    assert config.server.api_key.get_secret_value() == API_KEY
    assert config.tokenizer is not None and config.tokenizer.token is not None
    assert config.tokenizer.token.get_secret_value() == HF_TOKEN


def test_config_without_credentials_renders_unchanged() -> None:
    """The common case has nothing to hide and must render exactly as it did before."""
    dumped = _read(_config_dict(with_credentials=False)).model_dump(mode="json")

    assert redact(dumped, Config) == dumped


def test_every_credential_field_in_the_schema_is_a_secret() -> None:
    """Catches a credential added to the config as a plain string.

    Redaction follows the type, so a new key declared `str` would be printed like
    any other setting.
    """
    # The sweep has to reach nested sections, or it would find nothing and pass.
    assert ("report", "request_lifecycle", "summary") in {path for _, _, path in _fields(Config)}

    named_like_a_credential, secrets = _sweep(Config)

    assert named_like_a_credential == secrets, (
        "A config field named like a credential is not typed SecretStr, so it will be "
        "printed to the log and written into the report bundle."
    )


def test_walk_and_sweep_cover_every_shape_a_secret_can_take() -> None:
    """Declaring SecretStr is enough inside Annotated, `X | None`, lists, dicts and recursive models."""
    data = {
        "routes": {"gpu": {"api_key": "a", "token": "b", "headers": {"apikey": "c"}, "fallback": {"api_key": "d"}}},
        "pool": [{"token": "e"}],
        "api_keys": {"primary": "f"},
    }
    redacted = redact(data, _Gateway)

    assert redacted == {
        "routes": {
            "gpu": {"api_key": REDACTED, "token": REDACTED, "headers": {"apikey": REDACTED}, "fallback": {"api_key": REDACTED}}
        },
        "pool": [{"token": REDACTED}],
        "api_keys": {"primary": REDACTED},
    }
    assert redacted_credentials(redacted, _Gateway) == [
        "api_keys.primary",
        "pool.0.token",
        "routes.gpu.api_key",
        "routes.gpu.fallback.api_key",
        "routes.gpu.headers.apikey",
        "routes.gpu.token",
    ]
    expected = {(_Upstream, "api_key"), (_Upstream, "token"), (_Gateway, "api_keys")}
    assert _sweep(_Gateway) == (expected, expected)


@pytest.mark.parametrize("load", [_read, Config.model_validate], ids=["read_config", "model_validate"])
def test_saved_config_is_rejected_while_it_holds_the_placeholder(load: Callable[[dict[str, Any]], Config]) -> None:
    """Re-running from a report bundle has to fail on load, not as a 401 from the server."""
    saved = _saved_config(_config_dict())

    with pytest.raises(ValueError) as error:
        load(saved)

    message = str(error.value)
    assert REDACTED in message
    for setting in ("server.api_key", "tokenizer.token", "api.headers.Authorization"):
        assert setting in message


def test_config_dumped_by_pydantic_is_rejected() -> None:
    """model_dump(mode="json") writes ********** for a secret, which is no more usable than the marker."""
    dumped = _read(_config_dict()).model_dump(mode="json")

    with pytest.raises(ValueError) as error:
        Config.model_validate(dumped)

    message = str(error.value)
    assert "server.api_key" in message and "tokenizer.token" in message


def test_rejection_does_not_echo_the_credentials_that_are_set() -> None:
    """Pydantic quotes part of the input in its errors, and that input holds real credentials."""
    config = _config_dict()
    config["api"]["headers"]["Authorization"] = REDACTED

    with pytest.raises(ValueError) as error:
        Config.model_validate(config)

    message = str(error.value)
    assert "api.headers.Authorization" in message
    for credential in (API_KEY, HF_TOKEN):
        assert credential[-10:] not in message


def test_saved_config_loads_once_its_credentials_are_supplied_again() -> None:
    """The command line overrides the file, which is how a saved run is repeated."""
    config = _read(
        _saved_config(_config_dict()),
        cli_overrides={
            "api": {"headers": {"Authorization": AUTH_HEADER}},
            "server": {"api_key": API_KEY},
            "tokenizer": {"token": HF_TOKEN},
        },
    )

    assert config.api.headers is not None and config.api.headers["Authorization"] == AUTH_HEADER
    assert config.server is not None and config.server.api_key is not None
    assert config.server.api_key.get_secret_value() == API_KEY
    assert config.tokenizer is not None and config.tokenizer.token is not None
    assert config.tokenizer.token.get_secret_value() == HF_TOKEN


def test_placeholder_check_follows_the_header_the_client_sends() -> None:
    """Header names match ignoring case, and the client sends the last one given."""
    supplied_again = {"api": {"headers": {"Authorization": REDACTED, "authorization": AUTH_HEADER}}}
    overridden = {"api": {"headers": {"authorization": AUTH_HEADER, "Authorization": REDACTED}}}

    assert redacted_credentials(supplied_again, Config) == []
    assert redacted_credentials(overridden, Config) == ["api.headers.Authorization"]


def test_empty_credentials_are_not_masked(caplog: pytest.LogCaptureFixture) -> None:
    """An empty credential, as in the `api_key: ""` docs example, is not masked and loads back."""
    config = _config_dict(with_credentials=False)
    config["api"]["headers"]["Authorization"] = ""
    config["server"]["api_key"] = ""
    config["tokenizer"]["token"] = ""

    with caplog.at_level(logging.INFO, logger="inference_perf.config.config"):
        saved = _saved_config(config)

    assert "Benchmarking with the following config" in caplog.text
    assert REDACTED not in caplog.text
    assert saved["api"]["headers"]["Authorization"] == ""
    assert saved["server"]["api_key"] == ""
    assert saved["tokenizer"]["token"] == ""
    _read(saved)
