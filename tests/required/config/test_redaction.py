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
"""

import logging
import os
import tempfile
from typing import Any, Optional

import pytest
import yaml
from pydantic import BaseModel, SecretStr
from unittest.mock import Mock

from inference_perf.config import Config, read_config
from inference_perf.config.redaction import (
    REDACTED,
    _nested_models,
    _unwrap_optional,
    credential_locations,
    redact,
)
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


def _read(config: dict[str, Any]) -> Config:
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    try:
        return read_config(path)
    finally:
        os.unlink(path)


def _saved_config_report(config: Config) -> str:
    generator = ReportGenerator(metrics_client=None, metrics_collector=Mock(), config=config)
    return yaml.dump(generator.generate_config_report().get_contents())


def _field_paths(
    model: type[BaseModel],
    prefix: tuple[str, ...] = (),
    chain: tuple[type[BaseModel], ...] = (),
) -> set[tuple[str, ...]]:
    """Every field path in a config model, for the schema sweep below."""
    if model in chain:
        return set()
    paths: set[tuple[str, ...]] = set()
    for name, field in model.model_fields.items():
        path = prefix + (name,)
        paths.add(path)
        for nested in _nested_models(_unwrap_optional(field.annotation)):
            paths |= _field_paths(nested, path, chain + (model,))
    return paths


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
    any other setting. The vocabulary below is deliberately narrow: over the whole
    schema it matches the two credentials and nothing else, so a field it does
    match is one whose name says it holds a credential.
    """
    exact = {"token", "key", "password", "secret"}
    substrings = ("api_key", "apikey", "secret", "password", "passwd", "credential", "access_token", "auth_token", "bearer")

    paths = _field_paths(Config)
    # The sweep has to reach nested sections, or it would find nothing and pass.
    assert ("report", "request_lifecycle", "summary") in paths

    named_like_a_credential = {path for path in paths if path[-1] in exact or any(hint in path[-1] for hint in substrings)}
    secret_paths, _ = credential_locations(Config)

    assert named_like_a_credential == set(secret_paths), (
        "A config field named like a credential is not typed SecretStr, so it will be "
        "printed to the log and written into the report bundle."
    )


def test_redaction_reaches_a_credential_inside_a_list() -> None:
    """A config section that repeats is masked in every element, not just the first."""

    class Endpoint(BaseModel):
        name: str
        api_key: Optional[SecretStr] = None

    class Fleet(BaseModel):
        endpoints: list[Endpoint] = []

    data = {"endpoints": [{"name": "a", "api_key": "first"}, {"name": "b", "api_key": "second"}]}

    assert redact(data, Fleet) == {"endpoints": [{"name": "a", "api_key": REDACTED}, {"name": "b", "api_key": REDACTED}]}
