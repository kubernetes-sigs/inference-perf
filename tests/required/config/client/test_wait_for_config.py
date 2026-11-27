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
"""Validity rules for ``server.wait_for`` and ``server.wait_timeout`` in ``inference_perf.config.client.modelserver``."""

from argparse import ArgumentParser
from typing import List

import pytest

from inference_perf.config import Config, ModelServerClientConfig, WaitFor, read_config
from inference_perf.utils.cli_parser import add_global_args, add_pydantic_args, unflatten_dict


# Parses argv the way main_cli does, global args first and then every Config field as a flag, and returns the Config.
def _config_from_argv(argv: List[str]) -> Config:
    parser = ArgumentParser()
    base_args = add_global_args(parser)
    add_pydantic_args(parser, Config)
    args = parser.parse_args(argv)
    overrides = unflatten_dict({k: v for k, v in vars(args).items() if k not in base_args})
    return read_config(args.config_file, overrides)


# No wait_for, no model_name -> wait_for is "ready" and the name is left for auto-detection.
def test_wait_for_defaults_to_ready() -> None:
    config = ModelServerClientConfig(base_url="http://localhost:8000")
    assert config.wait_for == WaitFor.READY
    assert config.model_name is None


# wait_for "reachable" or "none" given as YAML strings, with model_name -> parsed to the enum.
@pytest.mark.parametrize("value", ["reachable", "none"])
def test_wait_for_without_lookup_accepts_a_model_name(value: str) -> None:
    config = ModelServerClientConfig.model_validate({"base_url": "http://gateway:80", "model_name": "m", "wait_for": value})
    assert config.wait_for == WaitFor(value)


# wait_for "reachable" or "none" without model_name -> error naming both fields.
@pytest.mark.parametrize("value", ["reachable", "none"])
def test_wait_for_without_lookup_requires_model_name(value: str) -> None:
    with pytest.raises(ValueError, match=rf"'wait_for: {value}' requires 'model_name'"):
        ModelServerClientConfig.model_validate({"base_url": "http://gateway:80", "wait_for": value})


# --server.wait_for reachable on the command line, with a model name -> server.wait_for is WaitFor.REACHABLE.
def test_wait_for_is_a_cli_flag() -> None:
    config = _config_from_argv(
        ["--server.base_url", "http://gateway:80", "--server.model_name", "m", "--server.wait_for", "reachable"]
    )
    assert config.server is not None
    assert config.server.wait_for == WaitFor.REACHABLE


# No wait_timeout -> unset, so the client keeps its own 600 s budget.
def test_wait_timeout_defaults_to_unset() -> None:
    assert ModelServerClientConfig(base_url="http://localhost:8000").wait_timeout is None


# wait_timeout: 30 in YAML -> 30.0 seconds.
def test_wait_timeout_accepts_seconds() -> None:
    config = ModelServerClientConfig.model_validate({"base_url": "http://localhost:8000", "wait_timeout": 30})
    assert config.wait_timeout == 30.0


# wait_timeout 0 or -5 -> rejected. A wait that can never pass is a config error; wait_for none is how to skip it.
@pytest.mark.parametrize("value", [0, -5])
def test_wait_timeout_must_be_positive(value: float) -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        ModelServerClientConfig.model_validate({"base_url": "http://localhost:8000", "wait_timeout": value})


# --server.wait_timeout 30 on the command line -> server.wait_timeout is 30.0.
def test_wait_timeout_is_a_cli_flag() -> None:
    config = _config_from_argv(["--server.base_url", "http://localhost:8000", "--server.wait_timeout", "30"])
    assert config.server is not None
    assert config.server.wait_timeout == 30.0
