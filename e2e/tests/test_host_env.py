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
"""The environment the e2e tests hand to guidellm, vllm and vllm bench."""

import pytest

from utils.host_env import isolated_env

PYTHON_PATH_VARS = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "NIX_PYTHONPATH", "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT")


# All six python-path variables set to /host, plus LD_LIBRARY_PATH=/nix/lib.
# isolated_env drops the six and keeps LD_LIBRARY_PATH as is.
def test_isolated_env_drops_python_path_vars_only(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PYTHON_PATH_VARS:
        monkeypatch.setenv(name, "/host")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/nix/lib")
    env = isolated_env()
    assert not set(PYTHON_PATH_VARS) & env.keys()
    assert env["LD_LIBRARY_PATH"] == "/nix/lib"
