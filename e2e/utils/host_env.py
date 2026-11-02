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
"""Environment for a tool the e2e tests spawn under a Python other than their own."""

import os
from typing import Dict

# Environment variables that tell a Python process where to look for packages.
# These tests usually run inside this repo's dev environment (nix devshell plus
# venv), which sets them for its own Python. guidellm, `vllm serve` and
# `vllm bench serve` run under a different Python, and PYTHONPATH wins over
# that Python's own installed packages, so an inherited value makes them import
# this repo's torch (built for a different Python) and crash on startup.
# LD_LIBRARY_PATH is kept: a Python built against a shared libpython can need
# it to find that library.
HOST_PYTHON_ENV_VARS = frozenset(
    {
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "NIX_PYTHONPATH",
        "VIRTUAL_ENV",
        "VIRTUAL_ENV_PROMPT",
    }
)


def isolated_env() -> Dict[str, str]:
    """A copy of os.environ with HOST_PYTHON_ENV_VARS removed."""
    return {k: v for k, v in os.environ.items() if k not in HOST_PYTHON_ENV_VARS}
