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
"""Repo-wide pytest setup (rootdir conftest, loaded for every test tier)."""

import multiprocessing as mp
import sys

import pytest


# Match the multiprocessing start method used in production (main.py): on
# macOS force "fork" so tokenizer and datagen objects are inherited by worker
# processes instead of pickled. Linux already defaults to fork, so this is a
# no-op there. Set once here, before any test module is imported, instead of
# at import time in individual test modules.
def pytest_configure(config: pytest.Config) -> None:
    if sys.platform == "darwin":
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
