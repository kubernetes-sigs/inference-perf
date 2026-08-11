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
"""Pytest plumbing for the golden-corpus validation tests.

``--update-goldens`` switches ``test_corpus.py`` from asserting against each
case's ``validation.json`` golden to rewriting it with the current validator
output. Run it via ``pdm run update:goldens`` and review the resulting git
diff; the flag parses only when the pytest args target this directory (or
deeper), which the pdm target does.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Rewrite each corpus case's validation.json with current validator output instead of asserting against it.",
    )
