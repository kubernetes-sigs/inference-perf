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
"""The argv scrubbing behind the emitted comment block."""

from inference_perf.tools.convert.model import REDACTED, redact_values, rewrite_values


def test_redact_values_covers_both_spellings() -> None:
    argv = ["--api-key", "sk-1", "--model", "m", "--api-key=sk-2", "--api-key"]
    assert redact_values(argv, frozenset({"--api-key"})) == [
        "--api-key",
        REDACTED,
        "--model",
        "m",
        f"--api-key={REDACTED}",
        "--api-key",  # a trailing flag with no value is copied as is
    ]


def test_rewrite_values_passes_the_value_to_the_rewriter() -> None:
    argv = ["-H", "a: 1", "--other", "x", "-H=b: 2"]
    assert rewrite_values(argv, {"-H": str.upper}) == ["-H", "A: 1", "--other", "x", "-H=B: 2"]
