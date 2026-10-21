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
"""Pins e2e/known_bad.py, which routes the floating vLLM legs around broken releases."""

from http.client import IncompleteRead
from pathlib import Path
from typing import NoReturn

import known_bad
import pytest
from known_bad import KNOWN_BAD, load_known_bad, main, resolved_version, skip_reason
from utils.net import get_free_port

KNOWN = {"0.32.0": "vllm#61131: dies on its first request"}


# A table with a comment, a blank line, and one entry. Loads to {"0.32.0": reason}
# with the reason's surrounding whitespace dropped.
def test_load_known_bad_parses_comments_blanks_and_entries(tmp_path: Path) -> None:
    table = tmp_path / "known_bad.txt"
    table.write_text("# header\n\n0.32.0   vllm#61131: dies on its first request  \n")
    assert load_known_bad(table) == KNOWN


# A tab between the version and a multi-word reason loads under the bare version
# 0.32.0, not under "0.32.0<tab>vllm#61131:" where nothing would ever match it.
def test_load_known_bad_accepts_a_tab_separator(tmp_path: Path) -> None:
    table = tmp_path / "known_bad.txt"
    table.write_text("0.32.0\tvllm#61131: dies on its first request\n")
    assert load_known_bad(table) == KNOWN


# A version with no reason, or a version listed twice, raises naming the file,
# line, and version.
@pytest.mark.parametrize(
    "text, error",
    [
        ("0.32.0\n", r"known_bad.txt:1: 0.32.0 has no reason"),
        ("0.32.0 first\n0.32.0 second\n", r"known_bad.txt:2: 0.32.0 is listed twice"),
    ],
)
def test_load_known_bad_rejects_a_malformed_table(tmp_path: Path, text: str, error: str) -> None:
    table = tmp_path / "known_bad.txt"
    table.write_text(text)
    with pytest.raises(ValueError, match=error):
        load_known_bad(table)


# latest resolved to the listed 0.32.0 is skipped with its reason. latest at an
# unlisted 0.33.0, latest with no version, and the pins v0.32.0 and v0.32.0-arm64
# at 0.32.0 all run.
@pytest.mark.parametrize(
    "tag, version, expected",
    [
        ("latest", "0.32.0", KNOWN["0.32.0"]),
        ("latest", "0.33.0", None),
        ("latest", "", None),
        ("v0.32.0", "0.32.0", None),
        ("v0.32.0-arm64", "0.32.0", None),
    ],
)
def test_skip_reason_only_for_floating_tags_at_listed_versions(tag: str, version: str, expected: str | None) -> None:
    assert skip_reason(tag, version, KNOWN) == expected


# Nothing listening on the port resolves to None rather than raising, so the caller
# runs the pass instead of skipping it.
def test_resolved_version_is_none_without_a_server() -> None:
    assert resolved_version(f"http://127.0.0.1:{get_free_port()}", timeout=1.0) is None


# A server that cuts its /version reply short raises IncompleteRead, which is not
# an OSError. Resolves to None like any other unanswered request.
def test_resolved_version_is_none_on_a_truncated_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    def truncated(url: str, timeout: float) -> NoReturn:
        raise IncompleteRead(b'{"vers')

    monkeypatch.setattr(known_bad, "urlopen", truncated)
    assert resolved_version("http://127.0.0.1:8000") is None


# With 0.32.0 listed, skip-reason prints the reason for latest at 0.32.0 and an
# empty line for latest at 0.33.0, and version prints "unknown" with no server.
# All exit 0, so the workflow reads any other exit as an error.
@pytest.mark.parametrize(
    "argv, printed",
    [
        (["skip-reason", "latest", "0.32.0"], KNOWN["0.32.0"]),
        (["skip-reason", "latest", "0.33.0"], ""),
        (["version", "http://127.0.0.1:{port}"], "unknown"),
    ],
)
def test_main_exits_0_with_an_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str], printed: str
) -> None:
    table = tmp_path / "known_bad.txt"
    table.write_text("0.32.0 vllm#61131: dies on its first request\n")
    monkeypatch.setattr(known_bad, "KNOWN_BAD", table)
    assert main([arg.format(port=get_free_port()) for arg in argv]) == 0
    assert capsys.readouterr().out == printed + "\n"


# A malformed table makes skip-reason raise instead of returning, so the process
# exits non-zero and the workflow fails the pass rather than running it.
def test_main_raises_on_a_malformed_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    table = tmp_path / "known_bad.txt"
    table.write_text("0.32.0\n")
    monkeypatch.setattr(known_bad, "KNOWN_BAD", table)
    with pytest.raises(ValueError):
        main(["skip-reason", "latest", "0.32.0"])


# The committed table loads, so a malformed entry fails here before the live loop
# reads it.
def test_committed_table_loads() -> None:
    assert load_known_bad(KNOWN_BAD) is not None
