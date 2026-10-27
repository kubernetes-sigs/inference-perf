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
"""The vllm version check and command choice behind the parity tests, run against fake vllm installs."""

import logging
from pathlib import Path
from typing import Optional

import pytest

from utils.vllm_bench import (
    VLLM_PINNED_REF,
    VllmUnavailable,
    _bench_serve_cmd,
    check_vllm_version,
    installed_vllm_version,
    vllm_required,
)

PINNED = VLLM_PINNED_REF.removeprefix("v")


# Fake vllm install: tmp_path/bin/vllm, plus a sibling `python` shell script
# with body `python_body` unless it is None. _fake_vllm(tmp_path, "echo 0.10.0")
# gives a vllm whose python prints 0.10.0 for any arguments.
def _fake_vllm(tmp_path: Path, python_body: Optional[str]) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    vllm = bin_dir / "vllm"
    vllm.write_text("#!/bin/sh\nexit 0\n")
    vllm.chmod(0o755)
    if python_body is not None:
        python = bin_dir / "python"
        python.write_text(f"#!/bin/sh\n{python_body}\n")
        python.chmod(0o755)
    return str(vllm)


# VLLM_REQUIRED unset, "" and "0" mean not required. "1" means required.
@pytest.mark.parametrize("value, required", [(None, False), ("", False), ("0", False), ("1", True)])
def test_vllm_required_reads_env(monkeypatch: pytest.MonkeyPatch, value: Optional[str], required: bool) -> None:
    if value is None:
        monkeypatch.delenv("VLLM_REQUIRED", raising=False)
    else:
        monkeypatch.setenv("VLLM_REQUIRED", value)
    assert vllm_required() is required


# Sibling python prints "0.10.0\n". The version read back is "0.10.0".
def test_version_read_from_sibling_python(tmp_path: Path) -> None:
    assert installed_vllm_version(_fake_vllm(tmp_path, "echo 0.10.0")) == "0.10.0"


# No python next to vllm (a wrapper script install). The version is None.
def test_version_unreadable_without_sibling_python(tmp_path: Path) -> None:
    assert installed_vllm_version(_fake_vllm(tmp_path, None)) is None


# Sibling python prints a traceback and exits 1 (vllm not installed in it).
# The version is None, not the traceback text.
def test_version_unreadable_when_python_fails(tmp_path: Path) -> None:
    assert installed_vllm_version(_fake_vllm(tmp_path, "echo Traceback; exit 1")) is None


# The pinned version passes with VLLM_REQUIRED=1 and logs no warning.
def test_pinned_version_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("VLLM_REQUIRED", "1")
    with caplog.at_level(logging.WARNING):
        check_vllm_version(_fake_vllm(tmp_path, f"echo {PINNED}"))
    assert not caplog.records


# The pinned version with a local build label (0.10.0+cpu) still counts as
# pinned, even with VLLM_REQUIRED=1.
def test_local_build_label_counts_as_pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_REQUIRED", "1")
    check_vllm_version(_fake_vllm(tmp_path, f"echo {PINNED}+cpu"))


# vllm 99.0.0 without VLLM_REQUIRED logs a warning naming 99.0.0 and goes on.
def test_other_version_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.delenv("VLLM_REQUIRED", raising=False)
    with caplog.at_level(logging.WARNING):
        check_vllm_version(_fake_vllm(tmp_path, "echo 99.0.0"))
    assert "99.0.0" in caplog.text


# vllm 99.0.0 with VLLM_REQUIRED=1 fails, naming 99.0.0.
def test_other_version_fails_when_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_REQUIRED", "1")
    with pytest.raises(VllmUnavailable, match="99.0.0"):
        check_vllm_version(_fake_vllm(tmp_path, "echo 99.0.0"))


# No sibling python, so no version. Without VLLM_REQUIRED that logs a
# warning and goes on.
def test_unreadable_version_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.delenv("VLLM_REQUIRED", raising=False)
    with caplog.at_level(logging.WARNING):
        check_vllm_version(_fake_vllm(tmp_path, None))
    assert "cannot read the vllm version" in caplog.text


# No sibling python, so no version. With VLLM_REQUIRED=1 that fails.
def test_unreadable_version_fails_when_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_REQUIRED", "1")
    with pytest.raises(VllmUnavailable, match="cannot read the vllm version"):
        check_vllm_version(_fake_vllm(tmp_path, None))


# Sibling python prints the pinned version, so it has vllm. The command runs
# the bench-serve shim under that python, then the args.
def test_bench_cmd_uses_shim_when_sibling_python_has_vllm(tmp_path: Path) -> None:
    cmd = _bench_serve_cmd(_fake_vllm(tmp_path, f"echo {PINNED}"), ["--num-prompts", "4"])
    assert Path(cmd[0]) == (tmp_path / "bin").resolve() / "python"
    assert cmd[1] == "-c"
    assert cmd[-2:] == ["--num-prompts", "4"]


# Sibling python exits 1 on the version probe: an unrelated python next to the
# vllm executable. The command falls back to `vllm bench serve --num-prompts 4`.
def test_bench_cmd_uses_cli_when_sibling_python_lacks_vllm(tmp_path: Path) -> None:
    vllm = _fake_vllm(tmp_path, "exit 1")
    assert _bench_serve_cmd(vllm, ["--num-prompts", "4"]) == [vllm, "bench", "serve", "--num-prompts", "4"]


# No python next to vllm. The command is `vllm bench serve --num-prompts 4`.
def test_bench_cmd_uses_cli_without_sibling_python(tmp_path: Path) -> None:
    vllm = _fake_vllm(tmp_path, None)
    assert _bench_serve_cmd(vllm, ["--num-prompts", "4"]) == [vllm, "bench", "serve", "--num-prompts", "4"]


# link/vllm is a symlink to bin/vllm, a fake venv whose python prints the pin,
# and link/python is an unrelated python that exits 1 (the pipx layout). The
# version is read from bin/python, and the shim runs under it.
def test_symlinked_vllm_uses_the_python_it_points_at(tmp_path: Path) -> None:
    target = _fake_vllm(tmp_path, f"echo {PINNED}")
    link_dir = tmp_path / "link"
    link_dir.mkdir()
    (link_dir / "vllm").symlink_to(target)
    unrelated = link_dir / "python"
    unrelated.write_text("#!/bin/sh\nexit 1\n")
    unrelated.chmod(0o755)
    link = str(link_dir / "vllm")
    assert installed_vllm_version(link) == PINNED
    assert Path(_bench_serve_cmd(link, [])[0]) == (tmp_path / "bin").resolve() / "python"
