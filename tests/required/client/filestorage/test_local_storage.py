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
"""Tests for LocalStorageClient's file writes.

Since #817 each stage's reports are saved when the stage ends, and the
end-of-run pass saves the same files again. A process that dies while that
second write is in progress must not leave a file it had already written
complete truncated or half-written, so the client writes each report aside and
renames it into place. These tests pin that, and that the files it leaves
behind look the same as before: same contents, no stray temporary files, and
the permissions a plain open() would give them.
"""

import json
import os
import stat
from pathlib import Path
from typing import Iterator, NoReturn, SupportsIndex

import pytest
import yaml

from inference_perf.client.filestorage.local import LocalStorageClient
from inference_perf.config import StorageConfigBase
from inference_perf.utils import ReportFile


def _client(path: Path) -> LocalStorageClient:
    return LocalStorageClient(StorageConfigBase(path=str(path)))


def _load(path: Path, file_type: str) -> object:
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if file_type == "json" else yaml.safe_load(text)


class _Unserializable:
    """A value neither json.dumps nor yaml.dump can serialize; both raise TypeError."""

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("cannot serialize this value")


@pytest.fixture
def umask() -> Iterator[None]:
    """Restores the process umask a test changes."""
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


@pytest.mark.parametrize("file_type", ["json", "yaml"])
def test_saving_a_report_again_replaces_its_contents(tmp_path: Path, file_type: str) -> None:
    """The end-of-run save of a file written at stage end leaves the end-of-run contents.

    Unchanged-behaviour pin: LocalStorageClient has always overwritten.
    """
    client = _client(tmp_path)
    client.save_report([ReportFile(name="stage_0_lifecycle_metrics", contents={"count": 1}, file_type=file_type)])
    client.save_report([ReportFile(name="stage_0_lifecycle_metrics", contents={"count": 2}, file_type=file_type)])

    assert _load(tmp_path / f"stage_0_lifecycle_metrics.{file_type}", file_type) == {"count": 2}


def test_saving_leaves_only_the_report_files_behind(tmp_path: Path) -> None:
    """Writing aside must not leave its temporary files in the report directory.

    Tools that read a report directory (analyze, downstream composers) would
    otherwise find files no inference-perf run is supposed to produce.
    Unchanged-behaviour pin.
    """
    client = _client(tmp_path)
    reports = [
        ReportFile(name="stage_0_lifecycle_metrics", contents={"count": 1}),
        ReportFile(name="inference-perf.partial.stage_0", contents={"run": {"uid": "a"}}, file_type="yaml"),
    ]
    client.save_report(reports)
    client.save_report(reports)

    assert sorted(os.listdir(tmp_path)) == ["inference-perf.partial.stage_0.yaml", "stage_0_lifecycle_metrics.json"]


@pytest.mark.usefixtures("umask")
@pytest.mark.parametrize("mask", [0o022, 0o077, 0o002])
def test_file_mode_follows_the_umask_like_a_plain_open(tmp_path: Path, mask: int) -> None:
    """A report file gets the permissions open(path, "w") would give it under the current umask.

    Temporary-file helpers such as tempfile.mkstemp create files 0o600 whatever
    the umask, which would hide reports from the other users and containers
    that read a shared report directory. Unchanged-behaviour pin.
    """
    os.umask(mask)
    reference = tmp_path / "reference"
    with open(reference, "w", encoding="utf-8"):
        pass
    out_dir = tmp_path / "out"
    _client(out_dir).save_report([ReportFile(name="config", contents={"load": {}}, file_type="yaml")])

    mode = stat.S_IMODE((out_dir / "config.yaml").stat().st_mode)
    assert mode == stat.S_IMODE(reference.stat().st_mode)
    assert mode == 0o666 & ~mask


@pytest.mark.parametrize("file_type", ["json", "yaml"])
def test_failed_write_leaves_the_previous_complete_file_intact(tmp_path: Path, file_type: str) -> None:
    """A save that fails partway leaves the copy written earlier as it was.

    A stage's files are written when the stage ends and again at the end of the
    run; if that second write dies (here a value that can't be serialized, in
    production an OOM kill or a full disk), the stage's file must still be the
    complete one from stage end, not a truncated one. Writing in place truncates
    the file before the new contents are ready.
    """
    client = _client(tmp_path)
    name = "stage_0_lifecycle_metrics"
    client.save_report([ReportFile(name=name, contents={"count": 4}, file_type=file_type)])

    with pytest.raises(TypeError):
        client.save_report([ReportFile(name=name, contents={"count": _Unserializable()}, file_type=file_type)])

    assert _load(tmp_path / f"{name}.{file_type}", file_type) == {"count": 4}
    # Nor does it leave its half-written temporary file behind.
    assert [path.name for path in tmp_path.iterdir()] == [f"{name}.{file_type}"]
