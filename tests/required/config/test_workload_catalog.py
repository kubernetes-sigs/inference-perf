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
"""Unit tests for running a workload catalog entry and the fields its mask locks."""

from pathlib import Path
from typing import List, Mapping, Optional
from unittest.mock import Mock

import pytest
import yaml
from pydantic import BaseModel

from inference_perf.config import Config, read_config
from inference_perf.config.workload_catalog import (
    CATALOG_DIR,
    CONFIG_FILE,
    MASK_FILE,
    WorkloadCatalogError,
    entry_names,
    load_entry,
    mask_problems,
)
from inference_perf.reportgen.base import ReportGenerator

ENTRY = "interactive-chat"


class _Server(BaseModel):
    base_url: str = ""
    ignore_eos: bool = True


class _Small(BaseModel):
    workload: Optional[str] = None
    server: Optional[_Server] = None
    stages: List[int] = []


# Writes one catalog entry into a scratch catalog: its config and, when given, its mask.
def _write_entry(catalog: Path, name: str, config: Mapping[str, object], mask: Optional[Mapping[str, object]]) -> None:
    (catalog / name).mkdir(parents=True)
    (catalog / name / CONFIG_FILE).write_text(yaml.safe_dump(config))
    if mask is not None:
        (catalog / name / MASK_FILE).write_text(yaml.safe_dump(mask))


# The catalog holds the seven entries that shipped before masks existed. Fails if
# the directory moved or the listing came back empty.
def test_catalog_lists_the_shipped_entries() -> None:
    assert ENTRY in entry_names()
    assert "agentic-trace-replay" in entry_names()


# Every directory in the catalog has a config and a mask that marks every field.
# Loading fails on a missing mask, a missing field or an unknown one.
@pytest.mark.parametrize("name", entry_names())
def test_every_entry_has_a_valid_mask(name: str) -> None:
    assert (CATALOG_DIR / name / MASK_FILE).is_file()
    entry = load_entry(name, Config)
    assert entry.locked


# Running each entry with nothing on top builds a config named after the entry.
# Its config.yaml report says workload: <name>.
@pytest.mark.parametrize("name", entry_names())
def test_every_entry_runs_unchanged_and_stamps_the_report(name: str) -> None:
    config = read_config(None, {"workload": name})
    assert config.workload == name
    report = ReportGenerator(metrics_client=None, metrics_collector=Mock(), config=config).generate_config_report()
    contents = report.get_contents()
    assert isinstance(contents, dict)
    assert contents["workload"] == name


# interactive-chat with a different base_url and num_workers: both are open, so the
# run starts with the user's values on top of the entry.
def test_open_fields_take_the_users_values() -> None:
    config = read_config(None, {"workload": ENTRY, "server": {"base_url": "http://other:9000"}, "load": {"num_workers": 3}})
    assert config.server is not None
    assert config.server.base_url == "http://other:9000"
    assert config.load.num_workers == 3


# --api.streaming false on interactive-chat, which locks api.streaming at true.
# The run refuses and names the field with the value the entry requires.
def test_changing_a_locked_field_refuses_the_run() -> None:
    with pytest.raises(WorkloadCatalogError, match=r"api\.streaming must be true"):
        read_config(None, {"workload": ENTRY, "api": {"streaming": False}})


# interactive-chat locks the whole data section. Changing the seed inside it names
# data.conversation_replay.seed and the entry's 42, not just "data".
def test_a_locked_section_names_the_field_that_changed() -> None:
    with pytest.raises(WorkloadCatalogError, match=r"data\.conversation_replay\.seed must be 42"):
        read_config(None, {"workload": ENTRY, "data": {"conversation_replay": {"seed": 7}}})


# Setting a locked field to the value the entry already has (streaming: true, seed 42)
# is not a change, so the run starts.
def test_restating_a_locked_value_is_allowed() -> None:
    config = read_config(None, {"workload": ENTRY, "api": {"streaming": True}, "load": {"base_seed": 42}})
    assert config.api.streaming is True


# A config file holding workload: interactive-chat plus a server loads the entry
# the same way --workload does.
def test_a_config_file_can_name_the_entry(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"workload": ENTRY, "server": {"type": "vllm", "base_url": "http://mine:8000"}}))
    config = read_config(str(path))
    assert config.workload == ENTRY
    assert config.server is not None
    assert config.server.base_url == "http://mine:8000"
    assert config.data.conversation_replay is not None


# The same file run with --api.streaming false still refuses: flags are checked
# against the lock like the file is.
def test_a_flag_on_top_of_a_config_file_is_checked(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"workload": ENTRY}))
    with pytest.raises(WorkloadCatalogError, match=r"api\.streaming"):
        read_config(str(path), {"api": {"streaming": False}})


# A file naming reasoning, run with --workload interactive-chat: the flag wins, as
# flags do over every other field. The data that runs is interactive-chat's, not reasoning's.
def test_the_flag_names_the_entry_over_the_file(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"workload": "reasoning"}))
    config = read_config(str(path), {"workload": ENTRY})
    assert config.workload == ENTRY
    assert config.data == read_config(None, {"workload": ENTRY}).data
    assert config.data != read_config(None, {"workload": "reasoning"}).data


# A file with server: null drops the whole server section, including the locked
# server.ignore_eos. The run refuses and names that field.
def test_dropping_a_section_that_holds_a_locked_field_refuses_the_run(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"workload": ENTRY, "server": None}))
    with pytest.raises(WorkloadCatalogError, match=r"server\.ignore_eos must be true"):
        read_config(str(path))


# A config with no workload runs as before, with workload left unset.
def test_a_run_without_an_entry_has_no_stamp() -> None:
    config = read_config(None, {"data": {"type": "random"}})
    assert config.workload is None


# "no-such-entry" and a path that walks out of the catalog and back into a real entry
# are both refused. The error lists the real entries.
@pytest.mark.parametrize("name", ["no-such-entry", "../workload_catalog/interactive-chat"])
def test_an_unknown_name_lists_the_entries(name: str) -> None:
    with pytest.raises(WorkloadCatalogError, match=ENTRY):
        read_config(None, {"workload": name})


# An entry directory with an inference-perf.yaml and no mask.yaml cannot load.
def test_an_entry_without_a_mask_cannot_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inference_perf.config.workload_catalog.CATALOG_DIR", tmp_path)
    _write_entry(tmp_path, "bare", {"server": {"base_url": "x"}}, mask=None)
    with pytest.raises(WorkloadCatalogError, match="has no mask.yaml"):
        load_entry("bare", _Small)


# An entry whose own config sets workload: other would load a second entry. Refused.
def test_an_entry_cannot_name_another_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inference_perf.config.workload_catalog.CATALOG_DIR", tmp_path)
    _write_entry(tmp_path, "nested", {"workload": "other"}, {"server": "open", "stages": "locked"})
    with pytest.raises(WorkloadCatalogError, match="cannot load another entry"):
        load_entry("nested", _Small)


# An entry whose mask leaves out stages cannot load, and the error names the mask file.
def test_an_entry_with_an_incomplete_mask_cannot_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inference_perf.config.workload_catalog.CATALOG_DIR", tmp_path)
    _write_entry(tmp_path, "partial", {}, {"server": "open"})
    with pytest.raises(
        WorkloadCatalogError, match=r"partial/mask\.yaml is not a valid mask:\n  the mask does not mark stages"
    ):
        load_entry("partial", _Small)


# A mask whose server section is broken into fields locks only the fields marked locked.
# Here: server.ignore_eos and stages, but not server.base_url.
def test_a_split_section_locks_only_its_locked_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inference_perf.config.workload_catalog.CATALOG_DIR", tmp_path)
    mask = {"server": {"base_url": "open", "ignore_eos": "locked"}, "stages": "locked"}
    _write_entry(tmp_path, "split", {}, mask)
    assert load_entry("split", _Small).locked == (("server", "ignore_eos"), ("stages",))


# A mask that marks every field of the small model, skipping workload, has no problems.
def test_a_complete_mask_has_no_problems() -> None:
    mask = {"server": {"base_url": "open", "ignore_eos": "locked"}, "stages": "locked"}
    assert mask_problems(mask, _Small, skip=frozenset({"workload"})) == []


# Leaving out server.ignore_eos is a problem that names it.
def test_a_mask_must_mark_every_field() -> None:
    problems = mask_problems({"server": {"base_url": "open"}, "stages": "locked"}, _Small, skip=frozenset({"workload"}))
    assert problems == ["server does not mark ignore_eos locked or open"]


# Marking server.port, which the model does not have, is a problem that names it.
def test_a_mask_cannot_mark_an_unknown_field() -> None:
    mask = {"server": "open", "stages": "locked", "port": "open"}
    assert mask_problems(mask, _Small, skip=frozenset({"workload"})) == ["port is not a config field a mask can mark"]


# workload is skipped, so a mask that marks it is told the field cannot be marked.
def test_a_mask_cannot_mark_the_workload_field() -> None:
    mask = {"workload": "locked", "server": "open", "stages": "locked"}
    assert mask_problems(mask, _Small, skip=frozenset({"workload"})) == ["workload is not a config field a mask can mark"]


# stages is a list with no fields of its own, so splitting it is a problem.
def test_only_a_section_with_fields_can_be_split() -> None:
    mask = {"server": "open", "stages": {"first": "locked"}}
    problems = mask_problems(mask, _Small, skip=frozenset({"workload"}))
    assert problems == ["stages has no fields of its own. Mark the whole of it locked or open"]


# "frozen" is neither locked nor open, so it is a problem that quotes it.
def test_a_field_must_be_locked_or_open() -> None:
    problems = mask_problems({"server": "frozen", "stages": "locked"}, _Small, skip=frozenset({"workload"}))
    assert problems == ["server must be locked, open or its own fields each marked, not 'frozen'"]


# A mask file holding a list instead of a mapping is one problem about the whole mask.
def test_a_mask_must_be_a_mapping() -> None:
    assert mask_problems(["server"], _Small) == ["the mask must mark each field locked or open, not ['server']"]
