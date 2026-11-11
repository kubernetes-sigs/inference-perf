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
"""Tests for loading a BR0.2 base report and building partials on top of it."""

from pathlib import Path
from typing import Dict

import pytest
import yaml

from inference_perf.config.reportgen.config import BenchmarkReportConfig, ReportConfig
from inference_perf.reportgen.br.v0_2 import apply_base_report, load_base_report

from .base_report_helpers import at, generate_reports, partial_validation_errors, partials

STACK_COMPONENT: Dict[str, object] = {
    "metadata": {"label": "vllm-0", "cfg_id": "vllm"},
    "standardized": {"kind": "generic", "tool": "vllm", "tool_version": "0.11.0"},
    "native": {"args": {"--max-num-seqs": 256}},
}


# Writes doc as YAML to tmp_path/base.yaml and returns the path.
def _write(tmp_path: Path, doc: object) -> str:
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(doc))
    return str(path)


# run.description, run.keywords and one scenario.stack component are all fillable,
# so the loader returns the document as written.
def test_load_returns_a_fillable_base(tmp_path: Path) -> None:
    doc = {"run": {"description": "nightly", "keywords": ["a", "b"]}, "scenario": {"stack": [STACK_COMPONENT]}}
    assert load_base_report(_write(tmp_path, doc)) == doc


# run.eid is DEFAULTED: inference-perf generates one, and a base may set its own.
def test_load_accepts_a_defaulted_leaf(tmp_path: Path) -> None:
    assert load_base_report(_write(tmp_path, {"run": {"eid": "sweep-7"}})) == {"run": {"eid": "sweep-7"}}


# run.uid is WRITTEN, so a base that sets it is rejected and the error names it.
def test_load_rejects_a_written_leaf(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"owns: run\.uid"):
        load_base_report(_write(tmp_path, {"run": {"uid": "mine"}}))


# requests.incomplete is RESERVED: inference-perf does not write it yet, and a base still may not.
def test_load_rejects_a_reserved_leaf(tmp_path: Path) -> None:
    doc = {"results": {"request_performance": {"aggregate": {"requests": {"incomplete": 0}}}}}
    with pytest.raises(ValueError, match=r"owns: results\.request_performance\.aggregate\.requests\.incomplete$"):
        load_base_report(_write(tmp_path, doc))


# run.uid and run.time.start are both WRITTEN. One error names both, so a base
# gets fixed in one pass.
def test_load_names_every_owned_leaf_at_once(tmp_path: Path) -> None:
    doc = {"run": {"uid": "mine", "time": {"start": "2026-11-06T10:00:00+00:00"}}}
    with pytest.raises(ValueError, match=r"owns: run\.time\.start, run\.uid$"):
        load_base_report(_write(tmp_path, doc))


# run.descripton is not a BR0.2 field, so the typo is an error naming the path.
def test_load_rejects_an_unknown_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"not BR0\.2 fields: run\.descripton"):
        load_base_report(_write(tmp_path, {"run": {"descripton": "typo"}}))


# "description:" with no value loads as null. The emitted partial never carries
# nulls, so the loader asks for the key to be dropped.
def test_load_rejects_a_null(tmp_path: Path) -> None:
    path = tmp_path / "base.yaml"
    path.write_text("run:\n  description:\n")
    with pytest.raises(ValueError, match=r"null values at run\.description"):
        load_base_report(str(path))


# A YAML list is not a report.
def test_load_rejects_a_non_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a YAML mapping, got list"):
        load_base_report(_write(tmp_path, ["run"]))


# A missing file is a ValueError, so config load reports it like any other bad value.
def test_load_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read base report"):
        load_base_report(str(tmp_path / "absent.yaml"))


# An unclosed flow sequence is not YAML.
def test_load_rejects_unparseable_yaml(tmp_path: Path) -> None:
    path = tmp_path / "base.yaml"
    path.write_text("run: [\n")
    with pytest.raises(ValueError, match="cannot read base report"):
        load_base_report(str(path))


# A stack component without metadata.label fails BR0.2 validation.
def test_load_rejects_a_schema_invalid_base(tmp_path: Path) -> None:
    component = {**STACK_COMPONENT, "metadata": {"cfg_id": "vllm"}}
    with pytest.raises(ValueError, match="is not valid BR0.2"):
        load_base_report(_write(tmp_path, {"scenario": {"stack": [component]}}))


# A pod startup aggregate with min 5 > max 1 passes the schema but would show up
# as an inference-perf error in validation.json, so the loader rejects it.
def test_load_rejects_an_inconsistent_distribution(tmp_path: Path) -> None:
    aggregate = {"units": "s", "mean": 3.0, "min": 5.0, "max": 1.0}
    doc = {"results": {"observability": {"pod_startup_times": {"aggregate": aggregate}}}}
    with pytest.raises(ValueError, match="inconsistent distributions"):
        load_base_report(_write(tmp_path, doc))


# A negative startup time is an error too, the same as in the emitted partial.
def test_load_rejects_a_negative_distribution(tmp_path: Path) -> None:
    aggregate = {"units": "s", "mean": -1.0, "min": -1.0, "max": -1.0}
    doc = {"results": {"observability": {"pod_startup_times": {"aggregate": aggregate}}}}
    with pytest.raises(ValueError, match="inconsistent distributions"):
        load_base_report(_write(tmp_path, doc))


# Unquoted YAML timestamps, at a map key and inside a list item, come back as
# ISO-8601 strings, the form inference-perf writes its own run.time in.
def test_load_turns_timestamps_into_iso_strings(tmp_path: Path) -> None:
    path = tmp_path / "base.yaml"
    path.write_text(
        "results:\n"
        "  observability:\n"
        "    pod_startup_times:\n"
        "      collected_at: 2026-11-06T10:00:00Z\n"
        "      pods:\n"
        "        - name: vllm-0\n"
        "          ready_timestamp: 2026-11-06T09:58:30Z\n"
    )
    startup = at(load_base_report(str(path)), "results", "observability", "pod_startup_times")
    assert at(startup, "collected_at") == "2026-11-06T10:00:00+00:00"
    assert at(startup, "pods", 0, "ready_timestamp") == "2026-11-06T09:58:30+00:00"


# The partial keeps run.uid and results. The base adds run.description and the stack.
def test_apply_merges_maps_key_by_key() -> None:
    partial = {"version": "0.2.1", "run": {"uid": "u", "eid": "e"}, "results": {"request_performance": {"x": 1}}}
    base = {"run": {"description": "d"}, "scenario": {"stack": [STACK_COMPONENT]}}
    assert apply_base_report(partial, base) == {
        "version": "0.2.1",
        "run": {"uid": "u", "eid": "e", "description": "d"},
        "results": {"request_performance": {"x": 1}},
        "scenario": {"stack": [STACK_COMPONENT]},
    }


# Both set run.eid, the one leaf they can share. The base's eid wins.
def test_apply_keeps_the_base_eid() -> None:
    merged = apply_base_report({"run": {"uid": "u", "eid": "generated"}}, {"run": {"eid": "sweep-7"}})
    assert merged["run"] == {"uid": "u", "eid": "sweep-7"}


# Every stage partial gets the same base. Editing one merged stack must not
# reach the base or the next stage.
def test_apply_copies_base_values() -> None:
    base = {"scenario": {"stack": [{"metadata": {"label": "a", "cfg_id": "c"}}]}}
    merged = apply_base_report({}, base)
    metadata = at(merged, "scenario", "stack", 0, "metadata")
    assert isinstance(metadata, dict)
    metadata["label"] = "changed"
    assert base["scenario"]["stack"][0]["metadata"]["label"] == "a"


# Merging builds new maps. The partial passed in is left as it was.
def test_apply_leaves_the_partial_unchanged() -> None:
    partial = {"run": {"uid": "u"}}
    apply_base_report(partial, {"run": {"description": "d"}})
    assert partial == {"run": {"uid": "u"}}


# A base that loaded at config time but has since been edited to set run.uid
# no longer loads. Both stages still emit their bare partials, without the base.
def test_generate_reports_drops_a_base_that_broke_mid_run(tmp_path: Path) -> None:
    path = _write(tmp_path, {"run": {"description": "nightly"}})
    report_config = ReportConfig(prometheus=None, benchmark_report=BenchmarkReportConfig(base_file=path))
    _write(tmp_path, {"run": {"uid": "mine"}})

    reports = generate_reports(report_config)

    stages = partials(reports)
    assert sorted(stages) == [0, 1]
    for partial in stages.values():
        run = at(partial, "run")
        assert isinstance(run, dict) and "description" not in run
        assert run["uid"].startswith("inference-perf-stage-")
    assert partial_validation_errors(reports) == []
