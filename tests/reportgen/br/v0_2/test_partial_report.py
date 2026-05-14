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
"""Tests for the BR0.2 partial-report loader, validator, and merger."""

import tempfile
from pathlib import Path

import pytest
import yaml

from inference_perf.config import (
    BRV02PartialReportLocalSource,
    BRV02PartialReportSource,
)
from inference_perf.reportgen.br.v0_2 import (
    PartialReportError,
    build_results,
    load_partial_report,
    merge_results,
    validate_partial_report,
)
from inference_perf.reportgen.br.v0_2.schema import BenchmarkReportV02


def _local_source(content: str) -> BRV02PartialReportSource:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    tmp.write(content)
    tmp.close()
    return BRV02PartialReportSource(local=BRV02PartialReportLocalSource(path=tmp.name))


# ---------------------------------------------------------------------------
# load_partial_report
# ---------------------------------------------------------------------------


def test_load_local_yaml_returns_parsed_dict() -> None:
    src = _local_source("run:\n  eid: experiment-1\nscenario:\n  stack: []\n")
    parsed = load_partial_report(src)
    assert parsed == {"run": {"eid": "experiment-1"}, "scenario": {"stack": []}}


def test_load_empty_file_returns_empty_dict() -> None:
    src = _local_source("")
    assert load_partial_report(src) == {}


def test_load_missing_file_raises() -> None:
    src = BRV02PartialReportSource(local=BRV02PartialReportLocalSource(path="/nonexistent/partial.yaml"))
    with pytest.raises(PartialReportError, match="Failed to read"):
        load_partial_report(src)


def test_load_invalid_yaml_raises() -> None:
    src = _local_source("foo: : :\n  bar\n")
    with pytest.raises(PartialReportError, match="not valid YAML/JSON"):
        load_partial_report(src)


def test_load_non_mapping_top_level_raises() -> None:
    src = _local_source("- just\n- a\n- list\n")
    with pytest.raises(PartialReportError, match="must be a mapping"):
        load_partial_report(src)


# ---------------------------------------------------------------------------
# validate_partial_report
# ---------------------------------------------------------------------------


def test_validate_empty_partial_passes() -> None:
    assert validate_partial_report({}) == {}


def test_validate_partial_with_run_passes() -> None:
    partial = {"run": {"eid": "exp", "description": "demo"}}
    assert validate_partial_report(partial) == partial


def test_validate_rejects_non_empty_results() -> None:
    # Any non-null sub-field under ``results`` indicates the user is supplying
    # performance information that inference-perf is supposed to fill — fail
    # the run before any work happens.
    partial: dict[str, object] = {"results": {"request_performance": {"aggregate": {}}}}
    with pytest.raises(PartialReportError, match="must not contain pre-existing performance"):
        validate_partial_report(partial)


def test_validate_allows_empty_results_block() -> None:
    # An explicitly empty results: {} (all sub-fields None) is a no-op partial.
    partial = {"results": {"request_performance": None, "observability": None}}
    assert validate_partial_report(partial) == partial


def test_validate_rejects_results_not_a_mapping() -> None:
    partial = {"results": "oops"}
    with pytest.raises(PartialReportError, match="'results' must be a mapping"):
        validate_partial_report(partial)


def test_validate_rejects_mistyped_run_field() -> None:
    # uid must be a string per the BR0.2 schema; a list should fail.
    partial = {"run": {"uid": ["not", "a", "string"]}}
    with pytest.raises(PartialReportError, match="schema validation"):
        validate_partial_report(partial)


# ---------------------------------------------------------------------------
# merge_results
# ---------------------------------------------------------------------------


def test_merge_results_fills_uid_when_partial_omits_it() -> None:
    results = build_results([])
    merged = merge_results({}, results, run_uid_fallback="fallback-uid")
    assert merged["run"]["uid"] == "fallback-uid"


def test_merge_results_keeps_user_supplied_run_uid() -> None:
    results = build_results([])
    merged = merge_results({"run": {"uid": "user-uid"}}, results, run_uid_fallback="ignored")
    assert merged["run"]["uid"] == "user-uid"


def test_merge_results_preserves_partial_fields() -> None:
    results = build_results([])
    merged = merge_results(
        {"run": {"eid": "exp", "description": "demo"}, "scenario": {"stack": []}},
        results,
        run_uid_fallback="fallback-uid",
    )
    assert merged["run"]["eid"] == "exp"
    assert merged["run"]["description"] == "demo"
    assert merged["scenario"]["stack"] == []


def test_merge_results_produces_schema_valid_report() -> None:
    results = build_results([])
    merged = merge_results({"run": {"eid": "exp"}}, results, run_uid_fallback="uid-1")
    # Round-trip through the vendored schema to confirm a valid BR0.2 doc.
    BenchmarkReportV02.model_validate(merged)


def test_partial_report_source_requires_exactly_one() -> None:
    # Zero set
    with pytest.raises(Exception, match="exactly one"):
        BRV02PartialReportSource()
    # Both set
    from inference_perf.config import BRV02PartialReportGCSSource

    with pytest.raises(Exception, match="exactly one"):
        BRV02PartialReportSource(
            local=BRV02PartialReportLocalSource(path="/x"),
            google_cloud_storage=BRV02PartialReportGCSSource(bucket_name="b", path="p"),
        )


# ---------------------------------------------------------------------------
# end-to-end smoke: load → validate → merge → schema-validate
# ---------------------------------------------------------------------------


def test_end_to_end_local_partial_round_trip() -> None:
    partial_yaml = yaml.safe_dump({"run": {"eid": "exp-1", "description": "smoke"}, "scenario": {"stack": []}})
    src = _local_source(partial_yaml)

    loaded = load_partial_report(src)
    validated = validate_partial_report(loaded)
    results = build_results([])  # empty perf
    merged = merge_results(validated, results, run_uid_fallback="uid-smoke")

    BenchmarkReportV02.model_validate(merged)
    assert merged["run"]["eid"] == "exp-1"
    assert merged["scenario"]["stack"] == []
    assert merged["results"].get("request_performance") is None


def test_end_to_end_rejects_partial_with_perf() -> None:
    src = _local_source(yaml.safe_dump({"results": {"request_performance": {"aggregate": {"requests": {"total": 10}}}}}))
    loaded = load_partial_report(src)
    with pytest.raises(PartialReportError, match="must not contain pre-existing performance"):
        validate_partial_report(loaded)


def _cleanup_tmp(path_str: str) -> None:  # pragma: no cover — best-effort
    try:
        Path(path_str).unlink()
    except OSError:
        pass
