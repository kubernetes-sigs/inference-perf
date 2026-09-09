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
"""Every exported runtime metric has a row in docs/runtime_metrics.md, and that stays true.

The doc is generated from ALL_SPECS, so the tests here check the two things
generation alone cannot: that the generated table covers what a scrape
actually shows, and that the check which enforces it runs on the merge path.
"""

import re
import tomllib
from pathlib import Path
from typing import Any, List

import pytest
from prometheus_client import Counter, Gauge, Histogram  # noqa: TID251 (throwaway metrics for these tests)

from inference_perf.observability.metrics import coverage
from inference_perf.observability.metrics.coverage import (
    coverage_problems,
    documented_metric_names,
    emitted_metric_names,
    specs_declared_under_sets,
    unaggregated_spec_names,
)
from inference_perf.observability.metrics.registry import MetricSpec
from inference_perf.observability.metrics.sets import ALL_SPECS

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_PATH = REPO_ROOT / "docs/runtime_metrics.md"

# The workflow that runs `pdm run validate`, as named in .github/workflows/format.yml
# and listed in the merge gate.
LINT_WORKFLOW = "Python Linting and Type Checks"


# Pulls the metric name out of the leading `| `name` |` cell of every table row in
# docs/runtime_metrics.md. Expects the 20 exported names; the Stability bullet list and
# the prose above the table have no such cell and are skipped.
def documented_names_in_the_file() -> List[str]:
    return re.findall(r"^\| `([a-z_0-9]+)` \|", DOC_PATH.read_text(), re.MULTILINE)


# Scrapes a registry built from the real ALL_SPECS with config gating bypassed, and
# compares the family names against the rows of the checked-in doc. Expects them to
# match exactly, in both directions: no metric a scrape can show is missing a row,
# and no row describes a metric that is not there.
def test_the_doc_lists_exactly_the_metrics_a_scrape_can_show() -> None:
    assert sorted(emitted_metric_names()) == sorted(documented_names_in_the_file())


# Runs the full coverage check over the real specs. Expects no problems: this is the
# same call `pdm run check:runtime-metrics` makes, so a failure here is the failure
# CI would report.
def test_real_metric_set_has_no_coverage_problems() -> None:
    assert coverage_problems() == []


# Counters are exported with a _total suffix that the spec name does not carry, so
# the doc has to use the exposition name. Expects the requests counter, declared as
# inference_perf_requests, to be documented and emitted as inference_perf_requests_total.
def test_counter_rows_use_the_exposition_name_not_the_spec_name() -> None:
    assert "inference_perf_requests_total" in documented_names_in_the_file()
    assert "inference_perf_requests_total" in emitted_metric_names()
    assert "inference_perf_requests" not in emitted_metric_names()


# Simulates a metric that reaches the registry without reaching the doc, by making the
# exposition report one extra name. Expects coverage_problems to name it and to say
# where a spec for it belongs.
def test_metric_exposed_without_a_row_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def one_extra(specs: Any = ALL_SPECS) -> set[str]:
        return documented_metric_names(specs) | {"inference_perf_undeclared_total"}

    monkeypatch.setattr(coverage, "emitted_metric_names", one_extra)

    problems = coverage_problems()
    assert len(problems) == 1
    assert "inference_perf_undeclared_total" in problems[0]
    assert "sets/" in problems[0]


# Simulates a row left behind after its metric was removed, by dropping one name from
# the exposition. Expects coverage_problems to name the stale row.
def test_row_with_no_metric_behind_it_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def one_fewer(specs: Any = ALL_SPECS) -> set[str]:
        return emitted_metric_names(specs) - {"inference_perf_stages"}

    monkeypatch.setattr(coverage, "emitted_metric_names", one_fewer)

    problems = coverage_problems()
    assert len(problems) == 1
    assert "inference_perf_stages" in problems[0]


# Passes ALL_SPECS minus one of its members as the aggregate. Expects the dropped spec
# to be reported as declared under sets/ but missing from ALL_SPECS, which is the
# shape of forgetting to add a new set module to sets/__init__.py.
def test_spec_left_out_of_all_specs_is_reported() -> None:
    dropped = ALL_SPECS[0]
    partial = ALL_SPECS[1:]

    assert unaggregated_spec_names(partial) == [dropped.name]

    problems = coverage_problems(partial)
    assert len(problems) == 1
    assert dropped.name in problems[0]
    assert "sets/__init__.py" in problems[0]


# Walks the module-level specs of every module under sets/. Expects each one to be in
# ALL_SPECS, and expects the walk to actually find specs, so the check cannot pass by
# looking at nothing.
def test_every_spec_declared_under_sets_is_aggregated() -> None:
    declared = specs_declared_under_sets()
    assert declared
    assert unaggregated_spec_names() == []


# Builds a spec whose enabled predicate is off, then bypasses the gate the way the
# coverage check does. Expects the metric to be absent from a normally built registry
# but present in emitted_metric_names, because the doc has to cover metrics no single
# config turns on.
def test_config_gated_metrics_are_still_counted_as_emitted() -> None:
    def never(config: Any) -> bool:
        """Never exported."""
        return False

    gated = MetricSpec(
        name="inference_perf_gated_example",
        documentation="Only under some config.",
        metric_type=Gauge,
        enabled=never,
    )
    plain = MetricSpec(name="inference_perf_plain_example", documentation="Always.", metric_type=Counter)

    assert emitted_metric_names((gated, plain)) == {
        "inference_perf_gated_example",
        "inference_perf_plain_example_total",
    }


# Reads pyproject.toml, .github/workflows/format.yml and .github/workflows/merge_gate.yml.
# Expects the chain that makes the doc check block a merge to be intact: the check is in
# the validate composite, the lint workflow runs validate, and the merge gate holds a PR
# until that workflow is green.
def test_doc_check_is_on_the_merge_blocking_path() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        scripts: dict[str, Any] = tomllib.load(f)["tool"]["pdm"]["scripts"]
    assert "check:runtime-metrics" in scripts["validate"]["composite"], (
        "the runtime metrics doc check must stay in `validate`, which is what CI runs"
    )

    lint_workflow = (REPO_ROOT / ".github/workflows/format.yml").read_text()
    assert f"name: {LINT_WORKFLOW}" in lint_workflow
    assert "pdm run validate" in lint_workflow

    merge_gate = (REPO_ROOT / ".github/workflows/merge_gate.yml").read_text()
    assert merge_gate.count(f"- {LINT_WORKFLOW}") == 1, "merge_gate.yml must wait on the lint workflow"
    assert f"'{LINT_WORKFLOW}'" in merge_gate, "merge_gate.yml must require the lint workflow to have succeeded"


# Constructs a prometheus metric directly, the way a contributor would if the ruff ban
# were not there. Expects the ban to be configured for every constructor and for the
# global registry, since any of them would export a metric with no row in the doc.
def test_prometheus_constructors_are_banned_outside_the_registry() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        lint: dict[str, Any] = tomllib.load(f)["tool"]["ruff"]["lint"]
    assert "TID251" in lint["select"]
    banned = lint["flake8-tidy-imports"]["banned-api"]
    for name in ("Counter", "Gauge", "Histogram", "Summary", "Info", "Enum", "REGISTRY"):
        assert f"prometheus_client.{name}" in banned, f"prometheus_client.{name} must stay banned"


# Builds an unlabelled counter and an unlabelled histogram, which prometheus_client
# exports with an extra `_created` gauge apiece. Expects only the two real family
# names, since a `_created` series belongs to the metric it comes from and gets no
# row of its own; counting it would report a metric that does not exist.
def test_derived_created_series_are_not_counted_as_metrics() -> None:
    counter = MetricSpec(name="inference_perf_unlabelled_example", documentation="No labels.", metric_type=Counter)
    histogram = MetricSpec(
        name="inference_perf_unlabelled_example_seconds",
        documentation="No labels.",
        metric_type=Histogram,
        buckets=(0.1, 1.0),
    )

    assert emitted_metric_names((counter, histogram)) == {
        "inference_perf_unlabelled_example_total",
        "inference_perf_unlabelled_example_seconds",
    }
