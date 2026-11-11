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
"""Load a user-supplied base report and build each stage's partial on top of it.

A base report is a partial BR0.2 document carrying what inference-perf cannot
measure: run metadata, the stack, cluster state. It may set any leaf
``fillability`` marks FILLABLE or DEFAULTED. Setting one inference-perf writes
or reserves is an error, so a base report can never stand in for a measurement.
"""

from __future__ import annotations

import copy
import datetime
from typing import Dict, List, Mapping

import yaml
from pydantic import ValidationError

from inference_perf.reportgen.validation import Severity, distribution_findings

from .fillability import document_leaves, fillability
from .partial_report import build_partial_report
from .schema import BenchmarkReportV021


def load_base_report(path: str) -> Dict[str, object]:
    """Read a base report and check it. Raises ValueError naming what is wrong."""
    try:
        with open(path) as f:
            document = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise ValueError(f"cannot read base report {path}: {e}") from e
    if not isinstance(document, dict):
        raise ValueError(f"base report {path} must be a YAML mapping, got {type(document).__name__}")

    nulls = _null_paths(document)
    if nulls:
        raise ValueError(f"base report {path} has null values at {', '.join(nulls)}, drop those keys instead")

    try:
        leaves = document_leaves(document)
    except ValueError as e:
        raise ValueError(f"base report {path}: {e}") from e
    owned = [leaf for leaf in leaves if not fillability(leaf).base_may_set]
    if owned:
        raise ValueError(f"base report {path} sets fields inference-perf owns: {', '.join(owned)}")

    _iso_datetimes(document)
    # The base alone is not a valid report: run.uid and results are required.
    # inference-perf's own minimum fills those in, as every stage's partial will.
    try:
        BenchmarkReportV021.model_validate(apply_base_report(build_partial_report([], None, run_uid="base"), document))
    except ValidationError as e:
        raise ValueError(f"base report {path} is not valid BR0.2: {e}") from e

    # The emitted partial is held to the same distribution checks in validation.json,
    # where an error is read as an inference-perf bug.
    errors = [
        f.message
        for f in distribution_findings(path, document, "base_report", negative_ok_substrings=())
        if f.severity == Severity.ERROR
    ]
    if errors:
        raise ValueError(f"base report {path} has inconsistent distributions: {', '.join(errors)}")
    return document


def apply_base_report(partial: Mapping[str, object], base: Mapping[str, object]) -> Dict[str, object]:
    """``partial`` with ``base`` merged over it, map by map."""
    # load_base_report rejects every leaf inference-perf writes, so the two only
    # overlap on DEFAULTED leaves, where the base wins.
    merged = dict(partial)
    for key, value in base.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = apply_base_report(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _null_paths(node: object, path: str = "") -> List[str]:
    if node is None:
        return [path]
    if isinstance(node, dict):
        return [p for key, value in node.items() for p in _null_paths(value, f"{path}.{key}" if path else str(key))]
    if isinstance(node, list):
        return [p for i, item in enumerate(node) for p in _null_paths(item, f"{path}[{i}]")]
    return []


# YAML loads an unquoted timestamp as a datetime. The partial keeps every
# datetime as an ISO-8601 string, the same as the fields inference-perf writes.
def _iso_datetimes(node: object) -> None:
    if isinstance(node, dict):
        entries = list(node.items())
    elif isinstance(node, list):
        entries = list(enumerate(node))
    else:
        return
    for key, value in entries:
        if isinstance(value, datetime.date):
            node[key] = value.isoformat()
        else:
            _iso_datetimes(value)
