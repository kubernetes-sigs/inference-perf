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
"""Partial BR0.2 report loading, validation, and merging.

A *partial* BR0.2 report is a user-supplied YAML/JSON document containing
the parts of a BR0.2 report inference-perf cannot derive from the run
itself: stack configuration (gpu_count, replica_count, model server
versions, …), run metadata (eid, description, keywords, user, …), and
workload metadata the user wants to record.

The partial must NOT contain the performance ``results`` — that section is
filled exclusively by inference-perf from the actual run. Loading a partial
whose ``results`` section is non-empty fails the run before any work is
done, as does an unparseable file or a schema-invalid partial (extra/
mistyped fields).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

import yaml

from inference_perf.config import (
    BRV02PartialReportGCSSource,
    BRV02PartialReportLocalSource,
    BRV02PartialReportSource,
)
from .schema_v0_2 import BenchmarkReportV02, Results

logger = logging.getLogger(__name__)


class PartialReportError(ValueError):
    """Raised when a partial BR0.2 report fails to load or validate."""


def load_partial_report(source: BRV02PartialReportSource) -> Dict[str, Any]:
    """Fetch and parse a partial BR0.2 report from the configured source.

    Returns the raw parsed dict; schema-level validation happens in
    ``validate_partial_report``. Supports YAML or JSON content; the parser
    accepts either.
    """
    if source.local is not None:
        raw = _read_local(source.local)
    elif source.google_cloud_storage is not None:
        raw = _read_gcs(source.google_cloud_storage)
    else:  # pragma: no cover — guarded by model_validator on the config type
        raise PartialReportError("BRV02PartialReportSource: no source set")

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise PartialReportError(f"BR0.2 partial report is not valid YAML/JSON: {e}") from e

    if parsed is None:
        # Empty document → empty partial. Allowed per spec.
        return {}
    if not isinstance(parsed, dict):
        raise PartialReportError(f"BR0.2 partial report must be a mapping at the top level, got {type(parsed).__name__}")
    return parsed


def _read_local(source: BRV02PartialReportLocalSource) -> str:
    try:
        with open(source.path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise PartialReportError(f"Failed to read BR0.2 partial report from {source.path!r}: {e}") from e


def _read_gcs(source: BRV02PartialReportGCSSource) -> str:
    # ADC auth, matching how the rest of inference-perf talks to GCS.
    import google.cloud.storage as storage  # local import — GCS is an optional dep
    from google.cloud.exceptions import GoogleCloudError

    try:
        client = storage.Client()
        bucket = client.lookup_bucket(source.bucket_name)
        if bucket is None:
            raise PartialReportError(f"GCS bucket '{source.bucket_name}' does not exist or is inaccessible")
        blob = bucket.blob(source.path)
        if not blob.exists():
            raise PartialReportError(f"Partial report object gs://{source.bucket_name}/{source.path} does not exist")
        text: str = blob.download_as_text()
        return text
    except GoogleCloudError as e:
        raise PartialReportError(f"Failed to fetch gs://{source.bucket_name}/{source.path}: {e}") from e


def validate_partial_report(partial: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a parsed partial BR0.2 report.

    Three checks, in order:
      1. The ``results`` section must be absent or contain no non-null
         sub-field. ``results`` is inference-perf-owned (perf measurements +
         collected metrics); any pre-existing performance information
         indicates a misconfiguration and fails the run.
      2. The whole document must conform to the BR0.2 schema with ``run``
         and ``results`` made optional. Extra and mistyped fields are
         rejected by the vendored schema's ``extra="forbid"`` config.

    Returns the original dict on success (for caller convenience). Raises
    ``PartialReportError`` on any failure.
    """
    results = partial.get("results")
    if results is not None:
        if not isinstance(results, dict):
            raise PartialReportError(
                f"BR0.2 partial report: 'results' must be a mapping or absent, got {type(results).__name__}"
            )
        non_empty = [k for k, v in results.items() if v is not None]
        if non_empty:
            raise PartialReportError(
                "BR0.2 partial report must not contain pre-existing performance "
                f"information; found non-empty 'results' fields: {sorted(non_empty)}. "
                "inference-perf is responsible for filling the entire 'results' "
                "section from the actual run."
            )

    # Structurally validate by tentatively merging an empty Results + a
    # placeholder run uid and round-tripping through the BR0.2 schema. This
    # catches mistyped/extra fields anywhere in the partial without requiring
    # a separate "partial" schema definition.
    probe: Dict[str, Any] = json.loads(json.dumps(partial))
    probe_run = probe.setdefault("run", {})
    if isinstance(probe_run, dict):
        probe_run.setdefault("uid", "__validate_probe__")
    probe["results"] = Results().model_dump(by_alias=True)
    try:
        BenchmarkReportV02.model_validate(probe)
    except Exception as e:
        raise PartialReportError(f"BR0.2 partial report failed schema validation: {e}") from e

    return partial


def merge_results(partial: Dict[str, Any], results: Results, *, run_uid_fallback: str) -> Dict[str, Any]:
    """Merge ``results`` into a validated partial and return the final BR0.2
    report as a dict.

    The partial supplies everything except performance results. ``run.uid``
    is auto-generated from ``run_uid_fallback`` only if the partial did not
    set one — required because the BR0.2 schema mandates ``run.uid``.

    The merged document is validated against the full BR0.2 schema; any
    structural issue surfaces here.
    """
    merged: Dict[str, Any] = json.loads(json.dumps(partial))  # deep copy via JSON round-trip

    run = merged.setdefault("run", {})
    if not isinstance(run, dict):
        raise PartialReportError(f"BR0.2 partial report: 'run' must be a mapping or absent, got {type(run).__name__}")
    run.setdefault("uid", run_uid_fallback)

    merged["results"] = results.dump() if hasattr(results, "dump") else results.model_dump(by_alias=True)

    try:
        BenchmarkReportV02.model_validate(merged)
    except Exception as e:
        raise PartialReportError(f"Merged BR0.2 report failed schema validation: {e}") from e
    return merged
