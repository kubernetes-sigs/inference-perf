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
"""Emit inference-perf's slice of a BR0.2 benchmark report.

Inference-perf only speaks to the sections of a BR0.2 report it can fill
truthfully from a run: the schema ``version``, the ``run`` block (a generated
``uid`` and the wall-clock ``time`` window of the stage), and the ``results``
block built from the actual request metrics. Everything else (stack,
scenario, observability beyond what we measure, user/eid/cid/pid/description)
is left absent so a downstream composer can merge another producer's partial
on top with ``yq '. * load("other.yaml")'`` and have no inference-perf field
silently overwrite the composer's data.

Convention: emitted maps omit ``None`` values entirely (``exclude_none=True``)
so a deep-merge never overwrites a real value with ``null``. Datetimes are
serialized as ISO-8601 strings (``mode="json"``); the document is otherwise
plain YAML with no anchors, tags, or aliases.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List

from inference_perf.apis import RequestLifecycleMetric
from inference_perf.utils.custom_tokenizer import CustomTokenizer

from .adapter import build_results
from .schema_v0_2 import VERSION, Run, RunTime


def generate_run_uid(stage_id: int) -> str:
    """Generate a run uid for a stage. Stable shape, unique per call."""
    return f"inference-perf-stage-{stage_id}-{uuid.uuid4().hex[:8]}"


def build_partial_report(
    stage_metrics: List[RequestLifecycleMetric],
    tokenizer: CustomTokenizer | None,
    *,
    run_uid: str,
) -> Dict[str, Any]:
    """Build the inference-perf partial of a BR0.2 report for one stage.

    Returns a plain dict, ready to be serialized as YAML and dropped alongside
    the other report files. ``None``-valued fields are stripped so the file
    yq-merges cleanly with partials from other producers.
    """
    run = Run(uid=run_uid, time=_build_run_time(stage_metrics))
    results = build_results(stage_metrics, tokenizer)

    return {
        "version": VERSION,
        "run": run.model_dump(mode="json", by_alias=True, exclude_none=True),
        "results": results.model_dump(mode="json", by_alias=True, exclude_none=True),
    }


def _build_run_time(stage_metrics: List[RequestLifecycleMetric]) -> RunTime | None:
    """Derive stage start/end/duration from request lifecycle timestamps.

    Returns ``None`` when no metrics are available, so the field is dropped
    from the emitted partial rather than emitted as a null block.
    """
    if not stage_metrics:
        return None
    start_ts = min(m.start_time for m in stage_metrics)
    end_ts = max(m.end_time for m in stage_metrics)
    start = datetime.datetime.fromtimestamp(start_ts, tz=datetime.timezone.utc)
    end = datetime.datetime.fromtimestamp(end_ts, tz=datetime.timezone.utc)
    return RunTime(
        start=start,
        end=end,
        duration=_iso8601_duration(end - start),
    )


def _iso8601_duration(delta: datetime.timedelta) -> str:
    """Format a positive timedelta as an ISO-8601 duration (``PT<seconds>S``).

    Sub-second precision is preserved to milliseconds; longer durations are
    not broken into hours/minutes (a ``PT<n>S`` form is well-formed per the
    spec and trivially parseable by downstream consumers).
    """
    total_seconds = max(delta.total_seconds(), 0.0)
    return f"PT{total_seconds:.3f}S"
