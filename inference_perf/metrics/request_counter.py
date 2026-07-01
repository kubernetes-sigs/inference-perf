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
from collections import defaultdict
from typing import Any, Iterable, Iterator, List, Optional, Tuple

from inference_perf.apis import RequestLifecycleMetric


class RequestSentCounter:
    """In-process counter for the total number of requests sent to the model server.

    A request is considered "sent" once its lifecycle completes and produces a
    RequestLifecycleMetric, which happens for both successful and failed requests
    (errors are captured rather than dropped). The counter keeps a running total
    plus a breakdown by (stage, status) so per-stage send volume and the
    success/failure split are distinguishable. It is a plain in-process counter,
    read by exposition surfaces (e.g. Prometheus) and surfaced in the run report.
    """

    def __init__(self) -> None:
        self.total: int = 0
        # keyed by (stage_id, status) -> count
        self._breakdown: "defaultdict[tuple[Any, str], int]" = defaultdict(int)

    def observe(self, metric: RequestLifecycleMetric) -> None:
        """Record a single sent request."""
        status = "failure" if metric.error is not None else "success"
        self.total += 1
        self._breakdown[(metric.stage_id, status)] += 1

    @classmethod
    def from_metrics(cls, metrics: Iterable[RequestLifecycleMetric]) -> "RequestSentCounter":
        counter = cls()
        for metric in metrics:
            counter.observe(metric)
        return counter

    def iter_counts(self) -> Iterator[Tuple[Optional[int], str, int]]:
        """Yield (stage_id, status, count) for each observed combination.

        Used by exposition surfaces (e.g. Prometheus) to read the live breakdown.
        """
        for (stage_id, status), count in self._breakdown.items():
            yield stage_id, status, count

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the counter for reports."""
        try:
            items = sorted(
                self._breakdown.items(),
                key=lambda item: (item[0][0] if item[0][0] is not None else -1, item[0][1]),
            )
        except TypeError:
            # Keys are normally (int|None, str); if a caller passes exotic
            # (e.g. mock) values that are not comparable, fall back to insertion order
            # rather than failing to serialize the report.
            items = list(self._breakdown.items())
        by_stage_status: List[dict[str, Any]] = [
            {
                "stage_id": stage_id,
                "status": status,
                "count": count,
            }
            for (stage_id, status), count in items
        ]
        return {
            "total": self.total,
            "by_stage_status": by_stage_status,
        }
