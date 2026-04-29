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
"""Per-pod Prometheus scraping.

Each pod's /metrics endpoint is scraped on an interval. Samples are bucketed
per-pod, aggregated at collect() time into the BR0.2
observability.components[] resource metrics + the ad-hoc `vllm_*` / `epp_*`
fields seen in the upstream example. No PromQL server required.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)


class PodScrapeTarget(NamedTuple):
    pod_name: str
    component_label: str
    role: str
    url: str


# Metrics we know how to project into BR0.2. Each entry: (prom_name, bucket_key, units).
_TRACKED_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("vllm:gpu_cache_usage_perc", "vllm_kv_cache_usage_perc", "percent"),
    ("vllm:num_requests_running", "vllm_num_requests_running", "count"),
    ("vllm:num_requests_waiting", "vllm_num_requests_waiting", "count"),
    ("vllm:num_preemptions_total", "vllm_num_preemptions_total", "count"),
    ("vllm:prefix_cache_hits_total", "vllm_prefix_cache_hits_total", "tokens"),
    ("vllm:prefix_cache_queries_total", "vllm_prefix_cache_queries_total", "tokens"),
    (
        "inference_extension:inference_pool:average_kv_cache_utilization",
        "epp_pool_avg_kv_cache_utilization",
        "percent",
    ),
    ("inference_extension:inference_pool:average_queue_size", "epp_pool_avg_queue_size", "count"),
    ("inference_extension:inference_pool:ready_pods", "epp_pool_ready_pods", "count"),
)
# Additional categories of the schema that a future, richer mapping should fill:
#   - GPU utilization / memory / power (require DCGM-exporter)
#   - vllm:nixl_xfer_time_seconds_{sum,count} (histograms)


class PerPodPromScraper:
    """Direct-scrape Prometheus text from each pod's /metrics endpoint.

    Targets are supplied by the composing collector (it knows which pods are
    in the stack). The scraper is safe to start with an empty target list.
    """

    def __init__(
        self,
        targets: Optional[List[PodScrapeTarget]] = None,
        interval_seconds: float = 15.0,
        request_timeout_seconds: float = 5.0,
    ) -> None:
        self.targets: List[PodScrapeTarget] = list(targets or [])
        self.interval_seconds = max(1.0, interval_seconds)
        self.request_timeout_seconds = max(1.0, request_timeout_seconds)
        self._task: Optional[asyncio.Task[None]] = None
        # samples[bucket_key][pod_name] = list[float] (one entry per scrape tick)
        self._samples: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        # pod -> (component_label, role) so we can attach per-component labels at collect()
        self._pod_metadata: Dict[str, Tuple[str, str]] = {}

    def set_targets(self, targets: List[PodScrapeTarget]) -> None:
        self.targets = list(targets)
        self._pod_metadata = {t.pod_name: (t.component_label, t.role) for t in targets}

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def collect(self) -> Dict[str, Any]:
        """Return aggregated per-pod statistics keyed by metric bucket name."""
        result: Dict[str, Any] = {}
        for _, bucket_key, units in _TRACKED_METRICS:
            per_pod = self._samples.get(bucket_key)
            if not per_pod:
                continue
            components = []
            all_values: List[float] = []
            for pod_name, values in per_pod.items():
                if not values:
                    continue
                component_label, role = self._pod_metadata.get(pod_name, (pod_name, "replica"))
                components.append(
                    {
                        "component_id": component_label,
                        "pod": pod_name,
                        "role": role,
                        "statistics": {**_summary(values), "units": units},
                    }
                )
                all_values.extend(values)
            entry: Dict[str, Any] = {"components": components}
            if all_values:
                entry["aggregated"] = {**_summary(all_values), "units": units}
            result[bucket_key] = entry
        return result

    async def _loop(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.request_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                try:
                    await asyncio.gather(
                        *(self._scrape_one(session, t) for t in self.targets),
                        return_exceptions=True,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning("PerPodPromScraper: tick failed: %s", e)
                await asyncio.sleep(self.interval_seconds)

    async def _scrape_one(self, session: aiohttp.ClientSession, target: PodScrapeTarget) -> None:
        try:
            async with session.get(target.url) as resp:
                if resp.status != 200:
                    return
                text = await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug("PerPodPromScraper: scrape %s failed: %s", target.url, e)
            return
        self._parse_into_samples(target, text)

    def _parse_into_samples(self, target: PodScrapeTarget, text: str) -> None:
        from prometheus_client.parser import text_string_to_metric_families

        wanted = {prom_name: bucket for prom_name, bucket, _ in _TRACKED_METRICS}
        for family in text_string_to_metric_families(text):
            bucket = wanted.get(family.name)
            if bucket is None:
                continue
            for sample in family.samples:
                bucket_key = bucket
                # Histograms expose `_sum` / `_count` etc; only aggregate the gauge/counter value.
                if sample.name != family.name:
                    continue
                value = sample.value
                if value != value:  # NaN
                    continue
                self._samples[bucket_key][target.pod_name].append(float(value))


def _summary(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "stddev": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }
