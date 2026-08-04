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
"""Declared vLLM Prometheus metric names exist on a real server (#669).

A stale metric name never errors: the PromQL query matches nothing and the
report field comes back empty (#382 hand-caught exactly such a rename). This
test scrapes a real vLLM's /metrics and requires every metric name the vLLM
client declares in ``get_prometheus_metric_metadata()`` to exist in the
exposition, so upstream renames surface as a red test instead of silently
empty report fields.

Presence is type-aware, mirroring what a Prometheus scrape stores: gauges
by bare name, counters by bare or ``_total``-suffixed name, histograms by
their ``_bucket``/``_count``/``_sum`` series. A request is sent first so
lazily-registered families are present.
"""

import re

import aiohttp
import pytest

from utils.testdata import extract_tarball
from utils.net import get_free_port
from utils.vllm_server import VLLMServerRunner

from inference_perf.client.modelserver.vllm_client import vLLMModelServerClient
from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector

# Vendored tokenizer so building the client stays offline; only the metric
# name declarations are read from it, never the tokenizer itself.
GEMMA_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

# Declared names that a STOCK vLLM does not expose. All five arrived in #348
# ("vLLM latest (0.15.0) production metrics") and are absent from a default
# v0.26.0 server, seemingly gated on optional features (KV offloading and
# similar) whose components never register their metric families on a stock
# configuration. Kept out of the strict check rather than deleted so the
# declarations can be triaged: each is either config-gated (then this list
# documents the gate) or stale (then it should be removed from vllm_client).
CONDITIONALLY_EXPOSED = {
    "vllm:corrupted_requests",
    "vllm:kv_block_idle_before_evict_seconds",
    "vllm:kv_block_lifetime_seconds",
    "vllm:kv_block_reuse_gap_seconds",
    "vllm:prompt_tokens_recomputed",
}


def _declared_metrics(base_url: str, model_name: str) -> dict[str, str]:
    """Metric base names -> metric type, as declared by the vLLM client."""
    client = vLLMModelServerClient(
        metrics_collector=LocalRequestMetricCollector(),
        api_config=APIConfig(type=APIType.Completion),
        uri=base_url,
        model_name=model_name,
        tokenizer_config=CustomTokenizerConfig(pretrained_model_name_or_path=str(extract_tarball(GEMMA_TARBALL))),
        max_tcp_connections=1,
        additional_filters=[],
    )
    declared: dict[str, str] = {}
    for metric in client.get_prometheus_metric_metadata().values():
        if metric is None:
            continue
        # A declaration is either a bare name or a PromQL selector like
        # {__name__=~"vllm:request_success(_total)?"}; either way the base
        # names are the vllm:-prefixed identifiers inside it.
        for name in re.findall(r"vllm:[A-Za-z0-9_]+", metric.name):
            declared[name] = metric.type
    return declared


def _exposed_families(metrics_text: str) -> set[str]:
    """All family and sample names present in a /metrics exposition."""
    names = set()
    for line in metrics_text.splitlines():
        if line.startswith("# TYPE ") or line.startswith("# HELP "):
            names.add(line.split(" ")[2])
        elif line and not line.startswith("#"):
            names.add(line.split("{")[0].split(" ")[0])
    return names


def _is_exposed(name: str, metric_type: str, families: set[str]) -> bool:
    if metric_type == "histogram":
        return all(f"{name}{suffix}" in families for suffix in ("_bucket", "_count", "_sum"))
    if metric_type == "counter":
        return name in families or f"{name}_total" in families
    return name in families


@pytest.mark.asyncio
@pytest.mark.skipif(not VLLMServerRunner.is_available(), reason="no vLLM server or executable available")
async def test_declared_metric_names_exist():
    async with VLLMServerRunner(port=get_free_port()) as server:
        # One real request so families that only register on first use exist.
        async with aiohttp.ClientSession() as http:
            body = {"model": server.model, "prompt": "1 2 3", "max_tokens": 4, "ignore_eos": True}
            async with http.post(f"{server.base_url}/v1/completions", json=body) as resp:
                assert resp.status == 200, f"warmup request failed: {resp.status}"

        families = _exposed_families(await server.fetch_metrics())
        declared = _declared_metrics(server.base_url, server.model)

    assert declared, "vLLM client declared no metric names"
    missing = sorted(
        name
        for name, metric_type in declared.items()
        if name not in CONDITIONALLY_EXPOSED and not _is_exposed(name, metric_type, families)
    )
    assert not missing, (
        f"{len(missing)}/{len(declared)} declared metric names absent from a real vLLM /metrics exposition "
        f"(stale names produce silently empty report fields): {missing}"
    )
