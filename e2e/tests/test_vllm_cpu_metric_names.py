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
"""Declared vLLM Prometheus metric names exist on the servers we gate on (#669).

A stale metric name never errors: the PromQL query matches nothing and the
report field comes back empty (#382 hand-caught exactly such a rename, #698
tracks five more). A pinned release's metric names never change, so for pins
the live server is only needed to prove the committed snapshot is honest;
``latest`` floats with upstream, so only a live server can vouch for it.
That splits the invariant into three checks, each in the cheapest tier that
can catch its failure mode:

1. ``test_declared_names_resolve_against_goldens`` (serverless, runs in the
   gating sim invocation): every name the client declares resolves against
   every committed per-release golden. Catches stale declarations, and
   catches changes to the declaration API itself, on the PR that makes them.
2. ``test_exposed_families_match_golden`` (live, pinned releases): the
   running server's ``vllm:*`` families equal the release's golden exactly,
   keeping the goldens honest. Skips when no golden exists for the running
   release, which is the deliberate state of ``latest``: a floating tag
   would guarantee golden rot, so it never gets one.
3. ``test_declared_metric_names_exist`` (live, every release in the table):
   declared names present in the real exposition, the end-invariant itself.
   On ``latest`` this is the early warning that an upstream rename is
   coming, red on live PRs before, not at, the next pin bump.

Which release the live server runs is not observable from the server itself
(``/version`` reports the app version, not the image tag), so CI exports
``E2E_VLLM_VERSION`` per slice; without it the golden check has nothing to
compare against and skips. Set ``E2E_UPDATE_METRIC_GOLDENS=1`` to rewrite
the running release's golden in place instead of asserting (then commit the
diff); capture and check share one code path, so a golden is by construction
what the check would have accepted.
"""

import os
from typing import Any

import aiohttp
import pytest

from utils.metric_families import (
    GOLDEN_DIR,
    declared_metrics,
    exposed_names,
    exposed_vllm_families,
    format_golden,
    golden_path,
    in_golden,
    is_exposed,
    load_golden,
)
from utils.net import get_free_port
from utils.testdata import extract_tarball
from utils.vllm_server import DEFAULT_MODEL, VLLMServerRunner

from inference_perf.client.modelserver.metrics.base import Metric
from inference_perf.client.modelserver.vllm_client import vLLMModelServerClient
from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector

# One worker for all vLLM CPU tests: the token-mismatch module's /metrics
# counter deltas need exclusive access to the shared server.
pytestmark = pytest.mark.xdist_group(name="vllm-cpu-server")

ENV_VLLM_VERSION = "E2E_VLLM_VERSION"
ENV_UPDATE_GOLDENS = "E2E_UPDATE_METRIC_GOLDENS"

# Vendored tokenizer so building the client stays offline; only the metric
# name declarations are read from it, never the tokenizer itself.
GEMMA_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

# Declared names that a STOCK vLLM does not expose. All five arrived in #348
# ("vLLM latest (0.15.0) production metrics") and are absent from a default
# v0.26.0 server, seemingly gated on optional features (KV offloading and
# similar) whose components never register their metric families on a stock
# configuration. Kept out of the strict checks rather than deleted so the
# declarations can be triaged: each is either config-gated (then this list
# documents the gate) or stale (then it should be removed from vllm_client).
CONDITIONALLY_EXPOSED = {
    "vllm:corrupted_requests",
    "vllm:kv_block_idle_before_evict_seconds",
    "vllm:kv_block_lifetime_seconds",
    "vllm:kv_block_reuse_gap_seconds",
    "vllm:prompt_tokens_recomputed",
}

# Declarations whose queries select nothing, with a fix already in flight. Kept
# apart from CONDITIONALLY_EXPOSED because the reason is different: those names
# are gated off on a stock server, these are simply queried under a name the
# server does not use. Found by this check once it started asking what the
# queries select rather than whether the declared name is findable (#669).
#
# The list cannot rot: test_known_unresolved_still_do_not_resolve fails the
# moment an entry starts resolving, which forces it to be deleted.
KNOWN_UNRESOLVED = {
    "vllm:prompt_tokens": "queried bare; v0.26.0 exposes vllm:prompt_tokens_total. Fixed by #568",
    "vllm:generation_tokens": "queried bare; v0.26.0 exposes vllm:generation_tokens_total. Fixed by #568",
}

# Names excluded from the strict checks for either reason.
SKIPPED = CONDITIONALLY_EXPOSED | set(KNOWN_UNRESOLVED)

GOLDEN_FILES = sorted(GOLDEN_DIR.glob("*.txt"))


def _declared(base_url: str, model_name: str) -> dict[str, Metric[Any]]:
    """Declared metric name -> the metric object, as declared by the vLLM client."""
    client = vLLMModelServerClient(
        metrics_collector=LocalRequestMetricCollector(),
        api_config=APIConfig(type=APIType.Completion),
        uri=base_url,
        model_name=model_name,
        tokenizer_config=CustomTokenizerConfig(pretrained_model_name_or_path=str(extract_tarball(GEMMA_TARBALL))),
        max_tcp_connections=1,
        additional_filters=[],
    )
    return declared_metrics(client.get_prometheus_metric_metadata())


async def _warmed_up_exposition(server: VLLMServerRunner) -> str:
    """The server's /metrics text after one real request, so families that
    only register on first use exist."""
    async with aiohttp.ClientSession() as http:
        body = {"model": server.model, "prompt": "1 2 3", "max_tokens": 4, "ignore_eos": True}
        async with http.post(f"{server.base_url}/v1/completions", json=body) as resp:
            assert resp.status == 200, f"warmup request failed: {resp.status}"
    return await server.fetch_metrics()


def test_goldens_are_committed() -> None:
    # An empty golden directory would silently deselect the serverless check
    # below, and a gate that silently skips gates nothing.
    assert GOLDEN_FILES, f"no metric-family goldens committed under {GOLDEN_DIR}"


@pytest.mark.parametrize("golden_file", GOLDEN_FILES, ids=lambda p: p.stem)
def test_declared_names_resolve_against_goldens(golden_file) -> None:
    golden = load_golden(golden_file)
    assert golden, f"golden {golden_file} is empty"
    declared = _declared("http://127.0.0.1:1", DEFAULT_MODEL)
    assert declared, "vLLM client declared no metric names"

    missing = sorted(name for name, metric in declared.items() if name not in SKIPPED and not in_golden(metric, golden))
    assert not missing, (
        f"{len(missing)}/{len(declared)} declared metric names do not resolve against {golden_file.name} "
        f"(stale names produce silently empty report fields): {missing}"
    )


@pytest.mark.parametrize("golden_file", GOLDEN_FILES, ids=lambda p: p.stem)
def test_known_unresolved_still_do_not_resolve(golden_file) -> None:
    # Guards the allowlist above. Each KNOWN_UNRESOLVED name must still be declared
    # and must still fail to resolve; the moment #568 lands and vllm:prompt_tokens
    # starts selecting vllm:prompt_tokens_total, this goes red and the entry has to
    # be deleted. An allowlist that quietly stops applying is how a gate rots.
    golden = load_golden(golden_file)
    declared = _declared("http://127.0.0.1:1", DEFAULT_MODEL)

    undeclared = sorted(name for name in KNOWN_UNRESOLVED if name not in declared)
    assert not undeclared, f"no longer declared, drop the KNOWN_UNRESOLVED entries: {undeclared}"

    now_resolving = sorted(name for name in KNOWN_UNRESOLVED if in_golden(declared[name], golden))
    assert not now_resolving, (
        f"{now_resolving} now resolve against {golden_file.name}; drop their KNOWN_UNRESOLVED "
        f"entries so the strict check covers them again"
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not VLLMServerRunner.is_available(), reason="no vLLM server or executable available")
async def test_exposed_families_match_golden():
    release_tag = os.environ.get(ENV_VLLM_VERSION)
    if not release_tag:
        pytest.skip(f"{ENV_VLLM_VERSION} not set; cannot tell which release the server runs")
    golden_file = golden_path(release_tag)
    if not os.environ.get(ENV_UPDATE_GOLDENS) and not golden_file.is_file():
        pytest.skip(f"no golden for release {release_tag} (deliberate for floating tags like latest)")

    async with VLLMServerRunner(port=get_free_port()) as server:
        families = exposed_vllm_families(await _warmed_up_exposition(server))
    assert families, "server exposed no vllm:* metric families"

    if os.environ.get(ENV_UPDATE_GOLDENS):
        golden_file.parent.mkdir(parents=True, exist_ok=True)
        golden_file.write_text(format_golden(families, release_tag), encoding="utf-8")

    golden = load_golden(golden_file)
    added = sorted(set(families) - set(golden))
    removed = sorted(set(golden) - set(families))
    retyped = sorted(n for n in set(golden) & set(families) if golden[n] != families[n])
    assert not (added or removed or retyped), (
        f"live {release_tag} exposition diverges from {golden_file.name} "
        f"(added={added}, removed={removed}, retyped={retyped}); if the launch config changed "
        f"on purpose, regenerate the golden (see its header) and commit the diff"
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not VLLMServerRunner.is_available(), reason="no vLLM server or executable available")
async def test_declared_metric_names_exist():
    async with VLLMServerRunner(port=get_free_port()) as server:
        names = exposed_names(await _warmed_up_exposition(server))
        declared = _declared(server.base_url, server.model)

    assert declared, "vLLM client declared no metric names"
    missing = sorted(name for name, metric in declared.items() if name not in SKIPPED and not is_exposed(metric, names))
    assert not missing, (
        f"{len(missing)}/{len(declared)} declared metric names absent from a real vLLM /metrics exposition "
        f"(stale names produce silently empty report fields): {missing}"
    )
