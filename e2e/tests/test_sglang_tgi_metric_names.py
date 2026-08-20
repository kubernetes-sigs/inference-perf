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
"""Declared SGLang and TGI Prometheus metric names still exist upstream (#669).

A stale metric name never errors. The PromQL query matches nothing, the
report field comes back empty, and a reader cannot tell "the server did not
report this" from "we asked for a name that no longer exists". #382 caught one
such rename (``sglang:cache_hit_rate`` to ``sglang:token_usage``) by hand.

The vLLM half of #669 (#697) pins releases and can start a real server on a
plain CI runner in CPU mode. SGLang and TGI cannot: both want an accelerator,
which this project has no runner for yet (#641). So this half is built the way
#669 proposes instead, around a checked-in fixture that a scheduled job
refreshes:

1. ``test_declared_names_resolve_against_fixture`` (serverless, runs today in
   the normal e2e job): every name the client declares must resolve against
   the committed metric-family fixture for that server. This is the check that
   catches a rename, and it needs no GPU because the fixture stands in for the
   server.
2. ``test_known_unresolved_are_still_unresolved`` (serverless): every entry in
   the ``KNOWN_UNRESOLVED`` triage list must still be both declared and
   unresolved. An allowlist that silently stops applying is how a gate rots,
   so fixing a declaration turns this red and forces the entry to be deleted.
3. ``test_exposed_families_match_fixture`` (live): the running server's
   families equal the fixture exactly, which is what keeps the fixture honest.
   Skips loudly while the fixture is source derived, because a source derived
   fixture is a superset of any live exposition and equality is the wrong
   oracle against it.
4. ``test_declared_metric_names_exist`` (live): the end invariant itself,
   declared names present in a real exposition.

The live checks read ``E2E_SGLANG_BASE_URL`` / ``E2E_TGI_BASE_URL`` and skip
when unset. Nothing here starts a server.

Regenerating a fixture is ``scripts/capture_server_metric_names.py``, which
the scheduled workflow (``.github/workflows/metric_name_drift.yml``) runs
against the latest upstream release and turns into a pull request when the
result differs. Capture and check share ``utils.server_metric_names``, so a
committed fixture is by construction what these checks would have parsed.
"""

from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from utils.server_metric_names import (
    LIVE_SCRAPE,
    PROVENANCES,
    REQUIRED_HEADER_KEYS,
    SERVERS,
    ServerSpec,
    declared_metrics,
    exposed_names,
    external_base_url,
    fetch_text,
    fixture_path,
    is_exposed,
    load_fixture,
    parse_exposition,
    resolves,
)

from inference_perf.config import APIConfig, APIType

SERVER_IDS = sorted(SERVERS)

# Declared names that do NOT resolve against the committed fixture, kept out of
# the strict check so the rest of the surface stays gated, with the reason
# written down so each is a triage item rather than a silent exception. Every
# entry is asserted to still apply by test_known_unresolved_are_still_unresolved,
# so none of these can rot into a permanent hole.
KNOWN_UNRESOLVED: Dict[str, Dict[str, str]] = {
    "sglang": {
        # Found by this test, not by hand: SGLang registers no metric of this
        # name anywhere in v0.5.17. The only occurrences left in that tree are
        # a stale exposition sample in sgl-model-gateway/tests and the equally
        # stale docs/docs/references/production_metrics.mdx dump. The live
        # consequence is that summary_prometheus_metrics.json carries an empty
        # time_per_output_token section for every SGLang run.
        # sglang:inter_token_latency_seconds is the surviving per-token
        # histogram and sglang_client already declares it as a custom metric,
        # so the fix is a one-line swap in sglang_client.py. Doing it here
        # would mix a client behaviour change into a test-only PR, so it is
        # left for its own change with this entry as the repro.
        "sglang:time_per_output_token_seconds": "not registered by SGLang v0.5.17; see #669",
    },
    "tgi": {},
}


def declared_for(spec: ServerSpec) -> Dict[str, str]:
    """Metric base names -> type, as the server's client subclass declares them.

    Only the declarations are read, never the tokenizer, so CustomTokenizer is
    patched out to keep construction offline (same approach as
    tests/required/client/metricsclient/test_prometheus_query_goldens.py).
    """
    with patch("inference_perf.client.modelserver.openai_client.CustomTokenizer"):
        client = spec.client_cls(
            metrics_collector=MagicMock(),
            api_config=APIConfig(type=APIType.Completion, streaming=False),
            uri="http://127.0.0.1:1",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            tokenizer_config=None,
            max_tcp_connections=1,
            additional_filters=[],
        )
    declared = declared_metrics(client.get_prometheus_metric_metadata())
    assert declared, f"{spec.name} client declared no metric names"
    return declared


def live_metrics_text(spec: ServerSpec) -> str:
    base_url = external_base_url(spec)
    if not base_url:
        pytest.skip(f"{spec.base_url_env} not set; no running {spec.name} to scrape")
    return fetch_text(f"{base_url}/metrics")


@pytest.mark.parametrize("server", SERVER_IDS)
def test_fixture_header_is_complete(server: str) -> None:
    # A fixture that does not say where it came from cannot be reviewed, and
    # an unattributed fixture is exactly how a hand-written guess becomes a
    # pinned expectation.
    fixture = load_fixture(server)
    missing = [key for key in REQUIRED_HEADER_KEYS if not fixture.header.get(key)]
    assert not missing, f"{fixture_path(server).name} header is missing {missing}"
    assert fixture.provenance in PROVENANCES, (
        f"{fixture_path(server).name} declares unknown provenance {fixture.provenance!r}, expected one of {PROVENANCES}"
    )
    assert fixture.header["server"] == server
    assert fixture.families, f"{fixture_path(server).name} lists no metric families"


@pytest.mark.parametrize("server", SERVER_IDS)
def test_declared_names_resolve_against_fixture(server: str) -> None:
    spec = SERVERS[server]
    fixture = load_fixture(server)
    declared = declared_for(spec)
    allowed = KNOWN_UNRESOLVED[server]

    missing = sorted(
        name for name, metric in declared.items() if name not in allowed and not resolves(metric, fixture.families)
    )
    assert not missing, (
        f"{len(missing)}/{len(declared)} names declared by the {server} client do not resolve against "
        f"{fixture_path(server).name} ({fixture.version}); a stale name produces a silently empty report "
        f"field, so either fix the declaration or, if upstream really dropped it, add it to "
        f"KNOWN_UNRESOLVED with the reason: {missing}"
    )


@pytest.mark.parametrize("server", SERVER_IDS)
def test_known_unresolved_are_still_unresolved(server: str) -> None:
    spec = SERVERS[server]
    allowed = KNOWN_UNRESOLVED[server]
    if not allowed:
        pytest.skip(f"no known-unresolved declarations for {server}")
    fixture = load_fixture(server)
    declared = declared_for(spec)

    undeclared = sorted(name for name in allowed if name not in declared)
    assert not undeclared, f"{server} no longer declares {undeclared}; drop the KNOWN_UNRESOLVED entries"

    now_resolving = sorted(name for name in allowed if resolves(declared[name], fixture.families))
    assert not now_resolving, (
        f"{server} declarations {now_resolving} now resolve against {fixture_path(server).name}; "
        f"drop their KNOWN_UNRESOLVED entries so the strict check covers them again"
    )


@pytest.mark.parametrize("server", SERVER_IDS)
def test_exposed_families_match_fixture(server: str) -> None:
    spec = SERVERS[server]
    fixture = load_fixture(server)
    if fixture.provenance != LIVE_SCRAPE:
        pytest.skip(
            f"{fixture_path(server).name} provenance is {fixture.provenance!r}, not {LIVE_SCRAPE!r}: it was derived "
            f"from pinned upstream source and is a superset of any live exposition, so family-for-family equality "
            f"would fail for reasons that are not drift. Regenerate it with "
            f"scripts/capture_server_metric_names.py against a running {server} and this check activates."
        )
    families = parse_exposition(live_metrics_text(spec), spec.prefix)
    assert families, f"running {server} exposed no {spec.prefix}* metric families"

    added = sorted(set(families) - set(fixture.families))
    removed = sorted(set(fixture.families) - set(families))
    retyped = sorted(n for n in set(fixture.families) & set(families) if fixture.families[n] != families[n])
    assert not (added or removed or retyped), (
        f"live {server} exposition diverges from {fixture_path(server).name} ({fixture.version}): "
        f"added={added}, removed={removed}, retyped={retyped}. Regenerate the fixture with "
        f"scripts/capture_server_metric_names.py and commit the diff."
    )


@pytest.mark.parametrize("server", SERVER_IDS)
def test_declared_metric_names_exist(server: str) -> None:
    spec = SERVERS[server]
    names = exposed_names(live_metrics_text(spec))
    declared = declared_for(spec)
    allowed = KNOWN_UNRESOLVED[server]

    missing = sorted(name for name, metric in declared.items() if name not in allowed and not is_exposed(metric, names))
    assert not missing, (
        f"{len(missing)}/{len(declared)} names declared by the {server} client are absent from a real "
        f"/metrics exposition (stale names produce silently empty report fields): {missing}"
    )
