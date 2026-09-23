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
"""Prometheus metric-family bookkeeping for the SGLang and TGI drift checks (#669).

Three representations of "which metrics exist" meet here, and this module
converts between them:

- **declared**: the metric objects a client will query, read from its
  ``get_prometheus_metric_metadata()``. Resolving them (``declared_metrics``,
  ``resolves``, ``is_exposed``, ``parse_exposition``) lives in
  ``utils.metric_families``, shared with the vLLM check, so the two checks
  cannot drift apart.
- **exposed**: what a live server's ``/metrics`` text contains.
- **fixture**: the committed snapshot of a server's metric families
  (``e2e/testdata/server_metric_families/<server>.txt``), which stands in for
  a live server so the declared-vs-exposed check runs with no GPU.

Unlike the vLLM half of #669, which pins releases and keeps one golden per
pin, this half tracks whatever upstream calls ``latest``: there is exactly one
fixture per server and the scheduled job rewrites it in place, so the review
artifact is a diff of names rather than a new file.

Fixture format: a header of ``# key: value`` lines followed by one
``<family> <type>`` pair per line, sorted. Other ``#`` comments and blank
lines are ignored. ``*_created`` series are dropped at capture time: they are
prometheus_client per-series creation timestamps, mechanical companions of
their family rather than metrics anyone declares.

The header's ``provenance`` key is load bearing. ``live-scrape`` means the
contents came off a running server's ``/metrics``. ``upstream-source`` means
they were derived by reading the pinned upstream registration code, which is a
strict superset of any single live exposition (it includes families gated on
optional features, and families a server only registers once exercised). A
check that needs family-for-family equality is only meaningful against
``live-scrape`` and must skip otherwise.
"""

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type

from inference_perf.client.modelserver.openai_client import openAIModelServerClient
from inference_perf.client.modelserver.sglang_client import SGlangModelServerClient
from inference_perf.client.modelserver.tgi_client import TGImodelServerClient

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "testdata" / "server_metric_families"

# Provenance values a fixture header may carry. See the module docstring.
LIVE_SCRAPE = "live-scrape"
UPSTREAM_SOURCE = "upstream-source"
PROVENANCES = (LIVE_SCRAPE, UPSTREAM_SOURCE)

# Header keys every fixture must carry, so a fixture can never be committed
# without saying where its contents came from.
REQUIRED_HEADER_KEYS = ("provenance", "server", "version", "source", "captured")


@dataclass(frozen=True)
class ServerSpec:
    """Everything the drift checks and the capture script need per server."""

    name: str
    # Family-name prefix that identifies this server's own metrics. Everything
    # else on /metrics (python_*, process_*, http_*) belongs to the runtime,
    # not the server, and is not what this repo declares against.
    prefix: str
    client_cls: Type[openAIModelServerClient]
    # Env var pointing at an already-running server, for the live checks.
    base_url_env: str
    # Endpoint returning a JSON object with a "version" key, used by the
    # capture script to stamp the fixture with the release it scraped.
    version_path: str
    version_key: str


SERVERS: Dict[str, ServerSpec] = {
    "sglang": ServerSpec(
        name="sglang",
        prefix="sglang:",
        client_cls=SGlangModelServerClient,
        base_url_env="E2E_SGLANG_BASE_URL",
        version_path="/server_info",
        version_key="version",
    ),
    "tgi": ServerSpec(
        name="tgi",
        prefix="tgi_",
        client_cls=TGImodelServerClient,
        base_url_env="E2E_TGI_BASE_URL",
        version_path="/info",
        version_key="version",
    ),
}


@dataclass(frozen=True)
class Fixture:
    """A parsed fixture file: its provenance header and its family -> type map."""

    header: Dict[str, str]
    families: Dict[str, str]

    @property
    def provenance(self) -> str:
        return self.header.get("provenance", "")

    @property
    def version(self) -> str:
        return self.header.get("version", "")


def fixture_path(server: str) -> Path:
    return FIXTURE_DIR / f"{server}.txt"


def external_base_url(spec: ServerSpec) -> Optional[str]:
    """The already-running server for this spec, or None.

    There is deliberately no spawning runner here, unlike the vLLM half's
    ``VLLMServerRunner``: SGLang and TGI both need an accelerator to start, so
    until #641 provides one the only honest provisioning mode is "somebody
    else started it and told us where". The live checks skip without it.
    """
    url = os.environ.get(spec.base_url_env, "").strip()
    return url.rstrip("/") or None


def fetch_text(url: str, timeout: float = 30.0) -> str:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"refusing to fetch non-http url: {url!r}")
    # Scheme is checked above, so this never reaches file:// or similar.
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return str(resp.read().decode("utf-8"))


def fetch_json(url: str, timeout: float = 30.0) -> Any:
    return json.loads(fetch_text(url, timeout))


def parse_fixture(text: str) -> Fixture:
    header: Dict[str, str] = {}
    families: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            key, sep, value = body.partition(":")
            if sep and key in REQUIRED_HEADER_KEYS:
                header[key] = value.strip()
            continue
        name, _, metric_type = line.partition(" ")
        families[name] = metric_type.strip()
    return Fixture(header=header, families=families)


def load_fixture(server: str) -> Fixture:
    return parse_fixture(fixture_path(server).read_text(encoding="utf-8"))


def format_fixture(
    families: Dict[str, str],
    *,
    server: str,
    version: str,
    provenance: str,
    source: str,
    captured: str,
    notes: str = "",
) -> str:
    """Render a fixture file. Capture and check share this format, so a
    committed fixture is by construction what the checks would have parsed."""
    if provenance not in PROVENANCES:
        raise ValueError(f"unknown provenance {provenance!r}, expected one of {PROVENANCES}")
    lines = [
        f"# {server} metric families, as inference-perf expects to find them on /metrics.",
        "#",
        f"# provenance: {provenance}",
        f"# server: {server}",
        f"# version: {version}",
        f"# source: {source}",
        f"# captured: {captured}",
        "#",
        "# Regenerate against a running server:",
        f"#   python scripts/capture_server_metric_names.py --server {server} \\",
        "#     --base-url http://127.0.0.1:<port>",
    ]
    if notes:
        lines += ["#"] + [f"# {line}".rstrip() for line in notes.strip("\n").splitlines()]
    body = "".join(f"{name} {metric_type}\n" for name, metric_type in sorted(families.items()))
    return "\n".join(lines) + "\n" + body
