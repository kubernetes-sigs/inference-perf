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
"""Rewrite a server's metric-family fixture from a running server (#669).

Scrapes ``<base-url>/metrics``, keeps the family names and types that belong
to the server itself, and writes them to
``e2e/testdata/server_metric_families/<server>.txt`` with a header recording
the release it came from. The scheduled drift workflow
(``.github/workflows/metric_name_drift.yml``) runs this against the latest
upstream release of SGLang and TGI and opens a pull request when the file
changes; run it by hand against your own server to refresh a fixture out of
band.

Exits 0 whether or not the file changed. ``--check`` instead exits 1 on a
difference without writing, for callers that only want to know.

Two things to know before trusting the output:

- Both servers register metric families lazily, so scrape only after the
  server has served at least one request. The workflow does that warmup; a
  by-hand run should too, or the capture will under-report.
- The server must actually be exporting. SGLang needs ``--enable-metrics``.

The parsing and formatting live in ``e2e/utils/server_metric_names.py``, the
same module the drift checks read fixtures with, so a captured file is by
construction what those checks would have parsed.
"""

import argparse
import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The e2e helpers are not an installed package; e2e/ is what pytest puts on
# sys.path for those tests, so put it there too rather than duplicating the
# fixture format in a second place.
sys.path.insert(0, str(REPO_ROOT / "e2e"))

from utils.server_metric_names import (  # noqa: E402
    LIVE_SCRAPE,
    SERVERS,
    ServerSpec,
    fetch_json,
    fetch_text,
    fixture_path,
    format_fixture,
    parse_exposition,
    parse_fixture,
)


def resolve_version(spec: ServerSpec, base_url: str) -> str:
    """The running release, from the server's own version endpoint."""
    payload = fetch_json(f"{base_url}{spec.version_path}")
    version = payload.get(spec.version_key) if isinstance(payload, dict) else None
    if not version:
        raise SystemExit(f"{base_url}{spec.version_path} did not report a {spec.version_key!r}; pass --version explicitly")
    return str(version)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", required=True, choices=sorted(SERVERS), help="which server's fixture to rewrite")
    parser.add_argument("--base-url", required=True, help="base URL of a running server, e.g. http://127.0.0.1:30000")
    parser.add_argument("--version", help="release tag to stamp; read from the server's version endpoint when omitted")
    parser.add_argument("--output", type=Path, help="fixture path to write; defaults to the checked-in one")
    parser.add_argument("--check", action="store_true", help="exit 1 if the fixture would change, and write nothing")
    args = parser.parse_args()

    spec = SERVERS[args.server]
    base_url = args.base_url.rstrip("/")
    version = args.version or resolve_version(spec, base_url)

    families = parse_exposition(fetch_text(f"{base_url}/metrics"), spec.prefix)
    if not families:
        raise SystemExit(
            f"{base_url}/metrics exposed no {spec.prefix}* families. The server may not have served a request yet, "
            f"or metrics may be disabled (SGLang needs --enable-metrics). Refusing to write an empty fixture."
        )

    rendered = format_fixture(
        families,
        server=spec.name,
        version=version,
        provenance=LIVE_SCRAPE,
        source=f"{base_url}/metrics",
        captured=datetime.date.today().isoformat(),
        notes=(
            "Captured from a running server after at least one request, so this lists the families that\n"
            "server actually exposed under its launch configuration, not every family it could register."
        ),
    )

    path = args.output or fixture_path(spec.name)
    # The header carries a capture date and a source URL that move on every
    # run, so comparing whole files would report a change every week. Compare
    # the family list and the reported version instead: those are the two
    # things a reviewer of a refresh PR is being asked about.
    previous = parse_fixture(path.read_text(encoding="utf-8")) if path.is_file() else None
    changed = previous is None or previous.families != families or previous.version != version

    if args.check:
        print(f"{spec.name} {version}: {len(families)} families, {'CHANGED' if changed else 'unchanged'}")
        return 1 if changed else 0

    if not changed:
        print(f"{path} already matches {spec.name} {version} ({len(families)} families); left untouched")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    print(f"wrote {path} ({len(families)} families from {spec.name} {version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
