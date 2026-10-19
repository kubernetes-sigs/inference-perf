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
"""Routes the floating vLLM e2e legs around releases that are broken upstream.

``e2e/vllm_known_bad.txt`` lists such releases, one ``<version> <reason>``
per line, spelled the way the server's ``/version`` reports them. A tag
that resolves to a listed version is skipped with a warning, unless the
tag itself names that version: a pin at a broken release stays red, only
``latest`` and other floating tags route around it. An entry goes inert
once the floating tag moves past it.

CLI, used by the e2e workflow:

    python e2e/known_bad.py version <base_url>          # prints it, or exits 1
    python e2e/known_bad.py skip-reason <tag> <version> # prints it and exits 0, or exits 1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping
from urllib.error import URLError
from urllib.request import urlopen

KNOWN_BAD = Path(__file__).resolve().parent / "vllm_known_bad.txt"


def load_known_bad(path: Path = KNOWN_BAD) -> dict[str, str]:
    """Version -> reason. A listed version without a reason is a malformed table."""
    known: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        version, _, reason = line.partition(" ")
        if not reason.strip():
            raise ValueError(f"{path}:{lineno}: {version} has no reason")
        known[version] = reason.strip()
    return known


def tag_names_version(tag: str, version: str) -> bool:
    """v0.32.0 and v0.32.0-arm64 name 0.32.0; latest names nothing."""
    return tag == f"v{version}" or tag.startswith(f"v{version}-")


def skip_reason(tag: str, version: str, known_bad: Mapping[str, str]) -> str | None:
    """Why the pass for ``tag``, resolved to ``version``, should be skipped, or None to run it."""
    if not version or tag_names_version(tag, version):
        return None
    return known_bad.get(version)


def resolved_version(base_url: str, timeout: float = 10.0) -> str | None:
    """The bare version a running vLLM server reports, or None if it did not say."""
    try:
        with urlopen(f"{base_url}/version", timeout=timeout) as response:
            payload = json.load(response)
    except (URLError, OSError, ValueError):
        return None
    version = payload.get("version") if isinstance(payload, dict) else None
    return version if isinstance(version, str) and version else None


def main(argv: list[str]) -> int:
    match argv:
        case ["version", base_url]:
            answer = resolved_version(base_url)
        case ["skip-reason", tag, version]:
            answer = skip_reason(tag, version, load_known_bad())
        case _:
            print("usage: known_bad.py version <base_url> | skip-reason <tag> <version>", file=sys.stderr)
            return 2
    if answer is None:
        return 1
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
