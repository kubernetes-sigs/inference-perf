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
"""YAML emission for converted configs.

The emitted file carries a comment block recording what was converted, from
which tool version, and every assumption made, so the config is auditable on
its own. Only fields the conversion actually derived are emitted
(``exclude_unset``): inference-perf defaults stay owned by inference-perf,
except where a table row emits a field explicitly because the peer default
disagrees with the inference-perf default (``server.ignore_eos`` and
``api.streaming``). Credentials are never written: a converted file is meant
to be checked in next to a parity case, so ``server.api_key`` stays unset and
the aiperf frontend records ``--api-key`` as an assumption to satisfy on the
run host.
"""

import shlex
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import List

import yaml

from inference_perf.tools.convert.model import Conversion


def _converter_version() -> str:
    try:
        return version("inference-perf")
    except PackageNotFoundError:
        return "unknown"


def _wrap(text: str, width: int, first_prefix: str, cont_prefix: str) -> List[str]:
    words = text.split()
    if not words:
        return [first_prefix.rstrip()]
    lines: List[str] = []
    current = first_prefix + words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) > width:
            lines.append(current)
            current = cont_prefix + word
        else:
            current = f"{current} {word}"
    lines.append(current)
    return lines


def comment_block(conversion: Conversion) -> str:
    """The human-readable conversion record, prepended to the YAML."""
    date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: List[str] = [
        "# Converted by inference-perf-convert (#755)",
        f"#   source : {conversion.source_tool} @ {conversion.source_version}",
    ]
    lines.extend(_wrap(shlex.join(conversion.argv), 100, "#   argv   : ", "#            "))
    lines.append(f"#   date   : {date}   converter: inference-perf {_converter_version()}")
    if conversion.assumptions:
        lines.append("# assumptions:")
        for item in conversion.assumptions:
            lines.extend(_wrap(item, 100, "#   - ", "#     "))
    if conversion.no_equivalents:
        lines.append("# NO EQUIVALENT:")
        for item in conversion.no_equivalents:
            lines.extend(_wrap(item, 100, "#   - ", "#     "))
    if conversion.dropped:
        lines.extend(_wrap(", ".join(conversion.dropped), 100, "# dropped (workload-neutral): ", "#   "))
    return "\n".join(lines) + "\n"


def emit_yaml(conversion: Conversion) -> str:
    """Render a successful conversion as commented YAML."""
    if conversion.config is None:
        raise ValueError("cannot emit a refused conversion")
    dumped = conversion.config.model_dump(mode="json", exclude_unset=True)
    body = yaml.safe_dump(dumped, sort_keys=False, default_flow_style=False)
    return comment_block(conversion) + body
