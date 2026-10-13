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
"""Find guidellm and run one benchmark with it, for the tool-parity tests.

guidellm is not a dependency of inference-perf and this file does not make it
one. The tests use whatever guidellm the environment provides:

  1. $GUIDELLM_BIN: path to a `guidellm` executable.
  2. `guidellm` on PATH.
  3. Neither: raise GuidellmUnavailable, which the tests turn into a skip.

With GUIDELLM_REQUIRED=1 (set in CI) nothing skips: a missing guidellm, or one
that is not the pinned version, fails the tests. A merge gate whose comparison
tool silently skips gates nothing.

The pin is e2e/guidellm_requirements.in, locked with all of guidellm's own
dependencies in e2e/guidellm_requirements.txt. Install it into its own
environment, away from this repo's venv (guidellm pulls in its own pydantic,
httpx, datasets and torch):

    python3 -m venv /tmp/guidellm
    /tmp/guidellm/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r e2e/guidellm_requirements.txt
    export GUIDELLM_BIN=/tmp/guidellm/bin/guidellm
"""

import asyncio
import json
import logging
import os
import shutil
import signal
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_REQUIREMENTS_IN = Path(__file__).resolve().parent.parent / "guidellm_requirements.in"


# Reads the pinned version out of e2e/guidellm_requirements.in, so the pin has
# one home. A file containing the line "guidellm==0.8.0" gives "0.8.0".
def _pinned_version() -> str:
    for line in _REQUIREMENTS_IN.read_text(encoding="utf-8").splitlines():
        if line.startswith("guidellm=="):
            return line.removeprefix("guidellm==").strip()
    raise AssertionError(f"no guidellm== line in {_REQUIREMENTS_IN}")


# The guidellm version the scenario file and the report field names below were
# written against. Both have changed between guidellm releases, so re-read
# guidellm/schemas/benchmark/entrypoints.py (scenario fields) and
# guidellm/schemas/base/request_stats.py (per-request metrics) when bumping.
GUIDELLM_PINNED_VERSION = _pinned_version()


# True when GUIDELLM_REQUIRED is set to anything but "" or "0". CI sets it.
def guidellm_required() -> bool:
    return os.environ.get("GUIDELLM_REQUIRED", "0") not in ("", "0")


# Environment variables that tell a Python process where to look for packages.
# These tests run inside this repo's dev environment, which sets them for its
# own Python. guidellm runs under a different Python, and if it inherited them
# it would import this repo's packages instead of its own.
_HOST_PYTHON_ENV_VARS = frozenset({"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "NIX_PYTHONPATH", "VIRTUAL_ENV"})


class GuidellmUnavailable(Exception):
    """No guidellm executable was found. The tests catch this and skip."""


@dataclass
class GuidellmResult:
    success: bool  # exit code 0, not timed out, and a report was written
    timed_out: bool
    return_code: int
    stdout: str  # combined stdout/stderr
    # One dict per successful request, from the JSON report's
    # benchmarks[0].requests.successful. Latency fields are milliseconds.
    requests: List[Dict[str, Any]]


# Returns the path of the guidellm to run, or raises GuidellmUnavailable.
# With GUIDELLM_BIN=/opt/gl/bin/guidellm that path is returned if it exists.
def find_guidellm_bin() -> str:
    explicit = os.environ.get("GUIDELLM_BIN")
    if explicit:
        if not Path(explicit).is_file():
            raise GuidellmUnavailable(f"GUIDELLM_BIN={explicit} does not exist")
        return explicit
    found = shutil.which("guidellm")
    if found is None:
        raise GuidellmUnavailable("guidellm not found: set GUIDELLM_BIN or put guidellm on PATH")
    return found


def _isolated_env() -> Dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _HOST_PYTHON_ENV_VARS}


# Compares the installed guidellm with the pinned version. On a mismatch:
# with GUIDELLM_REQUIRED=1 it fails, otherwise it logs a warning and the
# different version still runs (the warning explains a later KeyError on a
# renamed report field).
async def check_version(guidellm_bin: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        guidellm_bin,
        "--version",
        env=_isolated_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    version = out.decode().strip()
    if not version.endswith(GUIDELLM_PINNED_VERSION):
        assert not guidellm_required(), f"guidellm reports {version!r}, CI requires the pinned {GUIDELLM_PINNED_VERSION}"
        logger.warning("guidellm reports %r, parity tests were written against %s", version, GUIDELLM_PINNED_VERSION)


async def run_guidellm(
    *,
    guidellm_bin: str,
    base_url: str,
    model: str,
    prompts: List[str],
    work_dir: Optional[Path] = None,
    timeout_sec: Optional[int] = 300,
) -> GuidellmResult:
    """Send each prompt once, one request at a time, as a streamed chat request.

    Three prompts means three requests, the second sent after the first
    finishes. Returns guidellm's own per-request metrics from its JSON report.
    """
    wd = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="guidellm-e2e-"))
    wd.mkdir(parents=True, exist_ok=True)
    report_path = wd / "guidellm_report.json"
    scenario_path = wd / "scenario.json"
    scenario = {
        "spec": {
            "backend": {
                "kind": "openai_http",
                "target": base_url,
                "model": model,
                "request_format": "/v1/chat/completions",
            },
            "profile": {"kind": "synchronous"},
            "data": [{"kind": "in_memory_item_list", "data": prompts, "column_name": "prompt"}],
            "constraints": [{"kind": "max_requests", "count": len(prompts)}],
            "outputs": [{"kind": "json", "path": str(report_path)}],
        }
    }
    scenario_path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")

    args = [guidellm_bin, "run", "--scenario", str(scenario_path), "--disable-console-interactive"]
    logger.debug("starting guidellm: %s", " ".join(args))
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(wd),
        env=_isolated_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        preexec_fn=os.setpgrp,
    )

    stdout = ""
    timed_out = False
    return_code = -1
    try:
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        stdout = stdout_bytes.decode()
        logger.info("guidellm status %s, output:\n%s", proc.returncode, textwrap.indent(stdout, "  | "))
        assert proc.returncode is not None
        return_code = proc.returncode
    except asyncio.exceptions.TimeoutError:
        timed_out = True
        return_code = -9
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            await proc.wait()
        except ProcessLookupError:
            pass

    requests: List[Dict[str, Any]] = []
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        requests = list(report["benchmarks"][0]["requests"]["successful"])

    return GuidellmResult(
        success=(return_code == 0) and (not timed_out) and report_path.is_file(),
        timed_out=timed_out,
        return_code=return_code,
        stdout=stdout,
        requests=requests,
    )
