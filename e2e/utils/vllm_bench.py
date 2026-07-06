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
"""
Run `vllm bench serve` against a server, for tool-parity comparison.

Design intent (per request): vllm is NOT a dependency of inference-perf. We
instead clone a *pinned* vllm at test time and invoke its bench CLI, and we
warn (loudly, non-fatally) if the pin has drifted too far from upstream so the
comparison does not silently rot against a stale tool.

Resolution order for *how* to invoke vllm, fastest first:
  1. $VLLM_BENCH_BIN  -- path to an already-installed `vllm` executable. Skips
     all clone/install work. The fast path for local iteration.
  2. A cached, isolated venv under the e2e cache dir with the pinned vllm clone
     installed into it. Heavy one-time setup (vllm pulls torch); cached across
     runs. This is the CI path.

Any failure to provision vllm results in a SkipReason, so the parity test
degrades gracefully (skips) exactly like the llm-d-inference-sim tests do when
the sim is absent -- it never hard-fails the suite on a missing tool.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- pin
#
# Bump these together. VLLM_PIN_DATE is the date the pin was last reviewed and
# drives the offline staleness warning (no network needed). The optional
# upstream check (see warn_if_pin_stale) is best-effort on top of this.
VLLM_PINNED_REF = "v0.10.0"  # TODO: confirm the tag whose `vllm bench serve` CLI we target
# Extra pip specs installed alongside the pinned vllm (space-separated; the CI
# workflow greps this constant too). vllm leaves transformers unbounded above,
# and transformers 5 removed tokenizer attrs (all_special_tokens_extended)
# that this vllm still reads at bench startup. Review when bumping the pin.
VLLM_COMPAT_PINS = "transformers<5"
VLLM_PIN_DATE = datetime.date(2026, 1, 15)
VLLM_STALENESS_WARN_DAYS = 180
VLLM_REPO_URL = "https://github.com/vllm-project/vllm.git"

# Cache lives outside the repo tree by default; override for CI persistence.
_DEFAULT_CACHE = Path(os.environ.get("VLLM_BENCH_CACHE_DIR", Path(tempfile.gettempdir()) / "inference-perf-vllm-bench"))


@dataclass
class VllmBenchResult:
    success: bool
    timed_out: bool
    return_code: int
    stdout: str
    result_json: Optional[Dict[str, Any]]  # parsed --save-result output, if produced


class VllmUnavailable(Exception):
    """Raised when vllm could not be provisioned; callers should pytest.skip."""


def warn_if_pin_stale(*, check_upstream: bool = False) -> None:
    """Emit a warning if the pinned vllm ref looks out of date.

    Offline check: how long since the pin was reviewed. Optional online check:
    compare the pinned tag against GitHub's latest release. Both are advisory --
    a stale pin still runs, it just gets a loud log line so the comparison does
    not quietly drift against an old vllm.
    """
    age_days = (datetime.date.today() - VLLM_PIN_DATE).days
    if age_days > VLLM_STALENESS_WARN_DAYS:
        logger.warning(
            "vllm pin %s was last reviewed %d days ago (> %d). Consider bumping "
            "VLLM_PINNED_REF and re-checking the bench CLI flags in scenario.py.",
            VLLM_PINNED_REF,
            age_days,
            VLLM_STALENESS_WARN_DAYS,
        )

    if not check_upstream:
        return
    try:  # best-effort; never fail the test on a network hiccup
        import urllib.request

        with urllib.request.urlopen("https://api.github.com/repos/vllm-project/vllm/releases/latest", timeout=5) as resp:
            latest = json.loads(resp.read()).get("tag_name")
        if latest and latest != VLLM_PINNED_REF:
            logger.warning("vllm pin %s is behind upstream latest release %s.", VLLM_PINNED_REF, latest)
    except Exception as e:  # noqa: BLE001 - advisory only
        logger.debug("upstream vllm staleness check skipped: %s", e)


# Env vars that redirect a Python process's module resolution. The e2e suite
# typically runs inside this repo's dev environment (nix devshell + venv, which
# exports PYTHONPATH entries for its own interpreter), while vllm runs under a
# separate interpreter. PYTHONPATH outranks the child's site-packages, so
# inheriting these makes vllm import packages built for the wrong CPython ABI
# (e.g. the devshell's torch) and crash on startup. LD_LIBRARY_PATH is left
# alone: CI's hosted Python needs its entry to find libpython.
_HOST_PYTHON_ENV_VARS = frozenset(
    {
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "NIX_PYTHONPATH",
        "VIRTUAL_ENV",
        "VIRTUAL_ENV_PROMPT",
    }
)


def _isolated_env() -> Dict[str, str]:
    """os.environ minus the vars that leak the host Python's packages into vllm's."""
    return {k: v for k, v in os.environ.items() if k not in _HOST_PYTHON_ENV_VARS}


# The `vllm` CLI cannot run `bench serve` on a machine with no accelerator:
# main() eagerly builds every subparser, and the `vllm serve` (server) one
# instantiates a default DeviceConfig, whose platform inference raises
# "Failed to infer device type" when no device is present -- true of CI
# runners and GPU-less workstations. The bench serve code itself is a pure
# HTTP client that never touches a device, so run it directly instead. This
# shim reproduces exactly what the CLI subcommand does
# (vllm/entrypoints/cli/benchmark/serve.py: add_cli_args + main), minus the
# unrelated broken subparsers.
_BENCH_SERVE_SHIM = (
    "import argparse\n"
    "try:\n"
    "    from vllm.utils import FlexibleArgumentParser as Parser\n"
    "except Exception:\n"
    "    Parser = argparse.ArgumentParser\n"
    "from vllm.benchmarks.serve import add_cli_args, main\n"
    "parser = Parser(description='vllm bench serve (direct module invocation)')\n"
    "add_cli_args(parser)\n"
    "main(parser.parse_args())\n"
)


def _bench_serve_cmd(vllm_bin: str, args: List[str]) -> List[str]:
    """Build the argv for `vllm bench serve <args>` semantics.

    Prefer the shim above under the vllm install's own interpreter (the
    `python` sibling of the `vllm` executable, present in any venv or
    setup-python install). Fall back to the real CLI when no sibling python
    exists, e.g. if $VLLM_BENCH_BIN points at a wrapper script.
    """
    resolved = shutil.which(vllm_bin) or vllm_bin
    python = Path(resolved).parent / "python"
    if python.exists():
        return [str(python), "-c", _BENCH_SERVE_SHIM, *args]
    return [vllm_bin, "bench", "serve", *args]


def _run(cmd: List[str]) -> "subprocess.CompletedProcess[str]":
    logger.debug("running: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=True, env=_isolated_env())


def ensure_vllm_bench_bin(cache_dir: Optional[Path] = None) -> str:
    """Return a path to a runnable `vllm` executable, or raise VllmUnavailable.

    Resolution:
      1. $VLLM_BENCH_BIN -> use it directly (the path CI uses: a vllm installed
         under a compatible interpreter). Fast, no build.
      2. $VLLM_BENCH_PROVISION truthy -> opt in to the HEAVY path: clone the
         pinned vllm and pip-install it (with torch) into a cached venv.
      3. otherwise -> raise VllmUnavailable so the parity test skips cleanly.

    The default is to skip, NOT to silently kick off a multi-GB build during a
    normal `pdm run test:e2e`. The heavy path is also gated on interpreter
    compatibility: vllm supports Python <=3.12, while this repo runs 3.14, so
    the in-tree interpreter usually cannot build it -- hence CI installs vllm
    under a separate 3.12 and points $VLLM_BENCH_BIN at it.
    """
    override = os.environ.get("VLLM_BENCH_BIN")
    if override:
        if shutil.which(override) or Path(override).exists():
            return override
        raise VllmUnavailable(f"$VLLM_BENCH_BIN set but not executable: {override}")

    if not os.environ.get("VLLM_BENCH_PROVISION"):
        raise VllmUnavailable(
            "no vllm available: set $VLLM_BENCH_BIN to a vllm executable, or "
            "$VLLM_BENCH_PROVISION=1 to clone+install the pinned vllm (heavy; "
            "needs a Python <=3.12 interpreter via $VLLM_PYTHON)"
        )

    cache = cache_dir or _DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    clone_dir = cache / f"vllm-{VLLM_PINNED_REF}"
    venv_dir = cache / f"venv-{VLLM_PINNED_REF}"
    vllm_bin = venv_dir / "bin" / "vllm"

    if vllm_bin.exists():
        return str(vllm_bin)

    if not shutil.which("git"):
        raise VllmUnavailable("git not available to clone vllm")

    # vllm does not yet publish wheels for Python 3.14; let callers point at a
    # compatible interpreter (e.g. python3.12) for the isolated venv.
    venv_python = os.environ.get("VLLM_PYTHON", "python")

    try:
        if not clone_dir.exists():
            logger.info("cloning pinned vllm %s (shallow) -> %s", VLLM_PINNED_REF, clone_dir)
            _run(["git", "clone", "--depth", "1", "--branch", VLLM_PINNED_REF, VLLM_REPO_URL, str(clone_dir)])

        logger.info("creating isolated venv for vllm at %s (python=%s)", venv_dir, venv_python)
        _run([venv_python, "-m", "venv", str(venv_dir)])
        pip = str(venv_dir / "bin" / "pip")
        # CPU-only client bench: no GPU needed to drive an OpenAI-compatible
        # server. This is still a large install (torch); it is cached above.
        _run([pip, "install", "--upgrade", "pip"])
        _run([pip, "install", str(clone_dir), *VLLM_COMPAT_PINS.split()])
    except subprocess.CalledProcessError as e:
        raise VllmUnavailable(f"failed to provision vllm: {e.stderr or e}") from e

    if not vllm_bin.exists():
        raise VllmUnavailable(f"vllm install completed but no executable at {vllm_bin}")
    return str(vllm_bin)


async def run_vllm_bench(
    args: List[str],
    *,
    vllm_bin: str,
    work_dir: Optional[Path] = None,
    result_filename: str = "vllm_bench_result.json",
    timeout_sec: Optional[int] = 300,
) -> VllmBenchResult:
    """Invoke `vllm bench serve <args>` and parse its --save-result JSON.

    `args` should already contain `--save-result --result-filename <path>`
    (Scenario.vllm_bench_args does this). We read that file back here.
    """
    wd = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="vllm-bench-e2e-"))
    wd.mkdir(parents=True, exist_ok=True)

    full = _bench_serve_cmd(vllm_bin, args)
    logger.debug("starting vllm bench: %s", " ".join(full))
    proc = await asyncio.create_subprocess_exec(
        *full,
        cwd=str(wd),
        env=_isolated_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    timed_out = False
    return_code = -1
    stdout = ""
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        stdout = out.decode()
        assert proc.returncode is not None
        return_code = proc.returncode
    except asyncio.TimeoutError:
        timed_out = True
        return_code = -9
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass

    # --result-filename may be absolute (Scenario passes one); fall back to wd.
    rf = Path(result_filename)
    result_path = rf if rf.is_absolute() else wd / rf
    result_json = None
    if result_path.exists():
        try:
            result_json = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("vllm bench result file present but unparseable: %s", result_path)

    return VllmBenchResult(
        success=(return_code == 0 and not timed_out),
        timed_out=timed_out,
        return_code=return_code,
        stdout=stdout,
        result_json=result_json,
    )
