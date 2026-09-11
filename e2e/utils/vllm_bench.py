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
"""Find (or install) vllm and run `vllm bench serve`, for the tool-parity tests.

vllm is not a dependency of inference-perf and this file does not make it one.
Instead the tests use whatever vllm the environment provides, or optionally
clone and install one specific pinned version into a throwaway venv. A warning
is logged (never an error) if that pin has not been reviewed in a long time, so
the comparison does not quietly go stale against an old vllm.

Where the vllm executable comes from, tried in this order:
  1. $VLLM_BENCH_BIN: path to a `vllm` you already have. Nothing to install.
     Use this for local work and in CI.
  2. $VLLM_BENCH_PROVISION=1: clone the pinned vllm and pip-install it (which
     pulls in torch, so it is a large download) into a venv under a cache
     directory. Reused on later runs.
  3. Neither set: raise VllmUnavailable, which the tests turn into a skip.

A missing vllm therefore skips the vllm-side tests. It never fails the suite,
the same way the llm-d-inference-sim tests skip when the sim is absent.
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
# Which vllm version the parity cases are written for. Bump these together.
# VLLM_PIN_DATE is when someone last checked the pin; it drives the "this pin
# is old" warning without any network access. warn_if_pin_stale can also
# compare against GitHub's latest release, best-effort.
#
# The case files (cases/*/vllm-bench.args) were written by reading this exact
# tag (tree 6d8d0a24c02bfd84d46b3016b865a44f048ae84b). What each flag means is
# defined in vllm/benchmarks/serve.py and vllm/benchmarks/datasets.py and has
# changed between vllm versions, so re-read both files when bumping.
VLLM_PINNED_REF = "v0.10.0"
# Extra packages to pip-install next to the pinned vllm (space-separated; the
# CI workflow reads this constant too). vllm does not cap its transformers
# version, and transformers 5 removed a tokenizer attribute
# (all_special_tokens_extended) that this vllm reads at bench startup, so
# without this pin the run crashes before sending a request. Review when
# bumping the vllm pin.
VLLM_COMPAT_PINS = "transformers<5"
VLLM_PIN_DATE = datetime.date(2026, 1, 15)
VLLM_STALENESS_WARN_DAYS = 180
VLLM_REPO_URL = "https://github.com/vllm-project/vllm.git"

# Where the cloned vllm and its venv live. Outside the repo by default; set
# VLLM_BENCH_CACHE_DIR to somewhere CI keeps between runs.
_DEFAULT_CACHE = Path(os.environ.get("VLLM_BENCH_CACHE_DIR", Path(tempfile.gettempdir()) / "inference-perf-vllm-bench"))


@dataclass
class VllmBenchResult:
    success: bool
    timed_out: bool
    return_code: int
    stdout: str
    result_json: Optional[Dict[str, Any]]  # parsed --save-result output, if produced


class VllmUnavailable(Exception):
    """No usable vllm was found or installed. The tests catch this and skip."""


def warn_if_pin_stale(*, check_upstream: bool = False) -> None:
    """Log a warning if the pinned vllm looks old. Never raises.

    Always: warn if VLLM_PIN_DATE is more than VLLM_STALENESS_WARN_DAYS ago.
    With check_upstream=True: also ask GitHub for vllm's latest release and warn
    if the pin is behind it (skipped quietly on any network problem). A stale
    pin still runs; the warning is just so nobody forgets it exists.
    """
    age_days = (datetime.date.today() - VLLM_PIN_DATE).days
    if age_days > VLLM_STALENESS_WARN_DAYS:
        logger.warning(
            "vllm pin %s was last reviewed %d days ago (> %d). Consider bumping "
            "VLLM_PINNED_REF and re-checking the bench CLI flags in the parity case files.",
            VLLM_PINNED_REF,
            age_days,
            VLLM_STALENESS_WARN_DAYS,
        )

    if not check_upstream:
        return
    try:  # advisory only; a network problem here must not fail the test
        import urllib.request

        with urllib.request.urlopen("https://api.github.com/repos/vllm-project/vllm/releases/latest", timeout=5) as resp:
            latest = json.loads(resp.read()).get("tag_name")
        if latest and latest != VLLM_PINNED_REF:
            logger.warning("vllm pin %s is behind upstream latest release %s.", VLLM_PINNED_REF, latest)
    except Exception as e:  # noqa: BLE001 - advisory only
        logger.debug("upstream vllm staleness check skipped: %s", e)


# Environment variables that tell a Python process where to look for packages.
# These tests usually run inside this repo's dev environment (nix devshell plus
# venv), which sets PYTHONPATH for its own Python. vllm runs under a different
# Python, and PYTHONPATH wins over that Python's own installed packages, so if
# vllm inherited it, it would import this repo's torch (built for a different
# Python) and crash on startup. So these are stripped from vllm's environment.
# LD_LIBRARY_PATH is kept: CI's Python needs it to find libpython.
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
    """A copy of os.environ with _HOST_PYTHON_ENV_VARS removed."""
    return {k: v for k, v in os.environ.items() if k not in _HOST_PYTHON_ENV_VARS}


# Why we do not just run `vllm bench serve`: on a machine with no GPU (CI
# runners, most laptops) the `vllm` command crashes at startup with "Failed to
# infer device type". That happens while it sets up the argument parser for
# every subcommand, including `vllm serve`, which needs a GPU. `bench serve`
# itself is only an HTTP client and never touches a GPU. So this tiny script
# imports just the bench serve code and calls it the same way the CLI would
# (see vllm/entrypoints/cli/benchmark/serve.py: add_cli_args + main), skipping
# the broken parts.
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
    """The command line to run: equivalent to `vllm bench serve <args>`.

    Preferred: run _BENCH_SERVE_SHIM with the `python` that sits next to the
    `vllm` executable (any venv or setup-python install has one). Fallback:
    the real `vllm bench serve` when there is no such python, for example when
    $VLLM_BENCH_BIN points at a wrapper script.
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
    """Return the path of a runnable `vllm` executable, or raise VllmUnavailable.

    Tried in order:
      1. $VLLM_BENCH_BIN set -> use it as is (this is what CI does: it installs
         vllm under a compatible Python and points here). Nothing to install.
      2. $VLLM_BENCH_PROVISION set -> clone the pinned vllm and pip-install it
         (with torch, so a multi-GB download) into a venv under cache_dir.
         Reused if it already exists.
      3. neither -> raise VllmUnavailable, so the tests skip.

    The default is to skip rather than silently start a multi-GB install during
    an ordinary `pdm run test:e2e`. Note vllm supports Python <=3.12 and this
    repo runs 3.14, so option 2 usually also needs $VLLM_PYTHON pointed at a
    3.12 interpreter; CI avoids that by using option 1.
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

    # vllm does not build on Python 3.14 yet, so the venv is created with
    # $VLLM_PYTHON (e.g. python3.12) when set.
    venv_python = os.environ.get("VLLM_PYTHON", "python")

    try:
        if not clone_dir.exists():
            logger.info("cloning pinned vllm %s (shallow) -> %s", VLLM_PINNED_REF, clone_dir)
            _run(["git", "clone", "--depth", "1", "--branch", VLLM_PINNED_REF, VLLM_REPO_URL, str(clone_dir)])

        logger.info("creating isolated venv for vllm at %s (python=%s)", venv_dir, venv_python)
        _run([venv_python, "-m", "venv", str(venv_dir)])
        pip = str(venv_dir / "bin" / "pip")
        # No GPU needed: bench serve is only an HTTP client. Still a large
        # install because vllm pulls in torch; the venv is kept for next time.
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
    """Run `vllm bench serve <args>` and return its exit status, output, and --save-result JSON.

    `args` must already contain `--save-result --result-filename <path>` (the
    parity test adds them); that file is read back into result_json. Runs in a
    fresh temp dir unless work_dir is given, and is killed after timeout_sec.
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

    # result_filename may be absolute; if relative, it is inside the work dir.
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
