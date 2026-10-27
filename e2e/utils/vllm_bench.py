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
pip-install one specific pinned version into a throwaway venv. A warning is
logged (never an error) if that pin has not been reviewed in a long time, so
the comparison does not quietly go stale against an old vllm.

Where the vllm executable comes from, tried in this order:
  1. $VLLM_BENCH_BIN: path to a `vllm` you already have. Nothing to install.
  2. $VLLM_BENCH_PROVISION=1: pip-install the pinned vllm from PyPI (which
     pulls in torch, so it is a large download) into a venv under a cache
     directory. Reused on later runs.
  3. Neither set: raise VllmUnavailable, which the tests turn into a skip.

A missing vllm therefore skips the vllm-side tests, the same way the
llm-d-inference-sim tests skip when the sim is absent. With VLLM_REQUIRED=1 a
missing vllm, or one that is not the pinned version, fails them instead.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from utils.host_env import isolated_env

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- pin
#
# Which vllm version the parity cases are written for. Bump these together.
# VLLM_PIN_DATE is when someone last checked the pin; it drives the "this pin
# is old" warning without any network access.
#
# The case files (cases/*/vllm-bench.args) were written by reading this exact
# tag (commit 6d8d0a24c02bfd84d46b3016b865a44f048ae84b). What each flag means is
# defined in vllm/benchmarks/serve.py and vllm/benchmarks/datasets.py and has
# changed between vllm versions, so re-read both files when bumping.
VLLM_PINNED_REF = "v0.10.0"
# Extra packages to pip-install next to the pinned vllm (space-separated).
# vllm does not cap its transformers version, and transformers 5 removed a
# tokenizer attribute (all_special_tokens_extended) that this vllm reads at
# bench startup, so without this pin the run crashes before sending a
# request. Review when bumping the vllm pin.
VLLM_COMPAT_PINS = "transformers<5"
VLLM_PIN_DATE = datetime.date(2026, 1, 15)
VLLM_STALENESS_WARN_DAYS = 180

# Where the provisioned vllm venv lives. Outside the repo by default; set
# VLLM_BENCH_CACHE_DIR to somewhere CI keeps between runs.
_DEFAULT_CACHE = Path(os.environ.get("VLLM_BENCH_CACHE_DIR", Path(tempfile.gettempdir()) / "inference-perf-vllm-bench"))


@dataclass
class VllmBenchResult:
    success: bool
    timed_out: bool
    return_code: int
    stdout: str


class VllmUnavailable(Exception):
    """No usable vllm was found or installed, or VLLM_REQUIRED is set and it is not the pin.

    The tests catch this and skip, or fail under VLLM_REQUIRED.
    """


def vllm_required() -> bool:
    """True when VLLM_REQUIRED is set to anything but "" or "0"."""
    return os.environ.get("VLLM_REQUIRED", "0") not in ("", "0")


def warn_if_pin_stale() -> None:
    """Log a warning if VLLM_PIN_DATE is more than VLLM_STALENESS_WARN_DAYS ago. Never raises.

    A stale pin still runs. The warning is just so nobody forgets it exists.
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
    `vllm` executable (any venv or setup-python install has one), once that
    python is known to have vllm installed. Fallback: the real
    `vllm bench serve`, for example when $VLLM_BENCH_BIN points at a wrapper
    script, or at a vllm whose directory holds some unrelated python.
    """
    python = _sibling_python(vllm_bin)
    if python is not None and _vllm_version_under(python) is not None:
        return [str(python), "-c", _BENCH_SERVE_SHIM, *args]
    return [vllm_bin, "bench", "serve", *args]


def _sibling_python(vllm_bin: str) -> Optional[Path]:
    """The `python` next to the `vllm` executable, or None when there is none.

    Symlinks are followed first, so a pipx-style /usr/local/bin/vllm link finds
    the python of the venv it points into, not /usr/local/bin/python.
    """
    found = shutil.which(vllm_bin) or vllm_bin
    python = Path(found).resolve().parent / "python"
    return python if python.exists() else None


def _run(cmd: List[str]) -> "subprocess.CompletedProcess[str]":
    logger.debug("running: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=True, env=isolated_env())


def _vllm_version_under(python: Path) -> Optional[str]:
    """The vllm version installed for `python`, or None if it has none or does not run.

    Read from package metadata, so the probe does not import vllm and its torch.
    """
    probe = "import importlib.metadata as m; print(m.version('vllm'))"
    try:
        out = subprocess.run([str(python), "-c", probe], capture_output=True, text=True, timeout=60, env=isolated_env())
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def installed_vllm_version(vllm_bin: str) -> Optional[str]:
    """The version of the vllm package behind vllm_bin, read by the python next to it, or None."""
    python = _sibling_python(vllm_bin)
    return None if python is None else _vllm_version_under(python)


def check_vllm_version(vllm_bin: str) -> None:
    """Warn when the vllm behind vllm_bin is not VLLM_PINNED_REF, or raise VllmUnavailable under VLLM_REQUIRED.

    The args files were written against the pinned tag's flag meanings, so
    another version can run them and send a different workload without error.
    A raise rather than an assert, so `python -O` cannot drop the gate.
    """
    pinned = VLLM_PINNED_REF.removeprefix("v")
    version = installed_vllm_version(vllm_bin)
    # A local build label such as +cpu does not change the bench CLI.
    if version is not None and version.split("+")[0] == pinned:
        return
    if version is None:
        problem = f"cannot read the vllm version behind {vllm_bin}, the pin is {pinned}"
    else:
        problem = f"vllm reports {version!r}, not the pinned {pinned}"
    if vllm_required():
        raise VllmUnavailable(problem)
    logger.warning("%s. The parity cases were written against the pin.", problem)


def ensure_vllm_bench_bin(cache_dir: Optional[Path] = None) -> str:
    """Return the path of a runnable `vllm` executable, or raise VllmUnavailable.

    Tried in order:
      1. $VLLM_BENCH_BIN set -> use it as is, for example a vllm installed
         under a separate Python 3.12. Nothing to install.
      2. $VLLM_BENCH_PROVISION set -> pip-install the pinned vllm from PyPI
         (with torch, so a multi-GB download) into a venv under cache_dir.
         Reused if it already exists.
      3. neither -> raise VllmUnavailable, so the tests skip.

    The default is to skip rather than silently start a multi-GB install during
    an ordinary `pdm run test:e2e`. The pinned vllm needs Python <=3.12 and
    this repo's environment is usually newer, so option 2 also needs
    $VLLM_PYTHON pointed at a 3.12 interpreter. Option 1 avoids that.

    PyPI rather than a clone of the tag: PyPI has a prebuilt wheel for Linux
    x86_64, while installing a clone is a source build that needs nvcc there
    (the build pins a CUDA torch). Other platforms get no wheel, so there pip
    still builds from the source release, which needs cmake, ninja and a C++
    toolchain.
    """
    override = os.environ.get("VLLM_BENCH_BIN")
    if override:
        if shutil.which(override) or Path(override).exists():
            return override
        raise VllmUnavailable(f"$VLLM_BENCH_BIN set but not executable: {override}")

    if not os.environ.get("VLLM_BENCH_PROVISION"):
        raise VllmUnavailable(
            "no vllm available: set $VLLM_BENCH_BIN to a vllm executable, or "
            "$VLLM_BENCH_PROVISION=1 to pip-install the pinned vllm (heavy; "
            "needs a Python <=3.12 interpreter via $VLLM_PYTHON)"
        )

    cache = cache_dir or _DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    venv_dir = cache / f"venv-{VLLM_PINNED_REF}"
    vllm_bin = venv_dir / "bin" / "vllm"

    if vllm_bin.exists():
        return str(vllm_bin)

    # The pinned vllm needs Python <=3.12, so the venv is created with
    # $VLLM_PYTHON (e.g. python3.12) when set.
    venv_python = os.environ.get("VLLM_PYTHON", "python")

    try:
        logger.info("creating isolated venv for vllm at %s (python=%s)", venv_dir, venv_python)
        _run([venv_python, "-m", "venv", str(venv_dir)])
        pip = str(venv_dir / "bin" / "pip")
        # No GPU needed: bench serve is only an HTTP client. Still a large
        # install because vllm pulls in torch; the venv is kept for next time.
        _run([pip, "install", "--upgrade", "pip"])
        _run([pip, "install", f"vllm=={VLLM_PINNED_REF.removeprefix('v')}", *VLLM_COMPAT_PINS.split()])
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
    timeout_sec: Optional[int] = 300,
) -> VllmBenchResult:
    """Run `vllm bench serve <args>` and return its exit status and output.

    Runs in a fresh temp dir unless work_dir is given, and is killed after
    timeout_sec. Nothing vllm prints or saves is read: the parity tests check
    what the absorber recorded.
    """
    wd = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="vllm-bench-e2e-"))
    wd.mkdir(parents=True, exist_ok=True)

    full = _bench_serve_cmd(vllm_bin, args)
    logger.debug("starting vllm bench: %s", " ".join(full))
    proc = await asyncio.create_subprocess_exec(
        *full,
        cwd=str(wd),
        env=isolated_env(),
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

    return VllmBenchResult(
        success=(return_code == 0 and not timed_out),
        timed_out=timed_out,
        return_code=return_code,
        stdout=stdout,
    )
