import json
import os
import asyncio
import aiofiles
import aiofiles.os
import tempfile
import yaml
import signal
import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Union

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a minimal benchmark run."""

    success: bool  # True if process exit code == 0 and not timed out
    timed_out: bool  # True if we hit timeout and killed the process
    return_code: int  # Raw process return code (or -9/-15 on kill)
    stdout: str  # Combined stdout/stderr text
    work_dir: Path  # Working directory used for the run
    reports: Optional[Dict[str, Any]]  # Parsed json for reports if present


async def _process_yaml_config(config: Union[str, Path, Dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config_input.yaml"

    # if config is a string pointing to an existing path, then convert it to
    # Path.
    if isinstance(config, str):
        try:
            await aiofiles.os.stat(config)
            config = Path(config)
        except Exception:
            pass

    # if config is a Path, then open it as a file.
    if isinstance(config, Path):
        async with aiofiles.open(config, mode="r") as file:
            config = await file.read()

    # if config is (still) a string, then directly parse it as YAML.
    if isinstance(config, str):
        config = yaml.safe_load(config)
        assert isinstance(config, dict)

    # Overwrite output path to temporary folder
    config["storage"] = {"local_storage": {"path": out_dir.as_posix()}}

    cfg_path.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return cfg_path


def _find_report_files(path: Path) -> Optional[List[Path]]:
    """Return the json reports files under path (if any)."""
    candidates = list(path.glob("**/*.json"))
    if not candidates:
        return None
    return candidates


async def run_benchmark_minimal(
    config: Union[str, Path, Dict[str, Any]],
    *,
    work_dir: Optional[Union[str, Path]] = None,
    executable: str = "inference-perf",
    timeout_sec: Optional[int] = 300,
    extra_env: Optional[Dict[str, str]] = None,
) -> BenchmarkResult:
    """
    Minimal wrapper:
      - materializes config to YAML in work_dir,
      - runs `inference-perf --config_file <config.yml>`,
      - returns success/failure, stdout text, and parsed report.json (if present).
    On timeout:
      - kills the spawned process,
      - marks `timed_out=True`, returns collected stdout up to kill.
    """
    wd = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="inference-perf-e2e-"))
    cfg_path = await _process_yaml_config(config, wd)

    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    args = [executable, "--config_file", str(cfg_path), "--log-level", "DEBUG"]
    logger.debug(f"starting inference-perf, {args=}")

    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(wd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        preexec_fn=os.setpgrp,  # use process groups
    )
    logger.debug("inference-perf started!")

    stdout = ""
    timed_out = False
    return_code = -1
    try:
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        stdout = stdout_bytes.decode()
        logger.info(f"benchmark status {proc.returncode}, output:\n{textwrap.indent(stdout, '  | ')}")
        assert proc.returncode is not None
        return_code = proc.returncode
    except asyncio.exceptions.TimeoutError:
        timed_out = True
        return_code = -9
    finally:
        try:
            # kill whole process group to ensure that forked workers are also
            # terminated.
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            # wait for process to finish cleaning up.
            await proc.wait()
        except ProcessLookupError:
            pass

    success = (return_code == 0) and (not timed_out)

    # Attempt to read report.json (optional)
    report_path = _find_report_files(wd)
    reports = {report.name: json.loads(report.read_text(encoding="utf-8")) for report in report_path} if report_path else None

    return BenchmarkResult(
        success=success,
        timed_out=timed_out,
        return_code=return_code,
        stdout=stdout,
        work_dir=wd,
        reports=reports,
    )
