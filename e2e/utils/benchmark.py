import json
import os
import shlex
import subprocess
import tempfile
import yaml
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Union

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a minimal benchmark run."""

    success: bool  # True if process exit code == 0 and not timed out
    timed_out: bool  # True if we hit timeout and killed the process
    returncode: int  # Raw process return code (or -9/-15 on kill)
    stdout: str  # Combined stdout/stderr text
    work_dir: Path  # Working directory used for the run
    reports: Optional[Dict[str, Any]]  # Parsed json for reports if present


def _process_yaml_config(config: Union[str, Path, Dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config_input.yaml"

    if isinstance(config, (str, Path)):
        src = Path(config)
        if not src.exists():
            raise FileNotFoundError(f"Config file not found: {src}")
        config = yaml.safe_load(src.read_text(encoding="utf-8"))

    # Overwrite output path to temporaty folder
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


def run_benchmark_minimal(
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
    cfg_path = _process_yaml_config(config, wd)

    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    cmd = f"{shlex.quote(executable)} --config_file {shlex.quote(str(cfg_path))} --log-level DEBUG"

    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(wd),
            env=env,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
        )
        stdout = proc.stdout
        return_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = e.stdout
        return_code = -9

    success = (return_code == 0) and (not timed_out)

    logger.info("Benchmark output:\n%s", stdout)

    # Attempt to read report.json (optional)
    report_path = _find_report_files(wd)
    reports = {report.name: json.loads(report.read_text(encoding="utf-8")) for report in report_path} if report_path else None

    return BenchmarkResult(
        success=success,
        timed_out=timed_out,
        returncode=return_code,
        stdout=stdout or "",
        work_dir=wd,
        reports=reports,
    )
