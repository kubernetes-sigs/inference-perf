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
End-to-end test of a run that is killed after it has finished a stage (#817).

An OOM kill is a SIGKILL: no signal handler and no end-of-run report pass get
to run. Reports used to be written only by that pass, so a run killed in stage
N lost stages 0..N-1 along with it. Each stage's own reports are now written
when the stage ends. This test drives the real CLI against the mock server
through a short stage 0 into a long stage 1, SIGKILLs the whole process group
during stage 1, and checks what the run left on disk.

Needs no external server, so it never skips: the mock client answers every
request after about a second.
"""

import asyncio
import json
import os
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

from utils.benchmark import BenchmarkResult, run_benchmark_minimal

STAGE_0_RATE = 5
STAGE_0_DURATION = 2
STAGE_0_REQUESTS = STAGE_0_RATE * STAGE_0_DURATION
# Far longer than the test, so the run can only end by being killed.
STAGE_1_DURATION = 600
STAGE_INTERVAL = 1.0

STAGE_0_LIFECYCLE = "stage_0_lifecycle_metrics.json"
STAGE_0_PARTIAL = "inference-perf.partial.stage_0.yaml"
CONFIG_REPORT = "config.yaml"
# Saved in this order in one batch, config.yaml last.
STAGE_0_REPORTS = (STAGE_0_LIFECYCLE, STAGE_0_PARTIAL, CONFIG_REPORT)

PID_FILE = "inference-perf.pid"
# bash writes its pid, then execs the CLI in its place, so the pid is the CLI's.
# run_benchmark_minimal starts it in a new process group, so it is also the id
# of the group the CLI's workers belong to.
EXECUTABLE = ["bash", "-c", f'echo $$ > {PID_FILE}; exec inference-perf "$@"', "inference-perf"]

# Startup, stage 0, its last requests' mock latency and the report write take
# about 6s on a laptop; the rest is headroom for a loaded CI machine.
REPORTS_DEADLINE_SEC = 30.0
# Stage 1 starts once both the stage interval and stage 0's report write are
# done, and config.yaml is the last file that write saves.
STAGE_1_START_WAIT_SEC = STAGE_INTERVAL + 1.5
POLL_SEC = 0.1
# Backstop for run_benchmark_minimal, which SIGTERMs the group when it expires.
BENCHMARK_TIMEOUT_SEC = 45
REAP_TIMEOUT_SEC = 15.0


def _config() -> Dict[str, Any]:
    return {
        "data": {"type": "mock"},
        "load": {
            "type": "constant",
            "interval": STAGE_INTERVAL,
            "stages": [
                {"rate": STAGE_0_RATE, "duration": STAGE_0_DURATION},
                {"rate": 1, "duration": STAGE_1_DURATION},
            ],
            # Workers hand their metrics to the main process through a queue,
            # which is the path where a stage's metrics can still be in transit
            # when the stage ends.
            "num_workers": 2,
        },
        "api": {"type": "completion"},
        "server": {"type": "mock", "base_url": "http://127.0.0.1:8000"},
        "report": {
            "request_lifecycle": {
                "summary": True,
                "per_stage": True,
                "per_request": True,
            },
        },
    }


def _load(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".yaml":
        return yaml.safe_load(text)
    return json.loads(text)


def _parses(path: Path) -> bool:
    try:
        return bool(_load(path))
    except (OSError, ValueError, yaml.YAMLError):
        return False


def _read_pid(work_dir: Path) -> Optional[int]:
    try:
        return int((work_dir / PID_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Not written yet, or caught between bash creating and filling it.
        return None


def _live_group_members(pgid: int) -> List[int]:
    """Pids of the processes in group `pgid` that have not exited (zombies don't count).

    Reads /proc, so outside Linux it finds nothing and the leak check is skipped.
    """
    members = []
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            # "pid (comm) state ppid pgrp ...": comm can contain spaces and parens.
            fields = stat_path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if fields[2] == str(pgid) and fields[0] != "Z":
            members.append(int(stat_path.parent.name))
    return members


async def _wait_for_stage_0_reports(work_dir: Path, bench: "asyncio.Task[BenchmarkResult]") -> None:
    """Polls until every stage 0 report is on disk and parses, while the run is still going."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + REPORTS_DEADLINE_SEC
    while True:
        if bench.done():
            result = bench.result()
            pytest.fail(
                f"inference-perf exited with code {result.return_code} before stage 0's reports were written; "
                f"output tail:\n{result.stdout[-4000:]}"
            )
        missing = [name for name in STAGE_0_REPORTS if not _parses(work_dir / name)]
        if not missing:
            return
        if loop.time() > deadline:
            pytest.fail(
                f"{missing} not written within {REPORTS_DEADLINE_SEC:.0f}s of starting the run, although stage 0 "
                f"lasts {STAGE_0_DURATION}s and stage 1 ({STAGE_1_DURATION}s) is still running: stage 0's reports "
                "are not written when the stage ends, so a kill now would lose them"
            )
        await asyncio.sleep(POLL_SEC)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_stage_reports_survive_sigkill_in_later_stage(tmp_path: Path) -> None:
    """
    SIGKILLs a run in stage 1 and asserts stage 0's reports survive, complete
    and parseable, with no trace of stage 1 or of an end-of-run pass. This is
    the OOM kill from #817, which used to leave nothing behind.
    """
    bench = asyncio.create_task(
        run_benchmark_minimal(
            _config(),
            work_dir=tmp_path,
            executable=EXECUTABLE,
            timeout_sec=BENCHMARK_TIMEOUT_SEC,
        )
    )
    pid: Optional[int] = None
    try:
        await _wait_for_stage_0_reports(tmp_path, bench)
        # Kill in stage 1, not in the gap between the stages.
        await asyncio.sleep(STAGE_1_START_WAIT_SEC)
        assert not bench.done(), "inference-perf exited during stage 1 instead of running it"
        pid = _read_pid(tmp_path)
        assert pid is not None, f"{PID_FILE} was not written"
        # The whole group, workers included, as the OOM killer would.
        os.killpg(pid, signal.SIGKILL)
        result = await asyncio.wait_for(bench, timeout=REAP_TIMEOUT_SEC)
    finally:
        if not bench.done():
            # Something above failed: tear the run down so no process outlives the test.
            pid = pid or _read_pid(tmp_path)
            if pid is not None:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                bench.cancel()
            try:
                await asyncio.wait_for(bench, timeout=REAP_TIMEOUT_SEC)
            except (asyncio.CancelledError, Exception):
                pass

    assert pid is not None
    assert not result.timed_out, f"the run was stopped by the benchmark timeout, not killed by the test:\n{result.stdout}"
    assert result.return_code == -signal.SIGKILL, f"expected the run to die of SIGKILL, got {result.return_code}"
    assert "Stage 1 - run started" in result.stdout, "the run was killed before stage 1 started"

    loop = asyncio.get_running_loop()
    reap_deadline = loop.time() + REAP_TIMEOUT_SEC
    while _live_group_members(pid) and loop.time() < reap_deadline:
        await asyncio.sleep(POLL_SEC)
    leaked = _live_group_members(pid)
    assert not leaked, f"processes {leaked} of the killed run are still alive"

    stage_report = _load(tmp_path / STAGE_0_LIFECYCLE)
    # Written after every stage 0 request's metric reached the main process,
    # so the copy that survives counts the whole stage.
    assert stage_report["load_summary"]["count"] == STAGE_0_REQUESTS
    assert stage_report["successes"]["count"] == STAGE_0_REQUESTS

    partial = _load(tmp_path / STAGE_0_PARTIAL)
    assert partial["results"]["request_performance"]["aggregate"]["requests"]["total"] == STAGE_0_REQUESTS

    # The config goes out with the stage's reports, so what a dead run leaves
    # behind says what was run.
    config_report = _load(tmp_path / CONFIG_REPORT)
    assert [stage["duration"] for stage in config_report["load"]["stages"]] == [STAGE_0_DURATION, STAGE_1_DURATION]

    files = sorted(p.name for p in tmp_path.iterdir())
    assert not [name for name in files if "stage_1" in name], f"stage 1 never finished, yet left reports: {files}"
    # Run-wide reports belong to the end-of-run pass, which a SIGKILL skips.
    assert "summary_lifecycle_metrics.json" not in files
    assert "per_request_lifecycle_metrics.json" not in files
    # Reports are written aside and renamed into place.
    assert not list(tmp_path.rglob("*.tmp")), f"leftover temp files: {files}"
