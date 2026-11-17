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
import asyncio
import os
import signal
import socket
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import aiohttp
import pytest

from inference_perf.observability.metrics.prometheus import DEFAULT_PORT
from utils.benchmark import BenchmarkResult, _process_yaml_config, run_benchmark_minimal
from utils.net import get_free_port

# A family the run registers at start. Finding it means the endpoint served this
# run's registry, not just that something answered on the port.
RUN_FAMILY = "inference_perf_run_elapsed_seconds"


# A 10 request mock run, with observability.metrics set to `metrics` when given.
def _config(metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "data": {"type": "mock"},
        "load": {"type": "constant", "stages": [{"rate": 2, "duration": 5}], "num_workers": 2},
        "api": {"type": "chat"},
        "server": {"type": "mock", "base_url": "http://0.0.0.0:8000"},
    }
    if metrics is not None:
        config["observability"] = {"metrics": metrics}
    return config


# The host's outbound IPv4 address, e.g. 10.1.0.4 on a CI runner, or None with only loopback.
# Connecting a UDP socket sends nothing, it only picks the interface.
def _lan_address() -> Optional[str]:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
        except OSError:
            return None
        address: str = s.getsockname()[0]
    return None if address.startswith("127.") else address


# Polls http://<host>:<port>/metrics every 0.1s until the run ends.
# Returns the body of the first 200, or None if nothing answered while the run was up.
async def _scrape_during(run: "asyncio.Task[BenchmarkResult]", host: str, port: int) -> Optional[str]:
    url = f"http://{host}:{port}/metrics"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1)) as session:
        while not run.done():
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.text()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(0.1)
    return None


# The mock run with observability.metrics.port set in the YAML, then with only
# --observability.metrics.port on the command line, both with linger_seconds: 0 so the test
# exits without waiting for the default linger. Expects /metrics to serve the run's families on
# that port from loopback and from the host's LAN address.
@pytest.mark.asyncio
@pytest.mark.parametrize("via", ["yaml", "flag"])
async def test_endpoint_serves_the_run_on_every_interface(via: str) -> None:
    port = get_free_port()
    if via == "yaml":
        run = asyncio.create_task(run_benchmark_minimal(_config({"port": port, "linger_seconds": 0})))
    else:
        flag = [
            "inference-perf",
            "--observability.metrics.port",
            str(port),
            "--observability.metrics.linger_seconds",
            "0",
        ]
        run = asyncio.create_task(run_benchmark_minimal(_config(), executable=flag))

    over_loopback = await _scrape_during(run, "127.0.0.1", port)
    lan = _lan_address()
    over_lan = await _scrape_during(run, lan, port) if lan else None
    result = await run

    assert result.success, result.stdout
    assert over_loopback is not None and RUN_FAMILY in over_loopback
    # A host with only loopback has no second interface to check.
    if lan:
        assert over_lan is not None and RUN_FAMILY in over_lan, f"not served on {lan}"


# The same mock run with no observability block.
# Expects nothing to answer on 9464, the port the endpoint would use if it started by default.
@pytest.mark.asyncio
async def test_endpoint_is_off_without_the_block() -> None:
    run = asyncio.create_task(run_benchmark_minimal(_config()))
    scraped = await _scrape_during(run, "127.0.0.1", DEFAULT_PORT)
    result = await run

    assert result.success, result.stdout
    assert scraped is None, f"something served /metrics on {DEFAULT_PORT}"


# The line the run prints once its reports are saved. The endpoint is still up after it only because of the linger.
REPORTS_SAVED = "=== END_SUMMARY ==="


# Starts the mock run with the given linger_seconds and a fixed metrics port, as its own process group.
# Returns (process, port, saved) where `saved` is set once the run has printed REPORTS_SAVED.
async def _start_run(work_dir: Path, linger_seconds: float) -> Tuple["asyncio.subprocess.Process", int, asyncio.Event]:
    port = get_free_port()
    config = _config({"port": port, "linger_seconds": linger_seconds})
    cfg_path = await _process_yaml_config(config, work_dir)
    proc = await asyncio.create_subprocess_exec(
        "inference-perf",
        "--config_file",
        str(cfg_path),
        cwd=str(work_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    saved = asyncio.Event()

    async def drain() -> None:
        assert proc.stdout is not None
        async for line in proc.stdout:
            if REPORTS_SAVED in line.decode(errors="replace"):
                saved.set()

    asyncio.create_task(drain())
    return proc, port, saved


# Kills whatever is left of the run's process group, so a failed assertion never leaves a benchmark running.
def _reap(proc: "asyncio.subprocess.Process") -> None:
    if proc.returncode is None:
        os.killpg(proc.pid, signal.SIGKILL)


# The mock run with linger_seconds 20, scraped only after the run has printed that its reports are saved.
# Expects /metrics to still answer then, with every stage completed: the final values outlive the run.
@pytest.mark.asyncio
async def test_endpoint_outlives_the_run_for_the_linger(tmp_path: Path) -> None:
    proc, port, saved = await _start_run(tmp_path, linger_seconds=20)
    try:
        await asyncio.wait_for(saved.wait(), timeout=120)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
            async with session.get(f"http://127.0.0.1:{port}/metrics") as resp:
                body = await resp.text()
        assert resp.status == 200
        assert "inference_perf_stages_completed 1.0" in body
        assert proc.returncode is None, "the process exited before the linger was over"
    finally:
        _reap(proc)


# The mock run with linger_seconds 120, sent SIGTERM once its reports are saved.
# Expects a clean exit within 30s, well short of the 120s linger: the signal ends the wait instead of being ignored.
@pytest.mark.asyncio
async def test_sigterm_ends_the_linger(tmp_path: Path) -> None:
    proc, _port, saved = await _start_run(tmp_path, linger_seconds=120)
    try:
        await asyncio.wait_for(saved.wait(), timeout=120)
        proc.send_signal(signal.SIGTERM)
        assert await asyncio.wait_for(proc.wait(), timeout=30) == 0
    finally:
        _reap(proc)


# The mock run with linger_seconds 0 and the metrics block set.
# Expects the process to exit on its own soon after the reports are saved, with nothing lingering.
@pytest.mark.asyncio
async def test_zero_linger_exits_when_the_reports_are_saved(tmp_path: Path) -> None:
    proc, _port, saved = await _start_run(tmp_path, linger_seconds=0)
    try:
        await asyncio.wait_for(saved.wait(), timeout=120)
        assert await asyncio.wait_for(proc.wait(), timeout=30) == 0
    finally:
        _reap(proc)
