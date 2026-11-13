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
"""Fresh-interpreter imports and platform-specific multiprocessing context behavior."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


# A fresh interpreter must import the package without relying on the parent's module cache.
# This reproduces the Windows ForkServerContext ImportError before the platform split.
def test_package_import_in_fresh_interpreter() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import inference_perf"], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


# Windows must select spawn; POSIX must retain forkserver and its preloading process class.
# Constructing a process must preserve the importable class used as Worker's base.
def test_context_start_method_and_process_class() -> None:
    from inference_perf.loadgen.load_generator import Worker
    from inference_perf.utils.mp_context import MP_CONTEXT, PreloadedForkServerProcess

    assert MP_CONTEXT.get_start_method() == ("spawn" if sys.platform == "win32" else "forkserver")
    process = MP_CONTEXT.Process(target=os.getpid)
    assert isinstance(process, PreloadedForkServerProcess)
    assert issubclass(Worker, PreloadedForkServerProcess)
    process.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows spawn fallback")
# Queues, shared values, events, processes, and managers must work with the Windows context.
# Starting a real spawned process checks more than constructing the Process object alone.
def test_windows_context_primitives() -> None:
    from inference_perf.utils.mp_context import MP_CONTEXT

    queue = MP_CONTEXT.Queue()
    try:
        queue.put("queued")
        assert queue.get(timeout=10) == "queued"
    finally:
        queue.close()
        queue.join_thread()

    simple_queue = MP_CONTEXT.SimpleQueue()
    try:
        simple_queue.put("simple")
        assert simple_queue.get() == "simple"
    finally:
        simple_queue.close()

    value = MP_CONTEXT.Value("i", 7)
    assert value.value == 7
    value.value += 1
    assert value.value == 8
    event = MP_CONTEXT.Event()
    assert not event.is_set()
    event.set()
    assert event.wait(timeout=1)

    process = MP_CONTEXT.Process(target=os.getpid)
    try:
        process.start()
        process.join(timeout=30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        process.close()

    with MP_CONTEXT.Manager() as manager:
        shared = manager.list(["spawn"])
        assert list(shared) == ["spawn"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows spawn fallback")
# A fresh spawned subclass must receive queues, a shared value and an event from MP_CONTEXT.
# The child updates the value and signals the parent, proving these objects share a context.
def test_windows_context_objects_cross_process_boundary(tmp_path: Path) -> None:
    script = tmp_path / "probe_spawn_context.py"
    script.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            from multiprocessing.queues import Queue, SimpleQueue
            from multiprocessing.sharedctypes import Synchronized
            from multiprocessing.synchronize import Event

            from inference_perf.utils.mp_context import MP_CONTEXT, PreloadedForkServerProcess


            # Mirror Worker's inheritance from the public process class.
            class ProbeProcess(PreloadedForkServerProcess):
                pass


            # Read the parent's message, update shared state, then return it through SimpleQueue.
            def probe(queue: Queue[str], result: SimpleQueue[str], value: Synchronized[int], done: Event) -> None:
                message = queue.get(timeout=10)
                with value.get_lock():
                    value.value += 1
                result.put(message)
                done.set()


            if __name__ == "__main__":
                queue = MP_CONTEXT.Queue()
                result = MP_CONTEXT.SimpleQueue()
                value = MP_CONTEXT.Value("i", 7)
                done = MP_CONTEXT.Event()
                process = ProbeProcess(target=probe, args=(queue, result, value, done))
                try:
                    queue.put("shared spawn context")
                    process.start()
                    assert done.wait(timeout=60), "child did not signal completion"
                    process.join(timeout=10)
                    assert process.exitcode == 0, process.exitcode
                    assert value.value == 8, value.value
                    assert result.get() == "shared spawn context"
                    print("shared context verified")
                finally:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=10)
                    process.close()
                    queue.close()
                    queue.join_thread()
                    result.close()
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[3]), env.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120, check=False, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "shared context verified" in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Windows forkserver no-op")
@pytest.mark.parametrize("malloc_conf", [None, "background_thread:true"])
# Two Windows calls must neither start/configure a forkserver nor modify its environment.
# Check both absent and present JE_ARROW_MALLOC_CONF values without starting POSIX services.
def test_windows_start_forkserver_is_repeatable_noop(monkeypatch: pytest.MonkeyPatch, malloc_conf: str | None) -> None:
    from unittest.mock import patch

    from inference_perf.utils.mp_context import start_forkserver

    if malloc_conf is None:
        monkeypatch.delenv("JE_ARROW_MALLOC_CONF", raising=False)
    else:
        monkeypatch.setenv("JE_ARROW_MALLOC_CONF", malloc_conf)
    before = dict(os.environ)
    with (
        patch("multiprocessing.forkserver.ensure_running") as ensure_running,
        patch("multiprocessing.forkserver.set_forkserver_preload") as set_preload,
    ):
        start_forkserver()
        start_forkserver()
        ensure_running.assert_not_called()
        set_preload.assert_not_called()
    assert dict(os.environ) == before
