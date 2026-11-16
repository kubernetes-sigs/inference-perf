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


# Cross-platform test: MP_CONTEXT primitives work on Linux too
@pytest.mark.skipif(sys.platform == "win32", reason="Linux POSIX path")
def test_posix_context_primitives() -> None:
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
        shared = manager.list(["forkserver"])
        assert list(shared) == ["forkserver"]
