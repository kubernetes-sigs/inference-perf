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
"""Tests for the forkserver that load generator workers and managers start from."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

if sys.platform == "win32":
    pytest.skip("forkserver is POSIX-only; Windows uses spawn", allow_module_level=True)

from inference_perf.loadgen.load_generator import Worker
from inference_perf.utils.mp_context import MP_CONTEXT

# Run as a file, not with -c: the forkserver child finds probe() by re-running this path.
# Nothing from inference_perf is imported at the top: the server preloads "__main__" by
# default, and a top-level import would load the package there whatever start_forkserver() did.
_PROBE = textwrap.dedent(
    """
    import os
    import sys


    def probe(result):
        # A forkserver child's parent is the forkserver itself.
        with open(f"/proc/{os.getppid()}/status") as status:
            threads = next(int(line.split()[1]) for line in status if line.startswith("Threads:"))
        # The preload imports pyarrow, so its native library is mapped into the server.
        with open(f"/proc/{os.getppid()}/maps") as maps:
            preloaded = "libarrow" in maps.read()
        result.put((threads, preloaded))


    if __name__ == "__main__":
        from inference_perf.utils.mp_context import MP_CONTEXT, start_forkserver

        first = sys.argv[1]
        if first == "start_forkserver":
            start_forkserver()
        elif first == "manager":
            MP_CONTEXT.Manager()
        result = MP_CONTEXT.SimpleQueue()
        worker = MP_CONTEXT.Process(target=probe, args=(result,))
        worker.start()
        worker.join(120)
        threads, preloaded = result.get()
        print(f"threads={threads} preloaded={preloaded}")
    """
)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
@pytest.mark.parametrize("first", ["start_forkserver", "process", "manager"])
# A fresh interpreter starts the forkserver one of three ways: start_forkserver(), a plain
# MP_CONTEXT.Process, or MP_CONTEXT.Manager(). A worker then reads the server from /proc. Every
# way prints threads=1 preloaded=True (threads=2 without the jemalloc env, preloaded=False if
# a process launched the server before start_forkserver() ran).
def test_forkserver_starts_preloaded_and_single_threaded(tmp_path: Path, first: str) -> None:
    script = tmp_path / "probe_forkserver.py"
    script.write_text(_PROBE)

    out = subprocess.run([sys.executable, str(script), first], capture_output=True, text=True, timeout=180, check=True)

    assert "threads=1 preloaded=True" in out.stdout, out.stdout + out.stderr


# Worker and the processes MP_CONTEXT builds share the class whose start() runs start_forkserver().
# Fails if Worker goes back to subclassing the stdlib ForkServerProcess.
def test_worker_starts_through_the_preloading_process() -> None:
    assert issubclass(Worker, MP_CONTEXT.Process)
