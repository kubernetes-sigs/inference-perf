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
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Run as a file, not with -c: the forkserver child finds probe() by re-running this path.
_PROBE = textwrap.dedent(
    """
    import os
    import sys

    from inference_perf.utils.mp_context import MP_CONTEXT, start_forkserver


    def probe(result):
        # A forkserver child's parent is the forkserver itself.
        with open(f"/proc/{os.getppid()}/status") as status:
            threads = next(int(line.split()[1]) for line in status if line.startswith("Threads:"))
        result.put((threads, "inference_perf.loadgen.load_generator" in sys.modules))


    if __name__ == "__main__":
        start_forkserver()
        result = MP_CONTEXT.SimpleQueue()
        worker = MP_CONTEXT.Process(target=probe, args=(result,))
        worker.start()
        worker.join(120)
        threads, preloaded = result.get()
        print(f"threads={threads} preloaded={preloaded}")
    """
)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
# A fresh interpreter calls start_forkserver() and starts one worker, which reads its parent's
# thread count from /proc. Prints threads=1 preloaded=True: the server has the package loaded
# and no jemalloc purge thread (threads=2 without it).
def test_forkserver_is_single_threaded_after_preload(tmp_path: Path) -> None:
    script = tmp_path / "probe_forkserver.py"
    script.write_text(_PROBE)

    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=180, check=True)

    assert "threads=1 preloaded=True" in out.stdout, out.stdout + out.stderr
