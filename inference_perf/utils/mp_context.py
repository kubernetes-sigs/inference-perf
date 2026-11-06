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
"""Shared multiprocessing context for the load generator.

Workers are started with ``forkserver`` rather than ``fork``. The main process
runs an asyncio event loop and executor threads, and a child forked from it can
deadlock on a lock held by a thread that does not exist after the fork. Python
3.12 added a DeprecationWarning for that fork and 3.14 made ``forkserver`` the
Linux default. The ``forkserver`` server process is single-threaded, so the
forks that create workers are safe.

Every multiprocessing object that crosses the worker boundary (queues, shared
values, events, barriers, managers, and the ``Process`` subclass itself) must be
created from this same context. Mixing objects from different start-method
contexts raises at pickle/transfer time.
"""

import os
from multiprocessing import forkserver
from multiprocessing.context import ForkServerContext, ForkServerProcess
from typing import Dict, Optional

# Imported once in the server so workers fork with the package loaded instead of
# importing it again each (seconds and hundreds of MB per worker). "__main__" is
# the stdlib default and stays.
_PRELOAD = ["__main__", "inference_perf.loadgen.load_generator"]

# The preload imports pyarrow, whose bundled jemalloc starts a purge thread that keeps
# running across fork, so the server would fork every worker from a multi-threaded
# process again. pyarrow's default pool is not jemalloc, so nothing needs that thread.
# (numpy's OpenBLAS pool also starts on import but stops itself before each fork.)
_SERVER_ENV = {"JE_ARROW_MALLOC_CONF": "background_thread:false"}


_started = False


def start_forkserver() -> None:
    """Start the forkserver with the package preloaded and no native threads, once per process."""
    global _started
    # Later calls return here so a running process never rewrites os.environ for nothing.
    if _started:
        return
    forkserver.set_forkserver_preload(_PRELOAD)
    saved: Dict[str, Optional[str]] = {key: os.environ.get(key) for key in _SERVER_ENV}
    os.environ.update(_SERVER_ENV)
    try:
        forkserver.ensure_running()
        _started = True
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class PreloadedForkServerProcess(ForkServerProcess):
    """A forkserver Process that starts the server with start_forkserver() first."""

    # Whichever process starts first launches the one server per interpreter, and a server
    # launched without the preload cannot gain it later.
    def start(self) -> None:
        start_forkserver()
        super().start()


class _PreloadedForkServerContext(ForkServerContext):
    # SyncManager.start() builds its server process from ctx.Process, so managers get it too.
    Process = PreloadedForkServerProcess


# forkserver is available on Linux and macOS, which are the platforms this tool targets.
MP_CONTEXT: ForkServerContext = _PreloadedForkServerContext()
