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

import multiprocessing as mp
import os
from multiprocessing import forkserver
from multiprocessing.context import ForkServerContext
from typing import Dict, Optional

# forkserver is available on Linux and macOS, which are the platforms this tool
# targets. start_forkserver() starts the server; otherwise the first worker or
# manager start does, without the settings below.
MP_CONTEXT: ForkServerContext = mp.get_context("forkserver")

# Imported once in the server so workers fork with the package loaded instead of
# importing it again each (seconds and hundreds of MB per worker). "__main__" is
# the stdlib default and stays.
_PRELOAD = ["__main__", "inference_perf.loadgen.load_generator"]

# Those imports start native threads: OpenBLAS sizes a pool to the core count when
# numpy loads, and pyarrow's bundled jemalloc starts a purge thread. Either makes the
# server multi-threaded and every worker fork unsafe again. Workers inherit these
# settings and do no BLAS work, and pyarrow's default pool is not jemalloc.
_SERVER_ENV = {"OPENBLAS_NUM_THREADS": "1", "JE_ARROW_MALLOC_CONF": "background_thread:false"}


def start_forkserver() -> None:
    """Start the forkserver with the package preloaded and no native threads, if it is not running."""
    MP_CONTEXT.set_forkserver_preload(_PRELOAD)
    saved: Dict[str, Optional[str]] = {key: os.environ.get(key) for key in _SERVER_ENV}
    os.environ.update(_SERVER_ENV)
    try:
        forkserver.ensure_running()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
