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

Workers are started with ``forkserver`` rather than the platform-default
``fork``. Python 3.14 deprecates ``fork()`` from a multi-threaded process (the
main process runs an asyncio event loop and executor threads) because the child
can deadlock on locks held by threads that do not exist after the fork. The
``forkserver`` server process is single-threaded, so the forks that create
workers are safe.

Every multiprocessing object that crosses the worker boundary (queues, shared
values, events, barriers, managers, and the ``Process`` subclass itself) must be
created from this same context. Mixing objects from different start-method
contexts raises at pickle/transfer time.
"""

import multiprocessing as mp
from multiprocessing.context import BaseContext

# forkserver is available on Linux and macOS, which are the platforms this tool
# targets. The server is started lazily on first worker creation.
MP_CONTEXT: BaseContext = mp.get_context("forkserver")
