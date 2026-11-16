# Copyright 2025 The Kubernetes Authors.
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
"""Lets whoever sends a request learn when its first token arrives.

The stream parser sits several calls below the load generator, behind the
client and the API data classes. A context variable carries the callback down
without each of them passing it on: an asyncio task has its own context, so a
callback set in the task that sends a request is seen only by that request.
"""

from contextvars import ContextVar
from typing import Callable, Optional

_ON_FIRST_TOKEN: ContextVar[Optional[Callable[[], None]]] = ContextVar("inference_perf_on_first_token", default=None)


def watch_first_token(callback: Callable[[], None]) -> None:
    """Have ``callback`` called when the request sent from the current task receives its first token."""
    _ON_FIRST_TOKEN.set(callback)


def notify_first_token() -> None:
    """Report the first token of the response being read in the current task."""
    callback = _ON_FIRST_TOKEN.get()
    if callback is not None:
        callback()
