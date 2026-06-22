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
"""A file-based counting semaphore keyed by contention class.

One contention class == one (kubeconfig, hardware-requirement) pair. Capacity ==
matching node count. Tests in the same class acquire a slot before deploying and
release it on teardown, so they queue when they outnumber the nodes but reuse the
hardware sequentially. Distinct classes share nothing and run fully in parallel.

File-based (atomic O_EXCL slot files) so it holds across pytest-xdist workers, not
just threads. No third-party dependency.
"""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_ROOT = Path(tempfile.gettempdir()) / "inference-perf-optional-slots"


def class_key(kubeconfig: str | None, requirement: str) -> str:
    signature = requirement + "|" + (kubeconfig or "<ambient>")
    return hashlib.sha1(signature.encode()).hexdigest()[:16]


@contextmanager
def acquire_slot(key: str, capacity: int, *, poll: float = 2.0) -> Iterator[int]:
    """Block until one of ``capacity`` slots for ``key`` is free, then hold it."""
    class_dir = _ROOT / key
    class_dir.mkdir(parents=True, exist_ok=True)
    held: Path | None = None
    try:
        while held is None:
            for slot in range(max(capacity, 1)):
                candidate = class_dir / f"slot-{slot}.lock"
                try:
                    fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except OSError as exc:
                    if exc.errno == errno.EEXIST:
                        continue  # taken, try the next slot
                    raise
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                held = candidate
                break
            if held is None:
                time.sleep(poll)
        yield int(held.stem.split("-")[1])
    finally:
        if held is not None:
            held.unlink(missing_ok=True)
