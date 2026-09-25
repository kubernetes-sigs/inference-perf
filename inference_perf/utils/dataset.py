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
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def load_dataset_with_deadline(load: Callable[[], T], dataset_path: str, timeout: float | None) -> T:
    """Bound dataset loading without consuming the returned dataset or iterator."""
    # Match tokenizer loading: a daemon thread bounds waits on network/file I/O
    # that releases the GIL. It cannot cancel the load or interrupt GIL-holding code.
    result: list[T] = []
    error: list[Exception] = []

    def run() -> None:
        try:
            result.append(load())
        except Exception as exc:
            error.append(exc)

    logger.info("Loading dataset '%s'", dataset_path)
    thread = threading.Thread(target=run, name="dataset-load", daemon=True)
    thread.start()
    # None disables the deadline and waits for completion.
    thread.join(timeout=timeout)
    if thread.is_alive():
        raise TimeoutError(
            f"Loading dataset '{dataset_path}' did not finish within {timeout} seconds. "
            "This usually means the download from Hugging Face Hub is stuck "
            "(network issues or a Hub/CDN outage). Check connectivity to huggingface.co, "
            "try HF_HUB_DISABLE_XET=1 to surface the underlying download error, "
            "or pre-populate the HF cache. The deadline is configurable via "
            "'data.load_timeout' (null disables it)."
        )
    if error:
        raise error[0]
    if not result:
        raise RuntimeError(f"Dataset loader thread for '{dataset_path}' exited without a result or an error.")
    return result[0]
