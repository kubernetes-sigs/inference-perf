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

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

_console: Optional[Console] = None


def get_console() -> Optional[Console]:
    """Shared Console so RichHandler logs don't tear the Progress bar. None in non-TTY."""
    return _console


def setup_logging(level: str) -> None:
    """
    Setup logging configuration for inference_perf.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global _console

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    handler: logging.Handler
    if sys.stdout.isatty():
        _console = Console()
        handler = RichHandler(
            console=_console,
            show_path=numeric_level == logging.DEBUG,
            rich_tracebacks=True,
            markup=False,
        )
        log_format = "%(name)s - %(message)s"
    else:
        _console = None
        handler = logging.StreamHandler(sys.stdout)
        if numeric_level == logging.DEBUG:
            log_format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        else:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[handler],
        force=True,
    )
