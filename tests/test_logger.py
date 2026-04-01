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
import pytest
from inference_perf.logger import setup_logging


@pytest.mark.parametrize(
    "level,expected_level",
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ],
)
def test_setup_logging_valid_levels(level: str, expected_level: int) -> None:
    """Test setup_logging with valid logging levels."""
    setup_logging(level)
    root_logger = logging.getLogger()
    assert root_logger.level == expected_level


def test_setup_logging_invalid_level() -> None:
    """Test setup_logging with an invalid logging level."""
    with pytest.raises(ValueError, match="Invalid log level: INVALID"):
        setup_logging("INVALID")
