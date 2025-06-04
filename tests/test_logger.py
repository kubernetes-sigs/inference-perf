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
