from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Iterator, Optional, List, Tuple
from pathlib import Path
import csv
import json
import logging
import time

logger = logging.getLogger(__name__)


class TraceEntry:
    """Represents a single trace entry with timing and token information."""

    def __init__(
        self,
        timestamp: float,
        input_tokens: int,
        output_tokens: int,
        prompt: Optional[str] = None,
        completion: Optional[str] = None,
    ):
        self.timestamp = timestamp
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.prompt = prompt
        self.completion = completion


class TraceReader(ABC):
    """Abstract base class for streaming trace readers."""

    @abstractmethod
    def stream_token_entries(self, file_path: Path) -> Iterator[Tuple[int, int]]:
        """Stream trace entries one by one"""
        raise NotImplementedError

    @abstractmethod
    def load_traces(self, file_path: Path) -> List[Tuple[float, int, int]]:
        """Load traces from file."""
        raise NotImplementedError


class AzurePublicDatasetReader(TraceReader):
    """Trace reader for Azure Public Dataset format."""

    def __init__(self) -> None:
        self.timestamp_format = "%Y-%m-%d %H:%M:%S.%f"
        self.traces = None

    def load_traces(self, file_path: Path) -> List[Tuple[float, int, int]]:
        """Load traces from file into memory."""
        if self.traces is not None:
            return self.traces
        logger.info(f"Loading traces from {file_path}")
        traces = []
        start_line = 1
        initial_timestamp: float = 0
        before = time.time()
        with open(file_path, "r", encoding="utf-8") as f:
            if self.has_header(file_path):
                start_line = 2
                next(f)
            for line_num, line in enumerate(f, start_line):
                try:
                    if line.strip():  # Skip empty lines
                        entry_data = line.split(",")
                        timestamp = self.parse_timestamp(entry_data[0])
                        if line_num == start_line:
                            initial_timestamp = timestamp
                        traces.append((timestamp - initial_timestamp, int(entry_data[1].strip()), int(entry_data[2].strip())))
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
        after = time.time()
        logger.info(f"Time taken to load traces: {after - before} seconds")
        return traces

    def stream_token_entries(self, file_path: Path) -> Iterator[Tuple[int, int]]:
        """Stream entries from AzurePublicDataset format"""
        start_line = 1
        with open(file_path, "r", encoding="utf-8") as f:
            if self.has_header(file_path):
                start_line = 2
                next(f)
            for line_num, line in enumerate(f, start_line):
                try:
                    if line.strip():  # Skip empty lines
                        entry_data = line.split(",")
                        yield int(entry_data[1].strip()), int(entry_data[2].strip())
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")

    def parse_timestamp(self, timestamp: str) -> float:
        """Parse timestamp from string to float."""

        raw_ts = timestamp.strip().strip('"')
        # Normalize to "YYYY-MM-DD HH:MM:SS.ff" in UTC
        ts = raw_ts.replace("T", " ").rstrip("Z").strip()
        if "." in ts:
            head, frac = ts.split(".", 1)
            # Keep only digits in fractional seconds and coerce to 2 digits
            frac_digits = "".join(ch for ch in frac if ch.isdigit())
            frac2 = (frac_digits[:2]).ljust(2, "0")
            ts_clean = f"{head}.{frac2}"
        else:
            ts_clean = f"{ts}.00"
        return datetime.strptime(ts_clean, self.timestamp_format).replace(tzinfo=timezone.utc).timestamp()

    def has_header(self, file_path: Path) -> bool:
        """Check if the file has a header."""
        with open(file_path, "r", encoding="utf-8") as f:
            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(f.read(2048))
            f.seek(0)
            return has_header


class DatasetTraceEntry:
    """Represents a single dataset trace entry with prompt and optional output length."""

    def __init__(self, text_input: str, output_length: Optional[int] = None):
        self.text_input = text_input
        self.output_length = output_length


class DatasetTraceReader:
    """Trace reader for DatasetTrace format (JSONL with text_input and optional output_length)."""

    def __init__(self) -> None:
        self.entries: Optional[List[DatasetTraceEntry]] = None

    def load_entries(self, file_path: Path) -> List[DatasetTraceEntry]:
        """Load dataset trace entries from JSONL file into memory."""
        if self.entries is not None:
            return self.entries

        logger.info(f"Loading dataset trace entries from {file_path}")
        entries: List[DatasetTraceEntry] = []
        before = time.time()

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    data = json.loads(line)
                    if "text_input" not in data:
                        raise ValueError(f"Missing required field 'text_input' in {file_path} line {line_num}")
                    text_input = data["text_input"]
                    output_length = data.get("output_length")
                    if output_length is not None:
                        output_length = int(output_length)
                    entries.append(DatasetTraceEntry(text_input, output_length))
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing JSON on line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")

        after = time.time()
        logger.info(f"Loaded {len(entries)} entries in {after - before:.3f} seconds")
        self.entries = entries
        return entries

    def stream_entries(self, file_path: Path) -> Iterator[DatasetTraceEntry]:
        """Stream dataset trace entries one by one."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    data = json.loads(line)
                    if "text_input" not in data:
                        raise ValueError(f"Missing required field 'text_input'")
                    text_input = data["text_input"]
                    output_length = data.get("output_length")
                    if output_length is not None:
                        output_length = int(output_length)
                    yield DatasetTraceEntry(text_input, output_length)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing JSON on line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
