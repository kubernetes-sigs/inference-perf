from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Iterator, Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import numpy as np
from inference_perf.config import Distribution
import logging
import time
logger = logging.getLogger(__name__)

class TraceEntry:
    """Represents a single trace entry with timing and token information."""
    def __init__(self, timestamp: float, input_tokens: int, output_tokens: int, 
                 prompt: Optional[str] = None, completion: Optional[str] = None):
        self.timestamp = timestamp
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.prompt = prompt
        self.completion = completion

class TraceReader(ABC):
    """Abstract base class for streaming trace readers."""
    
    @abstractmethod
    def stream_timestamp_entries(self, file_path: Path) -> Iterator[float]:
        """Stream entries from JSONL format without loading entire file."""
        raise NotImplementedError

    @abstractmethod
    def stream_token_entries(self, file_path: Path) -> Iterator[Tuple[int, int]]:
        """Stream trace entries one by one without loading entire file."""
        raise NotImplementedError
    
    @abstractmethod
    def load_traces(self, file_path: Path) -> List[Tuple[float, int, int]]:
        """Load traces from file."""
        raise NotImplementedError

class AzurePublicDatasetReader(TraceReader):
    """Streaming reader for Azure Public Dataset format."""

    def __init__(self):
        self.timestamp_format = "%Y-%m-%d %H:%M:%S.%f"
        self.traces = None

    
    def stream_timestamp_entries(self, file_path: Path) -> Iterator[float]:
        """Stream entries from CSV format without loading entire file."""
        logger.info(f"Streaming trace entries from {file_path}")
        has_header = True
        prev_timestamp = 0
        first_entry = True
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if has_header:
                    has_header = False
                    continue
                try:
                    if line.strip():  # Skip empty lines
                        entry_data = line.split(',')
                        timestamp = self.parse_timestamp(entry_data[0])
                        if first_entry:
                            prev_timestamp = timestamp
                            first_entry = False
                        print("yielded timestamp: "+str(timestamp - prev_timestamp))
                        yield (timestamp - prev_timestamp)
                        prev_timestamp = timestamp
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")

    def load_traces(self, file_path: Path) -> List[Tuple[float, int, int]]:
        if self.traces is not None:
            logger.info(f"Using cached traces")
            return self.traces
        logger.info(f"Loading traces from {file_path}")
        traces = []
        has_header = True
        first_entry = True
        prev_timestamp = 0
        before = time.time()
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if has_header:
                    has_header = False
                    continue
                try:
                    if line.strip():  # Skip empty lines
                        entry_data = line.split(',')
                        timestamp = self.parse_timestamp(entry_data[0])
                        if first_entry:
                            prev_timestamp = timestamp
                            first_entry = False
                        traces.append((timestamp - prev_timestamp, int(entry_data[1].strip()), int(entry_data[2].strip())))
                        prev_timestamp = timestamp
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
        after = time.time()
        logger.info(f"Time taken to load traces: {after - before} seconds")
        return traces
    
    def stream_token_entries(self, file_path: Path) -> Iterator[Tuple[int, int]]:
        """Stream entries from JSONL format without loading entire file."""
        logger.info(f"Streaming trace entries from {file_path}")
        has_header = True
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if has_header:
                    has_header = False
                    continue
                try:
                    if line.strip():  # Skip empty lines
                        entry_data = line.split(',')
                        yield int(entry_data[1].strip()), int(entry_data[2].strip())
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
    
    def parse_timestamp(self, timestamp: str) -> float:
        """Parse timestamp from string to float."""

        raw_ts = timestamp.strip().strip('"')
        # Normalize to "YYYY-MM-DD HH:MM:SS.ff" in UTC
        ts = raw_ts.replace('T', ' ').rstrip('Z').strip()
        if '.' in ts:
            head, frac = ts.split('.', 1)
            # Keep only digits in fractional seconds and coerce to 2 digits
            frac_digits = ''.join(ch for ch in frac if ch.isdigit())
            frac2 = (frac_digits[:2]).ljust(2, '0')
            ts_clean = f"{head}.{frac2}"
        else:
            ts_clean = f"{ts}.00"
        return datetime.strptime(ts_clean, self.timestamp_format).replace(tzinfo=timezone.utc).timestamp()

class JSONLinesReader(TraceReader):
    """Streaming reader for generic JSONL format."""
    
    def stream_timestamp_entries(self, file_path: Path) -> Iterator[float]:
        """Stream entries from JSONL format without loading entire file."""
        logger.info(f"Streaming trace entries from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip():  # Skip empty lines
                        entry_data = json.loads(line)
                        yield entry_data.get('timestamp', 0.0)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
    
    def stream_token_entries(self, file_path: Path) -> Iterator[Tuple[int, int]]:
        """Stream entries from JSONL format without loading entire file."""
        logger.info(f"Streaming trace entries from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip():  # Skip empty lines
                        entry_data = json.loads(line)
                        yield entry_data.get('input_tokens', 0), entry_data.get('output_tokens', 0)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")