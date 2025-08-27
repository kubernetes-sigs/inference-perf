from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Iterator, Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import numpy as np
from inference_perf.config import Distribution
import logging

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

class TraceStreamReader(ABC):
    """Abstract base class for streaming trace readers."""
    
    @abstractmethod
    def stream_timestamp_entries(self, file_path: Path) -> Iterator[float]:
        """Stream entries from JSONL format without loading entire file."""
        raise NotImplementedError

    @abstractmethod
    def stream_token_entries(self, file_path: Path) -> Iterator[Tuple[int, int]]:
        """Stream trace entries one by one without loading entire file."""
        raise NotImplementedError

class AzurePublicDatasetReader(TraceStreamReader):
    """Streaming reader for Azure Public Dataset format."""

    def __init__(self):
        self.timestamp_format = "%Y-%m-%d %H:%M:%S.%f"

    
    def stream_timestamp_entries(self, file_path: Path) -> Iterator[float]:
        """Stream entries from CSV format without loading entire file."""
        logger.info(f"Streaming trace entries from {file_path}")
        has_header = True
        initial = 0
        first_entry = True
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if has_header:
                    has_header = False
                    continue
                try:
                    if line.strip():  # Skip empty lines
                        entry_data = line.split(',')
                        raw_ts = entry_data[0].strip().strip('"')
                        # Normalize to "YYYY-MM-DD HH:MM:SS.ffffff" in UTC
                        ts = raw_ts.replace('T', ' ').rstrip('Z').strip()
                        if '.' in ts:
                            head, frac = ts.split('.', 1)
                            # Keep only digits in fractional seconds and coerce to 6 digits
                            frac_digits = ''.join(ch for ch in frac if ch.isdigit())
                            frac6 = (frac_digits[:6]).ljust(6, '0')
                            ts_clean = f"{head}.{frac6}"
                        else:
                            ts_clean = f"{ts}.000000"
                        if first_entry:
                            initial = datetime.strptime(ts_clean, self.timestamp_format).replace(tzinfo=timezone.utc).timestamp()
                            first_entry = False
                        print("yielded timestamp: "+str(datetime.strptime(ts_clean, self.timestamp_format).replace(tzinfo=timezone.utc).timestamp() - initial))
                        yield (datetime.strptime(ts_clean, self.timestamp_format).replace(tzinfo=timezone.utc).timestamp() - initial)
                        initial = datetime.strptime(ts_clean, self.timestamp_format).replace(tzinfo=timezone.utc).timestamp()
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
    
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

class JSONLinesReader(TraceStreamReader):
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