from abc import ABC, abstractmethod
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