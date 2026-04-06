import json
import logging
from pathlib import Path
from typing import List, Optional
from .trace_reader import NewTraceReader, NewTraceEntry

logger = logging.getLogger(__name__)


class SharedPrefixTraceEntry(NewTraceEntry):
    shared_prefix_length: int
    tail_input_length: int
    output_length: int
    shared_prefix_id: Optional[int] = None


class SharedPrefixTraceReader(NewTraceReader[SharedPrefixTraceEntry]):
    """Trace reader for Shared Prefix trace format (JSONL)."""

    def __init__(self) -> None:
        self.traces: List[SharedPrefixTraceEntry] = []

    def load_entries(self, file_path: Path) -> List[SharedPrefixTraceEntry]:
        """
        Load traces from file into memory.
        Returns a list of SharedPrefixTraceEntry.
        """
        if not self.traces:
            self._load_data(file_path)

        return self.traces

    def _load_data(self, file_path: Path) -> None:
        logger.info(f"Loading shared prefix traces from {file_path}")
        self.traces = []
        initial_timestamp: float = -1.0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry_dict = json.loads(line)

                        # Validate required fields
                        if "timestamp" not in entry_dict:
                            logger.warning(f"Line {line_num} missing 'timestamp'. Skipping.")
                            continue

                        ts = float(entry_dict["timestamp"])
                        if initial_timestamp < 0:
                            initial_timestamp = ts

                        # Normalize timestamp to start from 0
                        entry_dict["timestamp"] = ts - initial_timestamp

                        entry = SharedPrefixTraceEntry(**entry_dict)

                        self.traces.append(entry)

                    except json.JSONDecodeError as e:
                        logger.warning(f"Error decoding JSON on line {line_num}: {e}")
                    except ValueError as e:
                        logger.warning(f"Error parsing value on line {line_num}: {e}")

        except FileNotFoundError:
            logger.error(f"Trace file not found: {file_path}")
            raise

        logger.info(f"Loaded {len(self.traces)} trace entries.")
