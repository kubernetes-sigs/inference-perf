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

"""From a trace config block to the sessions a run replays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from inference_perf.config import TraceReplayConfig
from inference_perf.workload import (
    Selection,
    Session,
    TraceLocation,
    format_class,
    in_seeded_order,
    read_sessions,
    with_waits,
)


@dataclass(frozen=True)
class Trace:
    """What a run replays: the kept sessions, in the order they run.

    `records_sessions` says which scheduler runs it: a trace of sessions goes
    to the session runner, a trace of independent timestamped requests to the
    stage runner.
    """

    source_id: str
    sessions: Sequence[Session]
    records_sessions: bool = True
    counts_special_tokens: bool = True


def _source_id(config: TraceReplayConfig) -> str:
    if config.trace_directory is not None:
        return Path(config.trace_directory).name
    if config.trace_files is not None:
        return Path(config.trace_files[0]).name if len(config.trace_files) == 1 else "trace"
    path = config.hf_dataset_path["path"] if isinstance(config.hf_dataset_path, dict) else config.hf_dataset_path
    return f"hf:{path}"


def load_trace(format_name: str, config: TraceReplayConfig, seed: int, block_size: Optional[int] = None) -> Trace:
    """Read the trace a config block describes with the format called `format_name`."""
    fmt = format_class(format_name)(block_size=block_size)
    sessions = read_sessions(
        fmt,
        TraceLocation(directory=config.trace_directory, files=config.trace_files, hf_dataset=config.hf_dataset_path),
        Selection(filter=config.filter, max_sessions=config.num_dataset_entries, skip_invalid=config.skip_invalid_files),
    )
    records_sessions = fmt.session_key is not None
    if records_sessions:
        sessions = in_seeded_order(sessions, seed)
        cap_seconds = config.trace_idle_gap_cap_seconds
        cap_ms = int(cap_seconds * 1000.0) if cap_seconds is not None and cap_seconds > 0 else None

        def replayed(wait_ms: int) -> int:
            if config.ignore_trace_delays:
                return 0
            return wait_ms if cap_ms is None else min(wait_ms, cap_ms)

        sessions = with_waits(sessions, replayed)
    return Trace(
        source_id=_source_id(config),
        sessions=sessions,
        records_sessions=records_sessions,
        counts_special_tokens=fmt.counts_special_tokens,
    )
