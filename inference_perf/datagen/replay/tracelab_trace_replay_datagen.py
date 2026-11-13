# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TraceLab Trace Replay Data Generator.

Replays UW TraceLab coding-agent traces (Claude Code / Codex) on the workload
record layer. What a trace row is, and how rounds, prefixes and think times
are read from it, lives with the format in
inference_perf/workload/sources/tracelab.py. This module is the
``tracelab_trace_replay`` config block applied to that: where the rows come
from, which sessions are kept, the order they run in and how the recorded
gaps are capped. See https://tracelab.cs.washington.edu and
https://arxiv.org/abs/2606.30560.
"""

import random
from multiprocessing.managers import SyncManager
from typing import Optional

from inference_perf.config import APIConfig, DataConfig, TraceLabTraceReplayConfig
from inference_perf.datagen.workload.session_datagen import WorkloadSessionGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.workload.arrangement import Arrangement
from inference_perf.workload.sources.base import Workload
from inference_perf.workload.sources.tracelab import (
    TRACELAB_BLOCK_SIZE,
    TraceLabReader,
    build_workload,
)
from inference_perf.workload.sources.tracelab import FilterExpressionError as FilterExpressionError


def load_tracelab_workload(config: TraceLabTraceReplayConfig, seed: int) -> Workload:
    """The workload a ``tracelab_trace_replay`` block describes."""
    sessions = TraceLabReader(
        trace_directory=config.trace_directory,
        trace_files=config.trace_files,
        hf_dataset_path=config.hf_dataset_path,
        filter=config.filter,
        max_sessions=config.num_dataset_entries,
        skip_invalid=config.skip_invalid_files,
    ).sessions()
    # Sessions run in a seeded shuffle of their sorted ids. The RNG is local so
    # building a generator never touches the process-global random state.
    order = sorted(sessions)
    random.Random(seed).shuffle(order)
    workload = build_workload({session_id: sessions[session_id] for session_id in order}, "tracelab", TRACELAB_BLOCK_SIZE)

    cap_seconds = config.trace_idle_gap_cap_seconds
    cap_ms = int(cap_seconds * 1000.0) if cap_seconds is not None and cap_seconds > 0 else None

    def replayed(think_ms: int) -> int:
        if config.ignore_trace_delays:
            return 0
        return think_ms if cap_ms is None else min(think_ms, cap_ms)

    nodes = [node.model_copy(update={"think_ms": replayed(node.think_ms)}) for node in workload.arrangement.nodes]
    return Workload(source_id=workload.source_id, records=workload.records, arrangement=Arrangement(nodes=nodes))


class TraceLabTraceReplayDataGenerator(WorkloadSessionGenerator):
    """Data generator that replays TraceLab coding-agent traces."""

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
        mp_manager: Optional[SyncManager] = None,
        base_seed: Optional[int] = None,
        num_workers: int = 1,
    ) -> None:
        if not hasattr(config, "tracelab_trace_replay") or config.tracelab_trace_replay is None:
            raise ValueError("tracelab_trace_replay configuration is required for TraceLabTraceReplayDataGenerator")

        self.tracelab_config = config.tracelab_trace_replay
        workload = load_tracelab_workload(self.tracelab_config, base_seed if base_seed is not None else 42)
        # a session's runtime id carries its place among the sorted trace ids
        session_ids = sorted({record.session_id or record.id for record in workload.records})
        self._trace_index = {session_id: index for index, session_id in enumerate(session_ids)}

        super().__init__(
            api_config,
            config,
            tokenizer,
            mp_manager=mp_manager,
            base_seed=base_seed,
            num_workers=num_workers,
            workload=workload,
            replay_config=self.tracelab_config,
        )

    def _runtime_session_id(self, slot: int, session_key: str) -> str:
        # The runtime splits qualified event ids on the first ':', and a
        # TraceLab id is 'claude:<uuid>'. The raw id stays in source_id.
        return f"tracelabtrace{self._trace_index[session_key]}_{session_key.replace(':', '_')}"

    def _session_source_id(self, session_key: str) -> str:
        return session_key
