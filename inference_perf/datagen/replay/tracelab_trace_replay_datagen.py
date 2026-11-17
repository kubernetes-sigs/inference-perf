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
record layer. What a trace row means lives with the format in
inference_perf/workload/formats/tracelab.py, and reading, selecting and
running a trace are shared by every format. This class is the
``tracelab_trace_replay`` config block naming that format. See
https://tracelab.cs.washington.edu and https://arxiv.org/abs/2606.30560.
"""

from multiprocessing.managers import SyncManager
from typing import Dict, Optional

from inference_perf.config import APIConfig, DataConfig
from inference_perf.datagen.workload import WorkloadSessionGenerator, load_trace
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.workload import FilterExpressionError as FilterExpressionError

# one decoded trace row, before the format validates it.
TraceLabRow = Dict[str, object]


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
        super().__init__(
            api_config,
            config,
            tokenizer,
            mp_manager=mp_manager,
            base_seed=base_seed,
            num_workers=num_workers,
            trace=load_trace("TraceLab", self.tracelab_config, base_seed if base_seed is not None else 42),
            replay_config=self.tracelab_config,
        )
