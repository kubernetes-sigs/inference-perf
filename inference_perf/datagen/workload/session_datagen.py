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

from __future__ import annotations

from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from inference_perf.config import APIConfig, DataConfig, SessionReplayConfig
from inference_perf.datagen.replay.replay_graph_builder import DEPENDENCY_TYPE, RawCall, build_graph
from inference_perf.datagen.replay.replay_graph_session_datagen import ReplayGraphSessionGeneratorBase, ReplaySession
from inference_perf.datagen.replay.replay_graph_types import ReplayMessage
from inference_perf.datagen.workload.prompts import RecordPrompter
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.workload.arrangement import Node
from inference_perf.workload.materialize import BlockTextMaterializer
from inference_perf.workload.record import TextPart
from inference_perf.workload.sources import Workload, load_workload


def _safe_id(value: str) -> str:
    """Session and event ids are joined with ':' by the session runtime."""
    return value.replace(":", "_")


class WorkloadSessionGenerator(ReplayGraphSessionGeneratorBase):
    """Runs an arrangement with dependencies on the session runner.

    Each record that has nodes becomes one session. Its nodes become the
    calls of a replay graph: the prompt is materialized from the record, the
    predecessors are the node's dependencies, and the wait before a call is
    the node's think time. Everything after that (dispatch, predecessor
    waits, output substitution, session metrics) is the shared session
    runtime, unchanged.
    """

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
        mp_manager: Optional[SyncManager] = None,
        base_seed: Optional[int] = None,
        num_workers: int = 1,
        workload: Optional[Workload] = None,
    ) -> None:
        replay_config = config.workload.session if config.workload and config.workload.session else SessionReplayConfig()
        super().__init__(
            api_config,
            config,
            tokenizer,
            mp_manager=mp_manager,
            base_seed=base_seed,
            num_workers=num_workers,
            replay_config=replay_config,
        )
        if tokenizer is None:
            raise ValueError("A tokenizer is required to replay a workload")
        if workload is None:
            if config.workload is None:
                raise ValueError("data.workload is required for the workload_replay data generator")
            workload = load_workload(config.workload.format, config.workload.file, config.workload.block_size)
        self.workload = workload

        # One session per record with nodes, in the order the arrangement
        # first names them. A dependency must stay inside its record.
        self._nodes_by_record: Dict[str, List[Node]] = {}
        for node in workload.arrangement.nodes:
            self._nodes_by_record.setdefault(node.record_id, []).append(node)
        for record_id, nodes in self._nodes_by_record.items():
            ids = {node.id for node in nodes}
            for node in nodes:
                foreign = [dep for dep in node.depends_on if dep not in ids]
                if foreign:
                    raise ValueError(f"node {node.id!r} depends on {foreign} outside its record {record_id!r}")
        self._record_ids = list(self._nodes_by_record)

        corpus_path = Path(config.corpus_file_path) if config.corpus_file_path else None
        self._prompter = RecordPrompter(
            workload.source_id, BlockTextMaterializer(tokenizer, base_seed=self.base_seed, corpus_path=corpus_path)
        )
        self.initialize_sessions_lazy([f"s{slot}_{_safe_id(rid)}" for slot, rid in enumerate(self._record_ids)])

    def _build_session(self, session_index: int) -> Optional[ReplaySession]:
        record_id = self._record_ids[session_index]
        record = self.workload.record(record_id)
        nodes = self._nodes_by_record[record_id]
        session_id = self._session_ids[session_index]
        index_of = {node.id: i for i, node in enumerate(nodes)}

        calls: List[RawCall] = []
        for node in nodes:
            messages = self._prompter.messages(record, node.turn)
            declared = [m.declared_tokens for m in messages]
            output_turn = record.turns[node.turn]
            recorded_output = "".join(p.text for p in output_turn.parts if isinstance(p, TextPart))
            calls.append(
                RawCall(
                    call_id=_safe_id(node.id),
                    trace_id=session_id,
                    t_start_ms=node.send_at_ms or 0,
                    t_end_ms=node.send_at_ms or 0,
                    model=str(record.metadata.get("model", "")),
                    messages=[ReplayMessage(role=m.role, text=m.text) for m in messages],
                    out_message=ReplayMessage(role="assistant", text=recorded_output),
                    prompt_tokens=sum(d for d in declared if d is not None) if all(d is not None for d in declared) else None,
                    completion_tokens=output_turn.output_tokens,
                    temperature=None,
                    max_tokens_recorded=output_turn.output_tokens,
                )
            )

        def from_arrangement(
            raw_calls: List[RawCall],
        ) -> Tuple[List[Dict[int, DEPENDENCY_TYPE]], Dict[Tuple[str, str], List[int]]]:
            return [{index_of[dep]: DEPENDENCY_TYPE.TEMPORAL for dep in node.depends_on} for node in nodes], {}

        graph = build_graph(calls, source_file=self.workload.source_id, predecessor_finder=from_arrangement)
        # The arrangement states the think time; the timestamps are informational.
        for i, node in enumerate(nodes):
            graph.events[f"event_{i:03d}_{calls[i].call_id}"].wait_ms = node.think_ms
        return ReplaySession(
            session_id=session_id, source_id=self.workload.source_id, session_index=session_index, graph=graph
        )
