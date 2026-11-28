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

from inference_perf.apis import InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.config import APIConfig, DataConfig, SessionReplayConfig
from inference_perf.datagen.replay.replay_graph_builder import DEPENDENCY_TYPE, RawCall, build_graph
from inference_perf.datagen.replay.replay_graph_session_datagen import ReplayGraphSessionGeneratorBase, ReplaySession
from inference_perf.datagen.replay.replay_graph_types import ReplayMessage
from inference_perf.datagen.workload.trace import Trace, load_trace
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.workload import BlockTextMaterializer, Prompter, Record


def _safe_id(value: str) -> str:
    """Session and event ids are joined with ':' by the session runtime."""
    return value.replace(":", "_")


class WorkloadSessionGenerator(ReplayGraphSessionGeneratorBase):
    """Runs a trace of sessions on the session runner.

    A session's records become the calls of a replay graph: the prompt is
    built from the record, the predecessors are the records it waited for,
    and the wait before a call is the record's recorded wait. Everything
    after that (dispatch, predecessor waits, output substitution, session
    metrics) is the shared session runtime, unchanged. A recorded assistant
    message that reappears in a later prompt is replaced there by the live reply
    of the record that elicited it.

    The trace and its replay settings come from `data.workload` unless they
    are passed in, which is how a format with a config block of its own runs.
    """

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
        mp_manager: Optional[SyncManager] = None,
        base_seed: Optional[int] = None,
        num_workers: int = 1,
        trace: Optional[Trace] = None,
        replay_config: Optional[SessionReplayConfig] = None,
    ) -> None:
        if replay_config is None:
            replay_config = config.workload if config.workload is not None else SessionReplayConfig()
        self._session_config = replay_config
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
            raise ValueError("A tokenizer is required to replay a trace")
        if trace is None:
            if config.workload is None:
                raise ValueError("data.workload is required for the workload_replay data generator")
            trace = load_trace(config.workload.format, config.workload, self.base_seed, config.workload.block_size)
        self.replay_trace = trace

        # slot -> index of the session it runs. Longer than the trace once a
        # slot copies an earlier one: a duplicate added up front or a replay
        # added on demand.
        self._slot_sessions = list(range(len(trace.sessions)))
        session_ids = [f"s{slot}_{_safe_id(session.id)}" for slot, session in enumerate(trace.sessions)]
        target = replay_config.duplicate_sessions_target
        if target is not None:
            copies = 0
            while len(session_ids) < target:
                source_slot = copies % len(trace.sessions)
                copies += 1
                session_ids.append(f"{session_ids[source_slot]}_dup{copies}")
                self._slot_sessions.append(source_slot)

        corpus_path = Path(config.corpus_file_path) if config.corpus_file_path else None
        self._prompter = Prompter(
            trace.source_id,
            BlockTextMaterializer(tokenizer.get_tokenizer(), base_seed=self.base_seed, corpus_path=corpus_path),
            count_special_tokens=trace.counts_special_tokens,
        )
        self.initialize_sessions_lazy(session_ids)

    def _model_for(self, record: Record) -> str:
        """The model the graph names for a call. The request itself goes to the server's model."""
        recorded = record.model or ""
        if self._session_config.use_static_model:
            return self._session_config.static_model_name
        return (self._session_config.model_mapping or {}).get(recorded, recorded)

    def _on_cycled_slot(self, session_index: int, source_slot: int) -> None:
        # the base class registers replay slots in index order, one call each
        assert session_index == len(self._slot_sessions), f"slot {session_index} registered out of order"
        self._slot_sessions.append(self._slot_sessions[source_slot])

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        api_data = super().load_lazy_data(data)
        event = self._resolve_event(data)
        session = self.replay_trace.sessions[self._slot_sessions[event.session_index]]
        # a call is named after its record, with the ':' the runtime reserves replaced
        api_data.record_id = next((record.id for record in session.records if _safe_id(record.id) == event.call_id), None)
        return api_data

    def _build_session(self, session_index: int) -> Optional[ReplaySession]:
        session = self.replay_trace.sessions[self._slot_sessions[session_index]]
        session_id = self._session_ids[session_index]
        index_of = {record.id: i for i, record in enumerate(session.records)}

        calls: List[RawCall] = []
        for record in session.records:
            messages = self._prompter.prompt(record, session.id)
            declared = [m.declared_tokens for m in messages]
            # The recorded output, built the way the same message is built when
            # a later prompt carries it, so the session runtime can find it
            # there and put the live reply in its place.
            recorded_output = self._prompter.reply(record, session.id).text if record.reply else ""
            start_ms = session.delivery.of(record.id).start_ms or 0
            calls.append(
                RawCall(
                    call_id=_safe_id(record.id),
                    trace_id=session_id,
                    t_start_ms=start_ms,
                    t_end_ms=start_ms,
                    model=self._model_for(record),
                    messages=[ReplayMessage(role=m.role, text=m.text) for m in messages],
                    out_message=ReplayMessage(role="assistant", text=recorded_output),
                    prompt_tokens=sum(d for d in declared if d is not None) if all(d is not None for d in declared) else None,
                    completion_tokens=record.output_tokens,
                    temperature=None,
                    # graph metadata only: the request's max_tokens is the
                    # recorded completion length above
                    max_tokens_recorded=record.output_tokens or self._session_config.default_max_tokens,
                )
            )

        def from_delivery(
            raw_calls: List[RawCall],
        ) -> Tuple[List[Dict[int, DEPENDENCY_TYPE]], Dict[Tuple[str, str], List[int]]]:
            return [
                {index_of[other]: DEPENDENCY_TYPE.TEMPORAL for other in session.delivery.of(record.id).after}
                for record in session.records
            ], {}

        graph = build_graph(calls, source_file=self.replay_trace.source_id, predecessor_finder=from_delivery)
        # The delivery states the wait; the timestamps are informational.
        for i, record in enumerate(session.records):
            graph.events[f"event_{i:03d}_{calls[i].call_id}"].wait_ms = session.delivery.of(record.id).wait_ms
        return ReplaySession(session_id=session_id, source_id=session.id, session_index=session_index, graph=graph)
