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

from pathlib import Path
from typing import Generator, List, Optional

from inference_perf.apis import (
    ChatCompletionAPIData,
    ChatMessage,
    CompletionAPIData,
    InferenceAPIData,
    LazyLoadInferenceAPIData,
)
from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.datagen.base import DataGenerator, LazyLoadDataMixin
from inference_perf.datagen.workload.trace import Trace, load_trace
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.workload import BlockTextMaterializer, Prompter


class WorkloadRequestDataGenerator(DataGenerator, LazyLoadDataMixin):
    """Runs a trace of independent requests on the stage runner.

    Requests go out in the order they were recorded. With
    `load.type: trace_replay` the load generator takes the send times from
    the records; with a rate or concurrency stage it cycles through them at
    the configured pace.
    """

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
        seed: Optional[int] = None,
        trace: Optional[Trace] = None,
    ) -> None:
        super().__init__(api_config, config, tokenizer)
        if tokenizer is None:
            raise ValueError("A tokenizer is required to replay a trace")
        if trace is None:
            if config.workload is None:
                raise ValueError("data.workload is required for the workload_replay data generator")
            trace = load_trace(config.workload.format, config.workload, seed or 0, config.workload.block_size)
        if trace.records_sessions:
            raise ValueError("This trace records sessions; run it with load.type trace_session_replay")
        self.replay_trace = trace
        requests = [
            (session.id, record, session.delivery.of(record.id).start_ms)
            for session in trace.sessions
            for record in session.records
        ]
        requests.sort(key=lambda r: r[2] or 0)
        self._requests = [(session_id, record) for session_id, record, _ in requests]
        self._starts_ms = [start_ms for _, _, start_ms in requests]
        corpus_path = Path(config.corpus_file_path) if config.corpus_file_path else None
        self._prompter = Prompter(
            trace.source_id,
            BlockTextMaterializer(tokenizer.get_tokenizer(), base_seed=seed or 0, corpus_path=corpus_path),
            count_special_tokens=trace.counts_special_tokens,
        )

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def is_io_distribution_supported(self) -> bool:
        return False

    def is_shared_prefix_supported(self) -> bool:
        return False

    def get_request_count(self) -> int:
        return len(self._requests)

    def send_offsets_ms(self) -> Optional[List[int]]:
        """Recorded send times in send order, or None when any request has none."""
        if any(start_ms is None for start_ms in self._starts_ms):
            return None
        return [start_ms for start_ms in self._starts_ms if start_ms is not None]

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        i = 0
        while True:
            yield LazyLoadInferenceAPIData(data_index=i % len(self._requests))
            i += 1

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        session_id, record = self._requests[data.data_index]
        messages = self._prompter.prompt(record, session_id)
        max_tokens = record.output_tokens or 0
        if self.api_config.type == APIType.Completion:
            return CompletionAPIData(prompt="\n".join(m.text for m in messages), max_tokens=max_tokens, record_id=record.id)
        return ChatCompletionAPIData(
            messages=[ChatMessage(role=m.role, content=m.text) for m in messages],
            max_tokens=max_tokens,
            record_id=record.id,
        )
