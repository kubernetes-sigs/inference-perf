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
from inference_perf.datagen.workload.prompts import RecordPrompter
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.workload.materialize import BlockTextMaterializer
from inference_perf.workload.sources import Workload, load_workload


class WorkloadRequestDataGenerator(DataGenerator, LazyLoadDataMixin):
    """Runs an arrangement without dependencies on the stage runner.

    Nodes go out in send-time order. With `load.type: trace_replay` the load
    generator takes the send times from the arrangement; with a rate or
    concurrency stage it cycles through the nodes at the configured pace.
    """

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
        seed: Optional[int] = None,
        workload: Optional[Workload] = None,
    ) -> None:
        super().__init__(api_config, config, tokenizer)
        if tokenizer is None:
            raise ValueError("A tokenizer is required to replay a workload")
        if workload is None:
            if config.workload is None:
                raise ValueError("data.workload is required for the workload_replay data generator")
            workload = load_workload(config.workload.format, config.workload.file, config.workload.block_size)
        if workload.arrangement.has_dependencies():
            raise ValueError(
                "This workload's arrangement has dependencies between requests; run it with load.type trace_session_replay"
            )
        self.workload = workload
        self._nodes = sorted(workload.arrangement.nodes, key=lambda n: n.send_at_ms if n.send_at_ms is not None else 0)
        corpus_path = Path(config.corpus_file_path) if config.corpus_file_path else None
        self._prompter = RecordPrompter(
            workload.source_id, BlockTextMaterializer(tokenizer, base_seed=seed or 0, corpus_path=corpus_path)
        )

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def is_io_distribution_supported(self) -> bool:
        return False

    def is_shared_prefix_supported(self) -> bool:
        return False

    def get_request_count(self) -> int:
        return len(self._nodes)

    def send_offsets_ms(self) -> Optional[List[int]]:
        """Recorded send times in node order, or None when any node has none."""
        offsets = [node.send_at_ms for node in self._nodes]
        if any(offset is None for offset in offsets):
            return None
        return [offset for offset in offsets if offset is not None]

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        i = 0
        while True:
            yield LazyLoadInferenceAPIData(data_index=i % len(self._nodes))
            i += 1

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        node = self._nodes[data.data_index]
        record = self.workload.record(node.record_id)
        messages = self._prompter.messages(record, node.turn)
        max_tokens = record.turns[node.turn].output_tokens or 0
        if self.api_config.type == APIType.Completion:
            return CompletionAPIData(prompt="\n".join(m.text for m in messages), max_tokens=max_tokens)
        return ChatCompletionAPIData(
            messages=[ChatMessage(role=m.role, content=m.text) for m in messages], max_tokens=max_tokens
        )
