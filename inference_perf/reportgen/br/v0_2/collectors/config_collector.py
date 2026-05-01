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
from typing import TYPE_CHECKING

from .base import CollectedStackObservability, StackObservabilityCollector

if TYPE_CHECKING:
    from inference_perf.config import BRV02Config


class ConfigStackCollector(StackObservabilityCollector):
    """Reads scenario.stack[] from report.br_v0_2.stack.

    Used when running outside K8s, or to override what k8s discovery would find.
    This is the migration path that lets llm-d-benchmark drop its translator: it
    sets the stack block in the inference-perf config and inference-perf emits
    BR0.2 directly.
    """

    def __init__(self, config: "BRV02Config") -> None:
        self.config = config

    async def start(self) -> None:
        return

    async def stop(self) -> None:
        return

    def collect(self) -> CollectedStackObservability:
        if not self.config.stack:
            return CollectedStackObservability()
        stack = [entry.model_dump(mode="json", by_alias=True, exclude_none=True) for entry in self.config.stack]
        return CollectedStackObservability(stack=stack)
