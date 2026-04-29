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
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class CollectedStackObservability(BaseModel):
    """Bundle of everything a collector produces, consumed by the BR0.2 adapter."""

    stack: List[Dict[str, Any]] = []
    observability: Optional[Dict[str, Any]] = None
    component_health: List[Dict[str, Any]] = []


class StackObservabilityCollector(ABC):
    """Produces the BR0.2 sections that depend on the deployment, not the workload.

    Implementations: ConfigStackCollector (manual override), KubernetesStackCollector
    (in-cluster discovery), NoopStackCollector (fallback).
    """

    @abstractmethod
    async def start(self) -> None:
        """Begin any background sampling (replica polling, pod watch, prom scraping)."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop background sampling and finalize collected state."""

    @abstractmethod
    def collect(self) -> CollectedStackObservability:
        """Return collected state. Safe to call after stop()."""
