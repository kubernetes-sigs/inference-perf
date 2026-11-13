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
from enum import Enum
from typing import Optional

from inference_perf.config.common import StrictBaseModel
from pydantic import Field, SecretStr, model_validator


class ModelServerType(Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    TGI = "tgi"
    MOCK = "mock"


class WaitFor(Enum):
    """What the client waits for before the first request of a run."""

    READY = "ready"
    REACHABLE = "reachable"
    NONE = "none"


class ModelServerClientConfig(StrictBaseModel):
    type: ModelServerType = Field(default=ModelServerType.VLLM, description="Type of model server being benchmarked.")
    model_name: Optional[str] = Field(
        default=None, description="Model name sent in each request. Auto-detected from the server if unset."
    )
    base_url: str = Field(description="Base URL of the model server, e.g. 'http://localhost:8000'.")
    ignore_eos: bool = Field(
        default=True,
        description="Ask the server to keep generating past the end-of-sequence token so outputs hit the requested length.",
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key sent as a bearer token with each request. Redacted in the logged and saved config.",
    )
    cert_path: Optional[str] = Field(default=None, description="Path to a client TLS certificate file.")
    key_path: Optional[str] = Field(default=None, description="Path to the private key for the client TLS certificate.")
    wait_for: WaitFor = Field(
        default=WaitFor.READY,
        description="What to wait for before the first request: 'ready' (the readiness route answers 200 or 404), "
        "'reachable' (any HTTP response) or 'none'. 'reachable' and 'none' require model_name and skip the /v1/models lookups.",
    )

    @model_validator(mode="after")
    def check_wait_for(self) -> "ModelServerClientConfig":
        return self._require_model_name_unless_ready()

    def _require_model_name_unless_ready(self) -> "ModelServerClientConfig":
        # An unset name is read from /v1/models, which a server that is not up yet cannot answer.
        if self.wait_for != WaitFor.READY and self.model_name is None:
            raise ValueError(
                f"'wait_for: {self.wait_for.value}' requires 'model_name': an unset name is read from /v1/models."
            )
        return self
