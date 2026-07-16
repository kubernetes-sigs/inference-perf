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

from pydantic import BaseModel, Field


class ModelServerType(Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    TGI = "tgi"
    MOCK = "mock"


class ModelServerClientConfig(BaseModel):
    type: ModelServerType = Field(default=ModelServerType.VLLM, description="Type of model server being benchmarked.")
    model_name: Optional[str] = Field(
        default=None, description="Model name sent in each request. Auto-detected from the server if unset."
    )
    base_url: str = Field(description="Base URL of the model server, e.g. 'http://localhost:8000'.")
    ignore_eos: bool = Field(
        default=True,
        description="Ask the server to keep generating past the end-of-sequence token so outputs hit the requested length.",
    )
    api_key: Optional[str] = Field(default=None, description="API key sent as a bearer token with each request.")
    cert_path: Optional[str] = Field(default=None, description="Path to a client TLS certificate file.")
    key_path: Optional[str] = Field(default=None, description="Path to the private key for the client TLS certificate.")
