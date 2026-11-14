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
from urllib.parse import urlsplit

from inference_perf.config.common import StrictBaseModel
from pydantic import Field, SecretStr, model_validator


class ModelServerType(Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    TGI = "tgi"
    MOCK = "mock"
    LLMD_ASYNC_REDIS = "llmd_async_redis"


class LlmdAsyncRedisConfig(StrictBaseModel):
    request_queue_name: str = Field(
        default="request-sortedset",
        description="llm-d-async request sorted set the requests are enqueued on (its queue entry's 'queue_name').",
    )
    result_queue_name: str = Field(
        default="result-list",
        description=(
            "Prefix of the result lists. Each worker process reads its own list, '<prefix>:<random id>', named on every "
            "request it enqueues, so the llm-d-async queue entry must not set its own 'result_queue_name'."
        ),
    )
    default_deadline_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Deadline of each request, in seconds after it is enqueued. llm-d-async dispatches earliest deadline first.",
    )
    result_grace_seconds: float = Field(
        default=30.0,
        ge=0,
        description=(
            "How long past the deadline to wait for a result before recording the request as timed out. llm-d-async "
            "reports an expired request itself, as DEADLINE_EXCEEDED, so this only covers a dispatcher that never answers."
        ),
    )
    password: Optional[SecretStr] = Field(
        default=None,
        description="Redis AUTH password. Kept out of base_url so it is redacted in the logged and saved config.",
    )


class ModelServerClientConfig(StrictBaseModel):
    type: ModelServerType = Field(default=ModelServerType.VLLM, description="Type of model server being benchmarked.")
    model_name: Optional[str] = Field(
        default=None, description="Model name sent in each request. Auto-detected from the server if unset."
    )
    base_url: str = Field(
        description="Base URL of the model server, e.g. 'http://localhost:8000'. For llmd_async_redis, the Redis URL."
    )
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
    llmd_async_redis: Optional[LlmdAsyncRedisConfig] = Field(
        default=None,
        description="Queue settings for type 'llmd_async_redis'. Defaults apply when the type is set and this is omitted.",
    )

    @model_validator(mode="after")
    def validate_llmd_async_redis(self) -> "ModelServerClientConfig":
        if self.type != ModelServerType.LLMD_ASYNC_REDIS:
            if self.llmd_async_redis is not None:
                raise ValueError("'llmd_async_redis' is only used with type 'llmd_async_redis'.")
            return self
        url = urlsplit(self.base_url)
        if url.scheme not in ("redis", "rediss"):
            raise ValueError("With type 'llmd_async_redis', base_url is the Redis URL: redis://host:port/db or rediss://...")
        if url.password is not None:
            raise ValueError("Put the Redis password in 'llmd_async_redis.password', not in base_url, so it is redacted.")
        if self.model_name is None:
            raise ValueError("Type 'llmd_async_redis' needs 'model_name': there is no /v1/models endpoint to detect it from.")
        if self.llmd_async_redis is None:
            self.llmd_async_redis = LlmdAsyncRedisConfig()
        return self
