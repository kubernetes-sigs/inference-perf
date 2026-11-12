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
"""Validity rules for the llm-d-async Redis server type in ``inference_perf.config.client.modelserver``."""

import pytest
from pydantic import ValidationError

from inference_perf.config import ModelServerClientConfig
from inference_perf.config.redaction import REDACTED, redact

REDIS = {"type": "llmd_async_redis", "base_url": "redis://redis:6379/0", "model_name": "m"}


# Type llmd_async_redis with a Redis base_url and a model but no queue block, the same with a rediss:// URL and the
# password "hunter2" in the block, then type vllm. Expects llm-d-async's defaults filled in (request-sortedset, result-list,
# 300 s deadline, 30 s grace), the password shown as [REDACTED] like every other credential, and no block for vllm.
def test_llmd_async_redis_defaults_and_redacted_password() -> None:
    queue = ModelServerClientConfig.model_validate(REDIS).llmd_async_redis
    assert queue is not None
    assert (queue.request_queue_name, queue.result_queue_name) == ("request-sortedset", "result-list")
    assert (queue.default_deadline_seconds, queue.result_grace_seconds) == (300, 30)
    settings = {**REDIS, "base_url": "rediss://bench@redis:6380/1", "llmd_async_redis": {"password": "hunter2"}}
    rendered = redact(ModelServerClientConfig.model_validate(settings).model_dump(mode="json"), ModelServerClientConfig)
    assert rendered["llmd_async_redis"]["password"] == REDACTED and "hunter2" not in str(rendered)
    assert ModelServerClientConfig(base_url="http://vllm:8000").llmd_async_redis is None


# Settings that cannot work: an http:// base_url, a password inside base_url (logged unredacted), no model_name, the
# queue block with type vllm, a 0 s deadline and a negative grace. Expects a validation error naming each.
@pytest.mark.parametrize(
    "change, match",
    [
        pytest.param({"base_url": "http://redis:6379"}, "Redis URL", id="http-url"),
        pytest.param({"base_url": "redis://:secret@redis:6379/0"}, "password", id="password-in-url"),
        pytest.param({"model_name": None}, "model_name", id="no-model-name"),
        pytest.param({"type": "vllm", "llmd_async_redis": {}}, "only used with type", id="block-without-its-type"),
        pytest.param({"llmd_async_redis": {"default_deadline_seconds": 0}}, "greater than 0", id="zero-deadline"),
        pytest.param({"llmd_async_redis": {"result_grace_seconds": -1}}, "greater than or equal to 0", id="negative-grace"),
    ],
)
def test_llmd_async_redis_rejects_settings_that_cannot_work(change: dict[str, object], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        ModelServerClientConfig.model_validate({**REDIS, **change})
