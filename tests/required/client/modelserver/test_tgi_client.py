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
"""TGI client subclass tests: BackendClientSuite run against a transcription of
TGI's wire format (system_fingerprint, empty request ids, role repeated in every
streaming delta, flat {"error", "error_type"} body at 422)."""

from backend_client_suite import CHAT_TEXT, COMPLETION_TEXT, MODEL, Backend, BackendClientSuite

from inference_perf.client.modelserver.tgi_client import TGImodelServerClient

BACKEND = Backend(
    client_cls=TGImodelServerClient,
    completion_response={
        "object": "text_completion",
        "id": "",
        "created": 1750000000,
        "model": MODEL,
        "system_fingerprint": "3.3.4-native",
        "choices": [{"index": 0, "text": COMPLETION_TEXT, "logprobs": None, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 8, "total_tokens": 15},
    },
    chat_response={
        "object": "chat.completion",
        "id": "",
        "created": 1750000000,
        "model": MODEL,
        "system_fingerprint": "3.3.4-native",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": CHAT_TEXT}, "logprobs": None, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
    },
    completion_stream=[
        {
            "object": "text_completion",
            "id": "",
            "created": 1750000000,
            "model": MODEL,
            "system_fingerprint": "3.3.4-native",
            "choices": [{"index": 0, "text": " Paris,", "logprobs": None, "finish_reason": None}],
            "usage": None,
        },
        {
            "object": "text_completion",
            "id": "",
            "created": 1750000000,
            "model": MODEL,
            "system_fingerprint": "3.3.4-native",
            "choices": [{"index": 0, "text": " the capital", "logprobs": None, "finish_reason": None}],
            "usage": None,
        },
        {
            "object": "text_completion",
            "id": "",
            "created": 1750000000,
            "model": MODEL,
            "system_fingerprint": "3.3.4-native",
            "choices": [{"index": 0, "text": " of France.", "logprobs": None, "finish_reason": "length"}],
            "usage": None,
        },
        {
            "object": "text_completion",
            "id": "",
            "created": 1750000000,
            "model": MODEL,
            "system_fingerprint": "3.3.4-native",
            "choices": [],
            "usage": {"prompt_tokens": 7, "completion_tokens": 8, "total_tokens": 15},
        },
    ],
    chat_stream=[
        {
            "object": "chat.completion.chunk",
            "id": "",
            "created": 1750000000,
            "model": MODEL,
            "system_fingerprint": "3.3.4-native",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}],
            "usage": None,
        },
        {
            "object": "chat.completion.chunk",
            "id": "",
            "created": 1750000000,
            "model": MODEL,
            "system_fingerprint": "3.3.4-native",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "The capital"}, "logprobs": None, "finish_reason": None}
            ],
            "usage": None,
        },
        {
            "object": "chat.completion.chunk",
            "id": "",
            "created": 1750000000,
            "model": MODEL,
            "system_fingerprint": "3.3.4-native",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": " of France"}, "logprobs": None, "finish_reason": None}
            ],
            "usage": None,
        },
        {
            "object": "chat.completion.chunk",
            "id": "",
            "created": 1750000000,
            "model": MODEL,
            "system_fingerprint": "3.3.4-native",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": " is Paris."},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": None,
        },
        {
            "object": "chat.completion.chunk",
            "id": "",
            "created": 1750000000,
            "model": MODEL,
            "system_fingerprint": "3.3.4-native",
            "choices": [],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
        },
    ],
    error_status=422,
    error_response={
        "error": "Input validation error: `inputs` tokens + `max_new_tokens` must be <= 4096. "
        "Given: 4031 `inputs` tokens and 200 `max_new_tokens`",
        "error_type": "validation",
    },
    models_response={
        "object": "list",
        "data": [{"id": MODEL, "object": "model", "created": 0, "owned_by": "meta-llama"}],
    },
    expected_metric_filters=[],
    queue_metric_name="tgi_queue_size",
)


class TestTGIClient(BackendClientSuite):
    backend = BACKEND
