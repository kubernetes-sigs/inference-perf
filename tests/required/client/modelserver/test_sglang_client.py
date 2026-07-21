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
"""SGLang client subclass tests: BackendClientSuite run against a transcription
of SGLang's wire format (matched_stop in every choice, unprefixed request ids)."""

from backend_client_suite import CHAT_TEXT, COMPLETION_TEXT, MODEL, Backend, BackendClientSuite

from inference_perf.client.modelserver.sglang_client import SGlangModelServerClient

BACKEND = Backend(
    client_cls=SGlangModelServerClient,
    completion_response={
        "id": "d3f5a5b8c7e64f2b8a1c9d0e7f6b5a4c",
        "object": "text_completion",
        "created": 1750000000,
        "model": MODEL,
        "choices": [{"index": 0, "text": COMPLETION_TEXT, "logprobs": None, "finish_reason": "length", "matched_stop": None}],
        "usage": {"prompt_tokens": 7, "total_tokens": 15, "completion_tokens": 8, "prompt_tokens_details": None},
    },
    chat_response={
        "id": "5f2f8f6a1b3c4d5e8f7a6b5c4d3e2f1a",
        "object": "chat.completion",
        "created": 1750000000,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": CHAT_TEXT, "reasoning_content": None, "tool_calls": None},
                "logprobs": None,
                "finish_reason": "stop",
                "matched_stop": 128009,
            }
        ],
        "usage": {"prompt_tokens": 12, "total_tokens": 19, "completion_tokens": 7, "prompt_tokens_details": None},
    },
    completion_stream=[
        {
            "id": "d3f5a5b8c7e64f2b8a1c9d0e7f6b5a4c",
            "object": "text_completion",
            "created": 1750000000,
            "model": MODEL,
            "choices": [{"index": 0, "text": " Paris,", "logprobs": None, "finish_reason": None, "matched_stop": None}],
            "usage": None,
        },
        {
            "id": "d3f5a5b8c7e64f2b8a1c9d0e7f6b5a4c",
            "object": "text_completion",
            "created": 1750000000,
            "model": MODEL,
            "choices": [{"index": 0, "text": " the capital", "logprobs": None, "finish_reason": None, "matched_stop": None}],
            "usage": None,
        },
        {
            "id": "d3f5a5b8c7e64f2b8a1c9d0e7f6b5a4c",
            "object": "text_completion",
            "created": 1750000000,
            "model": MODEL,
            "choices": [
                {"index": 0, "text": " of France.", "logprobs": None, "finish_reason": "length", "matched_stop": None}
            ],
            "usage": None,
        },
        {
            "id": "d3f5a5b8c7e64f2b8a1c9d0e7f6b5a4c",
            "object": "text_completion",
            "created": 1750000000,
            "model": MODEL,
            "choices": [],
            "usage": {"prompt_tokens": 7, "total_tokens": 15, "completion_tokens": 8, "prompt_tokens_details": None},
        },
    ],
    chat_stream=[
        {
            "id": "5f2f8f6a1b3c4d5e8f7a6b5c4d3e2f1a",
            "object": "chat.completion.chunk",
            "created": 1750000000,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "logprobs": None,
                    "finish_reason": None,
                    "matched_stop": None,
                }
            ],
        },
        {
            "id": "5f2f8f6a1b3c4d5e8f7a6b5c4d3e2f1a",
            "object": "chat.completion.chunk",
            "created": 1750000000,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "The capital"},
                    "logprobs": None,
                    "finish_reason": None,
                    "matched_stop": None,
                }
            ],
        },
        {
            "id": "5f2f8f6a1b3c4d5e8f7a6b5c4d3e2f1a",
            "object": "chat.completion.chunk",
            "created": 1750000000,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " of France"},
                    "logprobs": None,
                    "finish_reason": None,
                    "matched_stop": None,
                }
            ],
        },
        {
            "id": "5f2f8f6a1b3c4d5e8f7a6b5c4d3e2f1a",
            "object": "chat.completion.chunk",
            "created": 1750000000,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " is Paris."},
                    "logprobs": None,
                    "finish_reason": "stop",
                    "matched_stop": 128009,
                }
            ],
        },
        {
            "id": "5f2f8f6a1b3c4d5e8f7a6b5c4d3e2f1a",
            "object": "chat.completion.chunk",
            "created": 1750000000,
            "model": MODEL,
            "choices": [],
            "usage": {"prompt_tokens": 12, "total_tokens": 19, "completion_tokens": 7, "prompt_tokens_details": None},
        },
    ],
    error_status=400,
    error_response={
        "object": "error",
        "message": "Input length 4231 exceeds the maximum allowed length 4096.",
        "type": "BadRequestError",
        "param": None,
        "code": 400,
    },
    models_response={
        "object": "list",
        "data": [
            {
                "id": MODEL,
                "object": "model",
                "created": 1750000000,
                "owned_by": "sglang",
                "root": MODEL,
                "max_model_len": 131072,
            }
        ],
    },
    expected_metric_filters=[f"model_name='{MODEL}'"],
    queue_metric_name="sglang:num_queue_reqs",
)


class TestSGLangClient(BackendClientSuite):
    backend = BACKEND
