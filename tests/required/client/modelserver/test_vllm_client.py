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
"""vLLM client subclass tests: BackendClientSuite run against a transcription
of vLLM's wire format (stop_reason/prompt_logprobs in every choice)."""

from backend_client_suite import CHAT_TEXT, COMPLETION_TEXT, MODEL, Backend, BackendClientSuite

from inference_perf.client.modelserver.vllm_client import vLLMModelServerClient

BACKEND = Backend(
    client_cls=vLLMModelServerClient,
    completion_response={
        "id": "cmpl-b0f5a3c0e4b94b2f9a2d7c8e1f6a5d43",
        "object": "text_completion",
        "created": 1750000000,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "text": COMPLETION_TEXT,
                "logprobs": None,
                "finish_reason": "length",
                "stop_reason": None,
                "prompt_logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 7, "total_tokens": 15, "completion_tokens": 8, "prompt_tokens_details": None},
    },
    chat_response={
        "id": "chatcmpl-3c1a9f7e2b3d4e5f8a7b6c5d4e3f2a1b",
        "object": "chat.completion",
        "created": 1750000000,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "reasoning_content": None, "content": CHAT_TEXT, "tool_calls": []},
                "logprobs": None,
                "finish_reason": "stop",
                "stop_reason": None,
            }
        ],
        "usage": {"prompt_tokens": 12, "total_tokens": 19, "completion_tokens": 7, "prompt_tokens_details": None},
        "prompt_logprobs": None,
    },
    completion_stream=[
        {
            "id": "cmpl-b0f5a3c0e4b94b2f9a2d7c8e1f6a5d43",
            "object": "text_completion",
            "created": 1750000000,
            "model": MODEL,
            "choices": [{"index": 0, "text": " Paris,", "logprobs": None, "finish_reason": None, "stop_reason": None}],
            "usage": None,
        },
        {
            "id": "cmpl-b0f5a3c0e4b94b2f9a2d7c8e1f6a5d43",
            "object": "text_completion",
            "created": 1750000000,
            "model": MODEL,
            "choices": [{"index": 0, "text": " the capital", "logprobs": None, "finish_reason": None, "stop_reason": None}],
            "usage": None,
        },
        {
            "id": "cmpl-b0f5a3c0e4b94b2f9a2d7c8e1f6a5d43",
            "object": "text_completion",
            "created": 1750000000,
            "model": MODEL,
            "choices": [{"index": 0, "text": " of France.", "logprobs": None, "finish_reason": "length", "stop_reason": None}],
            "usage": None,
        },
        {
            "id": "cmpl-b0f5a3c0e4b94b2f9a2d7c8e1f6a5d43",
            "object": "text_completion",
            "created": 1750000000,
            "model": MODEL,
            "choices": [],
            "usage": {"prompt_tokens": 7, "total_tokens": 15, "completion_tokens": 8},
        },
    ],
    chat_stream=[
        {
            "id": "chatcmpl-3c1a9f7e2b3d4e5f8a7b6c5d4e3f2a1b",
            "object": "chat.completion.chunk",
            "created": 1750000000,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-3c1a9f7e2b3d4e5f8a7b6c5d4e3f2a1b",
            "object": "chat.completion.chunk",
            "created": 1750000000,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {"content": "The capital"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-3c1a9f7e2b3d4e5f8a7b6c5d4e3f2a1b",
            "object": "chat.completion.chunk",
            "created": 1750000000,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {"content": " of France"}, "logprobs": None, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-3c1a9f7e2b3d4e5f8a7b6c5d4e3f2a1b",
            "object": "chat.completion.chunk",
            "created": 1750000000,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " is Paris."},
                    "logprobs": None,
                    "finish_reason": "stop",
                    "stop_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-3c1a9f7e2b3d4e5f8a7b6c5d4e3f2a1b",
            "object": "chat.completion.chunk",
            "created": 1750000000,
            "model": MODEL,
            "choices": [],
            "usage": {"prompt_tokens": 12, "total_tokens": 19, "completion_tokens": 7},
        },
    ],
    error_status=400,
    error_response={
        "object": "error",
        "message": "This model's maximum context length is 4096 tokens. However, you requested 4231 tokens "
        "(4031 in the messages, 200 in the completion). Please reduce the length of the messages or completion.",
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
                "owned_by": "vllm",
                "root": MODEL,
                "parent": None,
                "max_model_len": 131072,
                "permission": [
                    {
                        "id": "modelperm-33f8d4a2b1c04e5f9a6b7c8d9e0f1a2b",
                        "object": "model_permission",
                        "created": 1750000000,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False,
                    }
                ],
            }
        ],
    },
    expected_metric_filters=[f"model_name='{MODEL}'"],
    queue_metric_name="vllm:num_requests_waiting",
)


class TestVLLMClient(BackendClientSuite):
    backend = BACKEND
