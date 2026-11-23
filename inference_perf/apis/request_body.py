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
"""The body of one request, built the same way whether it is sent on its own or as a line of a batch."""

from inference_perf.apis.base import InferenceAPIData
from inference_perf.apis.tool_schema import clean_request_tool_schemas
from inference_perf.config import APIConfig, APIType
from inference_perf.payloads import RequestBody


async def build_request_body(
    data: InferenceAPIData,
    api_config: APIConfig,
    *,
    model_name: str,
    max_tokens: int,
    ignore_eos: bool,
    streaming: bool,
) -> RequestBody:
    """The record's own body, plus the options `api_config` sets on top of it."""
    body = await data.to_request_body(
        effective_model_name=model_name, max_tokens=max_tokens, ignore_eos=ignore_eos, streaming=streaming
    )

    # Opt-in: rewrite tool schemas for servers that reject the recorded ones.
    if api_config.clean_tool_schemas:
        clean_request_tool_schemas(body, api_config.type)

    # Add response_format for structured output if configured.
    if api_config.type != APIType.AnthropicMessages and api_config.response_format:
        body["response_format"] = api_config.response_format.to_api_format()

    return body
