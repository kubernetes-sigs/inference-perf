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
"""Optional cleanup of tool parameter schemas for servers with strict schema validators.

By default tool schemas are sent to the server exactly as recorded. Some servers reject
valid JSON Schema they do not support: vLLM's xgrammar backend and llm-d-inference-sim
reject keys such as ``default``, ``title``, ``minimum`` and ``maxItems``, ``anyOf``/``oneOf``
without a plain ``type``, object-valued ``additionalProperties`` and ``type: object`` with no
``properties``. With ``api.clean_tool_schemas: true`` the client rewrites each tool schema
into a form those servers accept. This changes the prompt the server sees, so use it only
when the server rejects the original schemas.
"""

from typing import Any

from inference_perf.config import APIType

# Fields kept in a cleaned schema. Everything else (e.g. 'default', 'title', numeric and
# array constraints) is dropped.
_ALLOWED_JSON_SCHEMA_KEYS = {
    "type",
    "description",
    "enum",
    "properties",
    "required",
    "items",
    "anyOf",
    "oneOf",
    "allOf",
    "$ref",
    "const",
    "additionalProperties",
}


def clean_tool_schema(obj: Any, is_properties_map: bool = False) -> Any:
    """Recursively rewrite a tool parameters schema into the subset strict servers accept.

    is_properties_map: when True, obj is the value of a 'properties' key. Its keys are
    property names, not schema keywords, and are kept.
    """
    if isinstance(obj, dict):
        if is_properties_map:
            return {k: clean_tool_schema(v) for k, v in obj.items()}
        cleaned = {}
        for k, v in obj.items():
            if k not in _ALLOWED_JSON_SCHEMA_KEYS:
                continue
            # Strict servers require additionalProperties to be a boolean.
            if k == "additionalProperties" and not isinstance(v, bool):
                v = False
            cleaned[k] = clean_tool_schema(v, is_properties_map=(k == "properties"))
        # Strict servers require a plain 'type' and do not support anyOf/oneOf.
        # Collapse to 'type': 'string' when no type is given.
        if "type" not in cleaned:
            cleaned.pop("anyOf", None)
            cleaned.pop("oneOf", None)
            cleaned["type"] = "string"
        elif "anyOf" in cleaned or "oneOf" in cleaned:
            cleaned.pop("anyOf", None)
            cleaned.pop("oneOf", None)
        # Strict servers require 'type: object' schemas to include 'properties'.
        if cleaned.get("type") == "object" and "properties" not in cleaned:
            cleaned["properties"] = {}
        return cleaned
    if isinstance(obj, list):
        return [clean_tool_schema(v) for v in obj]
    return obj


def clean_request_tool_schemas(payload: dict[str, Any], api_type: APIType) -> None:
    """Clean the parameter schema of every tool in a request body, in place.

    Chat tools carry the schema at tools[].function.parameters; Anthropic Messages tools
    carry it at tools[].input_schema. Other API types and tools with no schema are left alone.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if api_type == APIType.Chat:
            function = tool.get("function")
            if isinstance(function, dict) and function.get("parameters") is not None:
                function["parameters"] = clean_tool_schema(function["parameters"])
        elif api_type == APIType.AnthropicMessages:
            if tool.get("input_schema") is not None:
                tool["input_schema"] = clean_tool_schema(tool["input_schema"])
