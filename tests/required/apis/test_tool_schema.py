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
"""Unit tests for the opt-in tool schema cleanup (api.clean_tool_schemas)."""

from typing import Any

from inference_perf.apis.tool_schema import clean_request_tool_schemas, clean_tool_schema
from inference_perf.config import APIType


# {"type": "string", "title": "Path", "default": "/", "minLength": 1} -> {"type": "string"}:
# keys outside the allowed set are dropped.
def test_drops_unsupported_keys() -> None:
    assert clean_tool_schema({"type": "string", "title": "Path", "default": "/", "minLength": 1}) == {"type": "string"}


# A property literally named "title" inside "properties" is a name, not a keyword, so it stays.
def test_keeps_property_names_that_look_like_keywords() -> None:
    schema = {"type": "object", "properties": {"title": {"type": "string", "default": "x"}}}
    assert clean_tool_schema(schema) == {"type": "object", "properties": {"title": {"type": "string"}}}


# {"anyOf": [{"type": "string"}, {"type": "null"}]} has no plain type -> {"type": "string"}.
def test_anyof_without_type_collapses_to_string() -> None:
    assert clean_tool_schema({"anyOf": [{"type": "string"}, {"type": "null"}]}) == {"type": "string"}


# {"oneOf": [...]} with no plain type -> {"type": "string"}.
def test_oneof_without_type_collapses_to_string() -> None:
    assert clean_tool_schema({"oneOf": [{"type": "integer"}, {"type": "string"}]}) == {"type": "string"}


# {"type": "integer", "anyOf": [...]} keeps its type and loses the anyOf.
def test_anyof_with_type_is_removed() -> None:
    assert clean_tool_schema({"type": "integer", "anyOf": [{"type": "integer"}]}) == {"type": "integer"}


# {"type": "integer", "oneOf": [...]} keeps its type and loses the oneOf.
def test_oneof_with_type_is_removed() -> None:
    assert clean_tool_schema({"type": "integer", "oneOf": [{"type": "integer"}]}) == {"type": "integer"}


# additionalProperties: {"type": "string"} -> false; a map-like object also gets "properties": {}.
def test_object_additional_properties_coerced_to_false() -> None:
    schema = {"type": "object", "additionalProperties": {"type": "string"}}
    assert clean_tool_schema(schema) == {"type": "object", "additionalProperties": False, "properties": {}}


# additionalProperties: true is already a boolean and is kept as true.
def test_boolean_additional_properties_kept() -> None:
    schema = {"type": "object", "properties": {}, "additionalProperties": True}
    assert clean_tool_schema(schema) == schema


# Nested array items are cleaned too: items.title is dropped.
def test_cleans_nested_items() -> None:
    schema = {"type": "array", "maxItems": 3, "items": {"type": "string", "title": "T"}}
    assert clean_tool_schema(schema) == {"type": "array", "items": {"type": "string"}}


# A Chat request's tools[].function.parameters is cleaned in place; messages are untouched.
def test_clean_request_chat_tools() -> None:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "title": "F"}}}],
    }
    clean_request_tool_schemas(payload, APIType.Chat)
    assert payload["tools"][0]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


# A Chat tool with parameters=None stays None (no schema to clean).
def test_clean_request_chat_tool_without_parameters() -> None:
    payload: dict[str, Any] = {"tools": [{"type": "function", "function": {"name": "f", "parameters": None}}]}
    clean_request_tool_schemas(payload, APIType.Chat)
    assert payload["tools"][0]["function"]["parameters"] is None


# An Anthropic Messages request's tools[].input_schema is cleaned in place.
def test_clean_request_anthropic_tools() -> None:
    payload: dict[str, Any] = {"tools": [{"name": "f", "input_schema": {"type": "object", "title": "F"}}]}
    clean_request_tool_schemas(payload, APIType.AnthropicMessages)
    assert payload["tools"][0]["input_schema"] == {"type": "object", "properties": {}}


# A Completion request is left alone even if it has a "tools" key.
def test_clean_request_ignores_other_api_types() -> None:
    payload: dict[str, Any] = {"tools": [{"function": {"parameters": {"title": "F"}}}]}
    clean_request_tool_schemas(payload, APIType.Completion)
    assert payload["tools"][0]["function"]["parameters"] == {"title": "F"}


# A request with no tools is unchanged.
def test_clean_request_without_tools() -> None:
    payload: dict[str, Any] = {"messages": []}
    clean_request_tool_schemas(payload, APIType.Chat)
    assert payload == {"messages": []}
