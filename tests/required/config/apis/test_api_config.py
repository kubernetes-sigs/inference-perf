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
"""Validity rules for ``inference_perf.config.apis``."""

from inference_perf.config import APIConfig, APIType, ResponseFormat, ResponseFormatType


def test_api_config_defaults() -> None:
    cfg = APIConfig()
    assert cfg.type == APIType.Completion
    assert cfg.streaming is False
    assert cfg.response_format is None


def test_response_format_json_schema_is_default() -> None:
    fmt = ResponseFormat(json_schema={"type": "object"})
    assert fmt.type == ResponseFormatType.JSON_SCHEMA
    assert fmt.to_api_format() == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": {"type": "object"},
        },
    }


def test_response_format_custom_name_in_api_format() -> None:
    fmt = ResponseFormat(name="my_schema", json_schema={"type": "object"})
    assert fmt.to_api_format()["json_schema"]["name"] == "my_schema"


def test_response_format_json_object() -> None:
    fmt = ResponseFormat(type=ResponseFormatType.JSON_OBJECT)
    assert fmt.to_api_format() == {"type": "json_object"}
