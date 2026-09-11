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
"""Credential masking for the places a config is rendered for someone to read.

A run renders its config twice: once to the log at startup, once into the
``config.yaml`` of the report bundle. Both copies get shared. Pod logs go to a
cluster's log store, the report bundle is uploaded to GCS or S3, and the bug
report template asks for "the entire one printed by the benchmark run". None of
those places should carry the operator's API key.

Where the credentials are is read off the config schema rather than listed here,
so a new one is covered by declaring its type:

- a field typed ``SecretStr`` is a credential, and its value never appears.
- a field named ``headers`` is a request header map, and the values of the
  headers named in ``CREDENTIAL_HEADER_NAMES`` never appear. That map is
  free-form so the secret cannot live in its type, but the headers that carry
  one are well known.

Nothing else is touched. A rendered config still has to be good enough to
reproduce a run from and to attach to a bug report.
"""

from copy import deepcopy
from functools import cache
from typing import Any, Callable, Mapping, Tuple, Union, get_args, get_origin

from pydantic import BaseModel, SecretStr

REDACTED = "[REDACTED]"

# Request headers whose value is a credential, matched case-insensitively.
# api.headers is how a run authenticates against a gateway that wants something
# other than a bearer token, so it carries the same secrets as server.api_key.
CREDENTIAL_HEADER_NAMES = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "api-key",
        "x-api-key",
        "x-goog-api-key",
        "x-goog-iam-authorization-token",
        "cookie",
    }
)

# Name of the free-form header map on a config model.
_HEADER_FIELD_NAME = "headers"

_Path = Tuple[str, ...]


def _unwrap_optional(annotation: Any) -> Any:
    """The annotation with Optional stripped, or the annotation unchanged."""
    if get_origin(annotation) is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _nested_models(annotation: Any) -> list[type[BaseModel]]:
    """Every config model an annotation reaches, through Optional, list and dict."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    found: list[type[BaseModel]] = []
    for arg in get_args(annotation):
        found.extend(_nested_models(arg))
    return found


def _collect(
    model: type[BaseModel],
    prefix: _Path,
    chain: Tuple[type[BaseModel], ...],
    secrets: set[_Path],
    headers: set[_Path],
) -> None:
    # A model that can contain itself would recurse forever. Everything below the
    # repeat was already collected on the way in, so stopping loses nothing.
    if model in chain:
        return
    for name, field in model.model_fields.items():
        path = prefix + (name,)
        annotation = _unwrap_optional(field.annotation)
        if annotation is SecretStr:
            secrets.add(path)
            continue
        if name == _HEADER_FIELD_NAME:
            headers.add(path)
        for nested in _nested_models(annotation):
            _collect(nested, path, chain + (model,), secrets, headers)


@cache
def credential_locations(model: type[BaseModel]) -> Tuple[frozenset[_Path], frozenset[_Path]]:
    """Where credentials sit in a config model, as (secret fields, header maps).

    Each entry is a path of field names from the root of the model. Derived from
    the schema on first use and cached, so declaring a field ``SecretStr`` is all
    it takes for it to be redacted everywhere a config is rendered.
    """
    secrets: set[_Path] = set()
    headers: set[_Path] = set()
    _collect(model, (), (), secrets, headers)
    return frozenset(secrets), frozenset(headers)


def _apply(node: Any, path: _Path, transform: Callable[[Any], Any]) -> None:
    """Replace the value at path under node, in place, wherever the path exists.

    Descends into lists so a path through a repeated config section reaches every
    element of it.
    """
    if isinstance(node, list):
        for item in node:
            _apply(item, path, transform)
        return
    if not isinstance(node, dict) or path[0] not in node:
        return
    if len(path) == 1:
        node[path[0]] = transform(node[path[0]])
        return
    _apply(node[path[0]], path[1:], transform)


def _mask_credential(value: Any) -> Any:
    return REDACTED if value is not None else value


def _mask_credential_headers(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {name: (REDACTED if str(name).lower() in CREDENTIAL_HEADER_NAMES else header) for name, header in value.items()}


def redact(data: Mapping[str, Any], model: type[BaseModel]) -> dict[str, Any]:
    """A copy of a config mapping with every credential replaced by ``REDACTED``.

    Takes a mapping rather than a model so the same masking covers the raw config
    read from disk, which is logged before it has been validated, and the dump of
    a validated config. A config that sets no credential renders unchanged.
    """
    redacted = deepcopy(dict(data))
    secrets, headers = credential_locations(model)
    for path in secrets:
        _apply(redacted, path, _mask_credential)
    for path in headers:
        _apply(redacted, path, _mask_credential_headers)
    return redacted
