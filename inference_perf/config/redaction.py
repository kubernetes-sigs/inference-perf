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

- a field typed ``SecretStr`` is a credential, and its value never appears. The
  type can sit inside Optional, a union, Annotated, a list, a dict or a model
  that contains itself.
- a field named ``headers`` is a request header map, and the value of any header
  whose name contains one of ``CREDENTIAL_HEADER_FRAGMENTS`` never appears. That
  map is free-form so the secret cannot live in its type.

Nothing else is touched. A rendered config still has to be good enough to
reproduce a run from and to attach to a bug report.

A saved or dumped config loaded back would send a placeholder in place of each
credential, so ``Config`` rejects it using ``redacted_credentials``.
"""

import collections.abc
from copy import deepcopy
from functools import cache
from types import UnionType
from typing import Annotated, Any, Callable, Iterator, Mapping, NamedTuple, Tuple, Union, get_args, get_origin

from pydantic import BaseModel, SecretStr

REDACTED = "[REDACTED]"

# What pydantic itself writes for a SecretStr in repr and model_dump(mode="json").
_PYDANTIC_MASK = str(SecretStr("secret"))

# A header whose name contains one of these, ignoring case, carries a credential.
# Over-matching only hides a routing value, while a missed header leaks a key.
CREDENTIAL_HEADER_FRAGMENTS = frozenset({"auth", "key", "token", "secret", "cookie"})

# Name of the free-form header map on a config model.
_HEADER_FIELD_NAME = "headers"

# A path step into every item of a list or every value of a dict.
_EACH = "*"

_SEQUENCES = (list, tuple, set, frozenset, collections.abc.Sequence)
_MAPPINGS = (dict, collections.abc.Mapping)

_Path = Tuple[str, ...]


def _shapes(annotation: Any, steps: _Path = ()) -> Iterator[Tuple[Any, _Path]]:
    """Each type an annotation holds, with the container steps that lead to it.

    Looks through Optional, unions, Annotated, lists and dicts.
    """
    origin = get_origin(annotation)
    if origin is Annotated:
        yield from _shapes(get_args(annotation)[0], steps)
    elif origin is Union or origin is UnionType:
        for arg in get_args(annotation):
            yield from _shapes(arg, steps)
    elif origin in _SEQUENCES:
        for arg in get_args(annotation):
            if arg is not Ellipsis:
                yield from _shapes(arg, steps + (_EACH,))
    elif origin in _MAPPINGS and get_args(annotation):
        yield from _shapes(get_args(annotation)[-1], steps + (_EACH,))
    else:
        yield annotation, steps


class _Rules(NamedTuple):
    """Where credentials sit in one config model, as paths from that model."""

    secrets: Tuple[_Path, ...]
    headers: Tuple[_Path, ...]
    models: Tuple[Tuple[_Path, type[BaseModel]], ...]


@cache
def _rules(model: type[BaseModel]) -> _Rules:
    secrets: list[_Path] = []
    headers: list[_Path] = []
    models: list[Tuple[_Path, type[BaseModel]]] = []
    for name, field in model.model_fields.items():
        if name == _HEADER_FIELD_NAME:
            headers.append((name,))
        for shape, steps in _shapes(field.annotation):
            if isinstance(shape, type) and issubclass(shape, SecretStr):
                secrets.append((name,) + steps)
            elif isinstance(shape, type) and issubclass(shape, BaseModel):
                models.append(((name,) + steps, shape))
    return _Rules(tuple(secrets), tuple(headers), tuple(models))


def _locate(node: Any, path: _Path, where: _Path) -> Iterator[Tuple[Any, Any, _Path]]:
    """Every place path reaches under node, as (container, key, location)."""
    step, rest = path[0], path[1:]
    if step == _EACH and isinstance(node, list):
        items: list[Tuple[Any, Any]] = list(enumerate(node))
    elif step == _EACH and isinstance(node, dict):
        items = list(node.items())
    elif isinstance(node, dict) and step in node:
        items = [(step, node[step])]
    else:
        return
    for key, value in items:
        location = where + (str(key),)
        if rest:
            yield from _locate(value, rest, location)
        else:
            yield node, key, location


def _visit(
    node: Any,
    model: type[BaseModel],
    where: _Path,
    on_secret: Callable[[Any, Any, _Path], None],
    on_headers: Callable[[Any, Any, _Path], None],
) -> None:
    """Call back on every credential under node, following the data down the schema.

    The data sets the depth, so a model that contains itself is followed as far as
    the config goes.
    """
    rules = _rules(model)
    for path in rules.secrets:
        for container, key, location in _locate(node, path, where):
            on_secret(container, key, location)
    for path in rules.headers:
        for container, key, location in _locate(node, path, where):
            on_headers(container, key, location)
    for path, nested in rules.models:
        for container, key, location in _locate(node, path, where):
            _visit(container[key], nested, location, on_secret, on_headers)


def _is_credential_header(name: Any) -> bool:
    lowered = str(name).lower()
    return any(fragment in lowered for fragment in CREDENTIAL_HEADER_FRAGMENTS)


def _is_placeholder(value: Any) -> bool:
    return value in (REDACTED, _PYDANTIC_MASK)


def _mask_credential(value: Any) -> Any:
    # An empty value is no credential, so it is left as is.
    return REDACTED if value else value


def _mask_credential_headers(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {name: (_mask_credential(header) if _is_credential_header(name) else header) for name, header in value.items()}


def redact(data: Mapping[str, Any], model: type[BaseModel]) -> dict[str, Any]:
    """A copy of a config mapping with every credential replaced by ``REDACTED``.

    Takes a mapping rather than a model so the same masking covers the raw config
    read from disk, which is logged before it has been validated, and the dump of
    a validated config. A config that sets no credential renders unchanged.
    """
    redacted = deepcopy(dict(data))

    def mask(container: Any, key: Any, location: _Path) -> None:
        container[key] = _mask_credential(container[key])

    def mask_headers(container: Any, key: Any, location: _Path) -> None:
        container[key] = _mask_credential_headers(container[key])

    _visit(redacted, model, (), mask, mask_headers)
    return redacted


def redacted_credentials(data: Mapping[str, Any], model: type[BaseModel]) -> list[str]:
    """The credentials in a config mapping that hold a placeholder, as dotted paths.

    A placeholder is ``REDACTED`` or the mask pydantic writes for a ``SecretStr``.
    Header names are compared ignoring case and the last one wins, as in the request
    the client builds.
    """
    found: set[str] = set()

    def check(container: Any, key: Any, location: _Path) -> None:
        if _is_placeholder(container[key]):
            found.add(".".join(location))

    def check_headers(container: Any, key: Any, location: _Path) -> None:
        if not isinstance(container[key], dict):
            return
        sent = {str(name).lower(): (name, header) for name, header in container[key].items()}
        for name, header in sent.values():
            if _is_credential_header(name) and _is_placeholder(header):
                found.add(".".join(location + (str(name),)))

    _visit(dict(data), model, (), check, check_headers)
    return sorted(found)
