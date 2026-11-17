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
"""Named workloads shipped with inference-perf, and the fields a run of one may not change."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Tuple

import yaml
from pydantic import BaseModel

from inference_perf.utils.cli_parser import unwrap_type

CATALOG_DIR = Path(__file__).resolve().parents[1] / "workload_catalog"
CONFIG_FILE = "inference-perf.yaml"
MASK_FILE = "mask.yaml"
LOCKED = "locked"
OPEN = "open"

# The config field that names the entry. It is what loads the mask, so no mask covers it.
WORKLOAD_FIELD = "workload"


class WorkloadCatalogError(ValueError):
    """A catalog entry is missing or malformed, or a run changed one of its locked fields."""


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    config: Dict[str, object]
    # Each locked field as a path of config keys, e.g. ("load", "stages").
    locked: Tuple[Tuple[str, ...], ...]


def entry_names() -> List[str]:
    return sorted(path.name for path in CATALOG_DIR.iterdir() if path.is_dir())


def load_entry(name: str, model: type[BaseModel]) -> CatalogEntry:
    """Read an entry's config and mask, and check the mask against the config model."""
    entry_dir = CATALOG_DIR / name
    # A name is one directory of the catalog, never a path that could leave it.
    if name != Path(name).name or not (entry_dir / CONFIG_FILE).is_file():
        raise WorkloadCatalogError(f"No workload catalog entry named {name!r}. Entries: {', '.join(entry_names())}.")
    config = _read_mapping(entry_dir / CONFIG_FILE)
    if WORKLOAD_FIELD in config:
        raise WorkloadCatalogError(f"{name}/{CONFIG_FILE} sets {WORKLOAD_FIELD!r}. An entry cannot load another entry.")
    if not (entry_dir / MASK_FILE).is_file():
        raise WorkloadCatalogError(f"Workload catalog entry {name!r} has no {MASK_FILE} saying which fields are locked.")
    mask = _read_mapping(entry_dir / MASK_FILE)
    problems = mask_problems(mask, model, skip=frozenset({WORKLOAD_FIELD}))
    if problems:
        raise WorkloadCatalogError(f"{name}/{MASK_FILE} is not a valid mask:\n" + "\n".join(f"  {p}" for p in problems))
    return CatalogEntry(name=name, config=config, locked=tuple(_locked_paths(mask, ())))


def mask_problems(mask: object, model: type[BaseModel], path: str = "", skip: frozenset[str] = frozenset()) -> List[str]:
    """Everything wrong with a mask, or an empty list. Every field of the model must be marked."""
    where = path or "the mask"
    if not isinstance(mask, dict):
        return [f"{where} must mark each field locked or open, not {mask!r}"]
    fields = [field for field in model.model_fields if field not in skip]
    problems = []
    missing = [field for field in fields if field not in mask]
    if missing:
        problems.append(f"{where} does not mark {', '.join(missing)} locked or open")
    for key, value in mask.items():
        dotted = f"{path}.{key}" if path else str(key)
        if key not in fields:
            problems.append(f"{dotted} is not a config field a mask can mark")
        elif isinstance(value, dict):
            section, _ = unwrap_type(model.model_fields[key].annotation)
            if isinstance(section, type) and issubclass(section, BaseModel):
                problems.extend(mask_problems(value, section, dotted))
            else:
                problems.append(f"{dotted} has no fields of its own. Mark the whole of it locked or open")
        elif value not in (LOCKED, OPEN):
            problems.append(f"{dotted} must be {LOCKED}, {OPEN} or its own fields each marked, not {value!r}")
    return problems


def changed_locked_fields(entry: CatalogEntry, required: Mapping[str, object], actual: Mapping[str, object]) -> List[str]:
    """One line per locked field the run changed, naming the value the entry requires.

    Both sides are full config dumps, so a field the entry leaves at its default
    compares against that default.
    """
    lines = []
    for path in entry.locked:
        for dotted, value in _differences(_value_at(required, path), _value_at(actual, path), ".".join(path)):
            lines.append(f"{dotted} must be {json.dumps(value)}")
    return lines


def _read_mapping(path: Path) -> Dict[str, object]:
    with open(path) as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise WorkloadCatalogError(f"{path} must hold a mapping, not {type(data).__name__}.")
    return data


def _locked_paths(mask: Mapping[str, object], prefix: Tuple[str, ...]) -> Iterator[Tuple[str, ...]]:
    for key, value in mask.items():
        if value == LOCKED:
            yield prefix + (key,)
        elif isinstance(value, dict):
            yield from _locked_paths(value, prefix + (key,))


def _value_at(tree: object, path: Tuple[str, ...]) -> object:
    for key in path:
        if not isinstance(tree, Mapping):
            return None
        tree = tree.get(key)
    return tree


# Narrows a difference to the leaves, so a locked section names the field that moved.
def _differences(required: object, actual: object, path: str) -> Iterator[Tuple[str, object]]:
    if isinstance(required, Mapping) and isinstance(actual, Mapping):
        for key in dict.fromkeys([*required, *actual]):
            yield from _differences(required.get(key), actual.get(key), f"{path}.{key}")
    elif required != actual:
        yield path, required
