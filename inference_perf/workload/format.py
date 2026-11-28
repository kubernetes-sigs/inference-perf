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
"""A trace format: what the rows of one session mean.

A format answers that one question and returns data. It takes rows as plain
mappings, wherever they came from, and gives back a session: its records and
their delivery. It never opens a dataset, picks sessions, builds text, reads
a clock or knows an API: `reader.py` and `text.py` do those once for every
format.

A format is named by a class attribute, and a config picks it by that name or
by `package.module:Class` for a format defined outside this package.
"""

from __future__ import annotations

import gzip
import importlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Type

from inference_perf.workload.record import Session

Row = Mapping[str, object]


def read_json_rows(path: Path) -> Iterator[Tuple[int, Row]]:
    """Rows of a .jsonl or .json file, gzipped or not, as (line number, row).

    A .json file is one array of rows, or one row, and reports line 0. Rows
    are yielded one at a time so a reader can stop early.
    """
    name = path.name
    with gzip.open(path, "rt", encoding="utf-8") if name.endswith(".gz") else open(path, "r", encoding="utf-8") as f:
        # Path.suffix is ".gz" for a .json.gz file, so the test looks at the whole name
        if name.endswith((".json", ".json.gz")):
            text = f.read()
            if not text.strip():
                return
            try:
                blob = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: not JSON: {e}") from e
            for row in blob if isinstance(blob, list) else [blob]:
                if isinstance(row, dict):
                    yield 0, row
            return
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: not JSON: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected an object, got {type(row).__name__}")
            yield line_no, row


class TraceFormat(ABC):
    """One trace format. `session` is the only method a format has to write."""

    # The name a config uses to pick this format.
    name: ClassVar[str]
    # The row field that says which session a row belongs to. None when every
    # row is a session of its own.
    session_key: ClassVar[Optional[str]] = "session_id"
    # Hugging Face tables that hold part of a row, as {row field: table name}.
    # Their rows are joined on `join_key` for the sessions that were kept.
    side_tables: ClassVar[Mapping[str, str]] = {}
    join_key: ClassVar[Optional[str]] = None
    # Whether a recorded length counts the special tokens the tokenizer adds.
    counts_special_tokens: ClassVar[bool] = True
    # Tokens per prefix block when the config does not say.
    default_block_size: ClassVar[int] = 512
    # Which files of a directory are trace files.
    file_suffixes: ClassVar[Tuple[str, ...]] = (".jsonl", ".jsonl.gz", ".json", ".json.gz")

    def __init__(self, block_size: Optional[int] = None) -> None:
        self.block_size = block_size if block_size is not None else self.default_block_size

    @abstractmethod
    def session(self, key: str, rows: Sequence[Row]) -> Session:
        """The requests of the session `key`, read from its rows."""
        raise NotImplementedError

    def check(self, row: Row) -> None:
        """Raise ValueError for a row that cannot be used. Accepts every row by default."""
        return None

    def summary(self, key: str, rows: Sequence[Row]) -> Dict[str, object]:
        """What a session filter is given to decide on."""
        return {"session_id": key, "num_rows": len(rows)}

    def read(self, path: Path) -> Iterator[Tuple[int, Row]]:
        """The rows of one file with their line numbers. JSON lines or a JSON array by default."""
        return read_json_rows(path)


# name -> where the class lives, so naming a format does not import the others
_BUILT_IN: Dict[str, str] = {
    "Weka": "inference_perf.workload.formats.weka:WekaFormat",
}
_REGISTERED: Dict[str, Type[TraceFormat]] = {}


def register(format_class: Type[TraceFormat]) -> Type[TraceFormat]:
    """Make a format selectable by its `name`. Usable as a class decorator."""
    _REGISTERED[format_class.name] = format_class
    return format_class


def format_names() -> Sequence[str]:
    return sorted({*_BUILT_IN, *_REGISTERED})


def format_class(name: str) -> Type[TraceFormat]:
    """The format called `name`: a registered or built-in name, or `package.module:Class`."""
    found = _REGISTERED.get(name)
    if found is not None:
        return found
    where = _BUILT_IN.get(name, name)
    module_name, _, class_name = where.partition(":")
    if not class_name:
        raise ValueError(f"Unknown trace format {name!r}; known formats: {', '.join(format_names()) or 'none'}")
    loaded = getattr(importlib.import_module(module_name), class_name, None)
    if not (isinstance(loaded, type) and issubclass(loaded, TraceFormat)):
        raise ValueError(f"{where} is not a TraceFormat")
    return loaded
