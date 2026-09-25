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
"""Checked examples in docs/*.md must be true.

The docs are hand-written, so this test keeps their examples honest. It reads
every markdown file under docs/ and checks two kinds of example:

- a table whose header row is registered in ``CHECKS``: every row must
  recompute to what it claims, or, for a "Rejected ..." table, be rejected;
- a ```yaml block on the line after ``<!-- checked-example -->``: it must load
  through the config model named by its single top-level key.

Tables with other headers are descriptive and not checked. Each registered
header must appear in some doc, so renaming one can't silently switch its
checks off. A config doc that adds a checked table registers its header here.
"""

import math
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Type

import numpy as np
import pytest
import yaml
from pydantic import BaseModel, ValidationError

from inference_perf.config import DataConfig, LoadConfig, StandardLoadStage
from inference_perf.config.datagen.multimodal import ImageDatagenConfig
from inference_perf.utils.numeric.expression import Expression, Predicate

DOCS = sorted((Path(__file__).resolve().parents[3] / "docs").glob("*.md"))

Row = List[str]


# Splits a markdown table row on unescaped pipes and unescapes '\|'. Input: '| `a \| b` | 60 |'.
# Expected: ['`a | b`', '60'].
def _cells(line: str) -> Row:
    parts = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [p.strip().replace("\\|", "|") for p in parts]


# The first backticked value in a cell, or the cell itself. Input: '`t >= 60`'. Expected: 't >= 60'.
def _code(cell: str) -> str:
    match = re.search(r"`([^`]*)`", cell)
    return match.group(1) if match else cell


# A number cell, where 'inf' and '-inf' mean infinity. Input: '-inf'. Expected: float('-inf').
def _number(cell: str) -> float:
    return float(cell.replace("inf", "Infinity"))


# Every table in one doc as (header, rows), skipping the '---' separator row.
def _tables(text: str) -> List[Tuple[Tuple[str, ...], List[Row]]]:
    tables: List[Tuple[Tuple[str, ...], List[Row]]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        is_header = lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*:?-{3,}", lines[i + 1]) is not None
        if not is_header:
            i += 1
            continue
        header = tuple(_cells(lines[i]))
        rows = []
        i += 2
        while i < len(lines) and lines[i].startswith("|"):
            rows.append(_cells(lines[i]))
            i += 1
        tables.append((header, rows))
    return tables


# Every ```yaml block that directly follows a '<!-- checked-example -->' line.
def _checked_yaml(text: str) -> List[str]:
    return re.findall(r"<!-- checked-example -->\n```yaml\n(.*?)```", text, flags=re.S)


# Mean of 20000 seeded draws must be within 2% of the claimed mean.
def _check_mean(row: Row) -> None:
    expr, claimed = _code(row[0]), float(row[1])
    draws = np.asarray(Expression(expr).sample(rng=np.random.default_rng(0), size=20000))
    assert math.isclose(float(draws.mean()), claimed, rel_tol=0.02), f"{expr}: mean {draws.mean():.4f}, doc says {claimed}"


# Proven bounds must equal the claimed lower and upper bounds.
def _check_bounds(row: Row) -> None:
    expr = _code(row[0])
    assert Expression(expr).bounds == (_number(row[1]), _number(row[2])), f"{expr}: bounds {Expression(expr).bounds}"


# A condition must become true at the claimed second (2 dp).
def _check_condition(row: Row) -> None:
    condition, ends_at = _code(row[0]), float(row[1])
    assert round(Predicate(condition).boundary, 2) == ends_at, condition


# A rate over a window of the given seconds must send exactly the claimed requests at the claimed mean (2 dp).
def _check_rate(row: Row) -> None:
    rate, window, requests, mean = _code(row[0]), int(row[1]), int(row[2]), float(row[3])
    stage = StandardLoadStage(rate=rate, duration=window)
    assert (stage.expected_requests, round(stage.mean_rate, 2)) == (requests, mean), f"{rate} over {window}s"


# Each example in a rejected table must raise when built the way its table says.
def _rejects(build: Callable[[str], object]) -> Callable[[Row], None]:
    def check(row: Row) -> None:
        with pytest.raises((ValueError, ValidationError)):
            build(_code(row[0]))

    return check


CHECKS: Dict[Tuple[str, ...], Callable[[Row], None]] = {
    ("Expression", "Mean"): _check_mean,
    ("Expression", "Lower", "Upper"): _check_bounds,
    ("Condition", "Ends at (s)"): _check_condition,
    ("Rejected expression", "Why"): _rejects(Expression),
    ("Rejected condition", "Why"): _rejects(Predicate),
    ("Rate", "Window (s)", "Requests", "Mean req/s"): _check_rate,
    ("Rejected rate", "Why"): _rejects(lambda raw: StandardLoadStage(rate=raw, duration=60)),
    ("Rejected insertion_point", "Why"): _rejects(lambda raw: ImageDatagenConfig(insertion_point=raw)),
}

MODELS: Dict[str, Type[BaseModel]] = {"load": LoadConfig, "data": DataConfig}


def _table_cases() -> List[Tuple[str, Tuple[str, ...], Row]]:
    return [
        (doc.name, header, row)
        for doc in DOCS
        for header, rows in _tables(doc.read_text())
        if header in CHECKS
        for row in rows
    ]


def _yaml_cases() -> List[Tuple[str, str]]:
    return [(doc.name, block) for doc in DOCS for block in _checked_yaml(doc.read_text())]


# Each row of each registered table holds. Test ids name the doc and the row.
@pytest.mark.parametrize(
    ("doc", "header", "row"),
    _table_cases(),
    ids=[f"{d}: {' '.join(r)}" for d, _h, r in _table_cases()],
)
def test_table_example_holds(doc: str, header: Tuple[str, ...], row: Row) -> None:
    CHECKS[header](row)


# Each checked YAML block has one top-level key, 'load' or 'data', and loads through that model.
@pytest.mark.parametrize(("doc", "block"), _yaml_cases() or [("none", "")], ids=lambda v: str(v)[:40])
def test_checked_yaml_example_loads(doc: str, block: str) -> None:
    if doc == "none":
        pytest.skip("no checked YAML examples yet")
    parsed = yaml.safe_load(block)
    assert len(parsed) == 1 and next(iter(parsed)) in MODELS, f"{doc}: unexpected top-level keys {list(parsed)}"
    key, value = next(iter(parsed.items()))
    MODELS[key].model_validate(value)


# Every registered header appears in some doc.
def test_every_checked_table_is_present() -> None:
    present = {header for doc in DOCS for header, _rows in _tables(doc.read_text())}
    assert set(CHECKS) <= present, f"missing tables: {set(CHECKS) - present}"
