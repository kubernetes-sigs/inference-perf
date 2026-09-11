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
"""Golden-corpus validation tests.

Every directory under ``corpus/`` containing a ``validation.json`` is one
case: the report files of a single run, frozen at capture time, plus the
findings the validators are expected to produce for them. Each case loads
every report file (all ``.json``/``.yaml`` files except the golden), runs the
full default validator stack — exactly what ``validate_reports`` does in
production — and asserts the result matches the golden.

``validation.json`` doubles as the golden because a real run already emits it
next to the other report files, so capturing a case is copying a run's output
directory into ``corpus/``. The golden means "what current code is expected
to say about this frozen input", not "what the original run said" — see
``corpus/README.md`` for the capture workflow and ``pdm run update:goldens``
for regenerating goldens after an intentional behavior change.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest
import yaml

from inference_perf.reportgen.validation import default_validators, run_validators
from inference_perf.utils import ReportFile

CORPUS_DIR = Path(__file__).parent / "corpus"
GOLDEN_FILENAME = "validation.json"

_LOADERS: Dict[str, Callable[[str], Any]] = {
    ".json": json.loads,
    ".yaml": yaml.safe_load,
    ".yml": yaml.safe_load,
}


def corpus_cases() -> List[Path]:
    return sorted(p.parent for p in CORPUS_DIR.rglob(GOLDEN_FILENAME))


def load_case_reports(case_dir: Path) -> List[ReportFile]:
    """All report files of a case; the golden and non-report files are skipped."""
    reports: List[ReportFile] = []
    for path in sorted(case_dir.iterdir()):
        if path.name == GOLDEN_FILENAME or path.suffix not in _LOADERS:
            continue
        reports.append(
            ReportFile(
                name=path.stem,
                contents=_LOADERS[path.suffix](path.read_text()),
                file_type=path.suffix.lstrip("."),
            )
        )
    return reports


def _dump(value: Any) -> str:
    return json.dumps(value, indent=2) + "\n"


@pytest.mark.parametrize("case_dir", corpus_cases(), ids=lambda p: str(p.relative_to(CORPUS_DIR)))
def test_corpus_case(case_dir: Path, request: pytest.FixtureRequest) -> None:
    golden_path = case_dir / GOLDEN_FILENAME
    expected = json.loads(golden_path.read_text())

    result = run_validators(default_validators(), load_case_reports(case_dir))
    actual = result.model_dump(mode="json", by_alias=True)

    if actual == expected:
        return

    case_id = case_dir.relative_to(CORPUS_DIR)
    if request.config.getoption("--update-goldens"):
        golden_path.write_text(_dump(actual))
        print(f"updated golden: {case_id}/{GOLDEN_FILENAME}")
        return

    diff = "\n".join(
        difflib.unified_diff(
            _dump(expected).splitlines(),
            _dump(actual).splitlines(),
            fromfile=f"{case_id}/{GOLDEN_FILENAME} (expected)",
            tofile="current validator output",
            lineterm="",
        )
    )
    pytest.fail(
        f"Validator output for corpus case '{case_id}' no longer matches its golden.\n"
        f"{diff}\n"
        "If this behavior change is intentional, run `pdm run update:goldens` and review the git diff.\n"
        f"If this case's report format is no longer supported, delete its directory instead: {case_dir}",
        pytrace=False,
    )
