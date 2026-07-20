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
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

_ASSETS = Path(__file__).parent.parent / "assets" / "synthetic_themes"

DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous agent. Use the available tools to complete the given task, "
    "reason step by step, and produce a concise final answer. Prefer read-only actions first."
)


class Theme(BaseModel):
    name: str
    system_prompt: Optional[str] = None
    verbs: list[str]
    entities: dict[str, list[str]]
    enumerated: dict[str, str] = {}
    tool_names: list[str]
    result_templates: dict[str, str]
    objective_template: str
    followup_templates: list[str] = []
    followup_connectives: list[str] = []


def _validate(theme: Theme) -> Theme:
    if not theme.verbs:
        raise ValueError(f"theme {theme.name}: 'verbs' must be non-empty")
    if not theme.tool_names:
        raise ValueError(f"theme {theme.name}: 'tool_names' must be non-empty")
    if "default" not in theme.result_templates:
        raise ValueError(f"theme {theme.name}: 'result_templates' must include a 'default' key")
    return theme


def load_theme(name: str) -> Theme:
    path = _ASSETS / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Unknown synthetic theme {name!r} (looked in {_ASSETS})")
    data = json.loads(path.read_text())
    return _validate(Theme(**data))


GENERIC_THEME = Theme(
    name="generic",
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    verbs=["Analyze", "Investigate", "Review"],
    entities={"target": ["service-a", "service-b", "service-c"]},
    enumerated={"item": "ITEM"},
    tool_names=["get_status", "get_metrics", "run_check"],
    result_templates={"default": "result for {entity}: value={n0} at {t0}"},
    objective_template="{verb} the {target} incident: find the cause and recommend a fix.",
    followup_templates=["What about {target}?"],
    followup_connectives=["Following up, ", "Next, "],
)
