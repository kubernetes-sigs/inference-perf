"""Generate docs/config-reference.md from the Pydantic models in inference_perf.config.

Discovers every BaseModel reachable from `Config`, emits one section per model with a
table of fields (type, default, description), and enforces that every field has a
non-empty description. Run via:

    pdm run update:config-docs   # regenerate the doc
    pdm run check:config-docs    # fail if doc is stale or descriptions are missing
"""

from __future__ import annotations

import argparse
import difflib
import sys
import typing
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from inference_perf import config as config_module
from inference_perf.config import Config

DOC_PATH = Path("docs/config-reference.md")

HEADER = """# Configuration Reference

<!--
  This file is auto-generated from the Pydantic models in inference_perf/config.py.
  Do not edit by hand. Run `pdm run update:config-docs` after changing the schema.
-->

This reference enumerates every configuration option, generated directly from the
schema. For tutorial-style examples and prose, see [config.md](config.md).

Each section corresponds to one model. Nested models are linked.

"""


def _model_anchor(model_cls: type[BaseModel]) -> str:
    return model_cls.__name__.lower()


def _is_basemodel(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _is_enum(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, Enum)


def _collect_models(root: type[BaseModel], seen: dict[str, type[BaseModel]]) -> None:
    if root.__name__ in seen:
        return
    seen[root.__name__] = root
    for field in root.model_fields.values():
        for tp in _iter_annotated_types(field.annotation):
            if _is_basemodel(tp):
                _collect_models(tp, seen)


def _iter_annotated_types(annotation: Any) -> list[Any]:
    """Walk a type annotation and yield every concrete leaf type it references."""
    out: list[Any] = []
    stack: list[Any] = [annotation]
    while stack:
        tp = stack.pop()
        if tp is None or tp is type(None):
            continue
        origin = typing.get_origin(tp)
        if origin is not None:
            stack.extend(typing.get_args(tp))
            continue
        out.append(tp)
    return out


def _render_type(annotation: Any) -> str:
    """Render a type annotation as readable markdown."""
    if annotation is None or annotation is type(None):
        return "None"

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        rendered = " \\| ".join(_render_type(a) for a in non_none)
        if len(non_none) < len(args):
            rendered = f"{rendered} (optional)"
        return rendered

    if origin in (list, typing.List):
        inner = _render_type(args[0]) if args else "Any"
        return f"List[{inner}]"

    if origin in (dict, typing.Dict):
        if len(args) == 2:
            return f"Dict[{_render_type(args[0])}, {_render_type(args[1])}]"
        return "Dict"

    if origin in (tuple, typing.Tuple):
        return "Tuple[" + ", ".join(_render_type(a) for a in args) + "]"

    if typing.get_origin(annotation) is typing.Literal or origin is typing.Literal:
        return "Literal[" + ", ".join(repr(a) for a in args) + "]"

    if _is_basemodel(annotation):
        return f"[{annotation.__name__}](#{_model_anchor(annotation)})"

    if _is_enum(annotation):
        values = " \\| ".join(repr(e.value) for e in annotation)
        return f"{annotation.__name__} ({values})"

    if isinstance(annotation, type):
        return annotation.__name__

    return str(annotation).replace("typing.", "")


def _render_default(field: FieldInfo) -> str:
    from pydantic_core import PydanticUndefined

    if field.is_required():
        return "**required**"
    default = field.default
    if default is PydanticUndefined:
        if field.default_factory is not None:
            return "_factory_"
        return "**required**"
    if isinstance(default, Enum):
        return f"`{default.value!r}`"
    if isinstance(default, BaseModel):
        return f"_default {type(default).__name__}_"
    if default is None:
        return "`None`"
    if isinstance(default, (list, dict)) and not default:
        return f"`{default!r}`"
    return f"`{default!r}`"


def _render_model(model_cls: type[BaseModel]) -> str:
    lines: list[str] = []
    lines.append(f"## {model_cls.__name__}")
    lines.append("")
    doc = (model_cls.__doc__ or "").strip()
    if doc:
        lines.append(doc)
        lines.append("")
    if not model_cls.model_fields:
        lines.append("_No fields._")
        lines.append("")
        return "\n".join(lines)
    lines.append("| Field | Type | Default | Description |")
    lines.append("| --- | --- | --- | --- |")
    for name, field in model_cls.model_fields.items():
        type_str = _render_type(field.annotation)
        default_str = _render_default(field)
        desc = (field.description or "").strip().replace("\n", " ")
        lines.append(f"| `{name}` | {type_str} | {default_str} | {desc} |")
    lines.append("")
    return "\n".join(lines)


def _find_missing_descriptions(models: dict[str, type[BaseModel]]) -> list[str]:
    """Return a list of `Model.field` for every field missing a description."""
    missing: list[str] = []
    for model_name, model_cls in models.items():
        for field_name, field in model_cls.model_fields.items():
            desc = (field.description or "").strip()
            if not desc:
                missing.append(f"{model_name}.{field_name}")
    return missing


def generate_doc() -> tuple[str, list[str]]:
    """Return (markdown, missing_descriptions)."""
    models: dict[str, type[BaseModel]] = {}
    _collect_models(Config, models)

    # Ensure deterministic ordering: Config first, then in declaration order from the
    # config module (falls back to alphabetic for anything else).
    declared_order = [name for name in dir(config_module) if name in models]
    ordered: list[type[BaseModel]] = [Config]
    for name in declared_order:
        if name == "Config":
            continue
        ordered.append(models[name])
    # Any models reachable but not declared in config module (defensive).
    seen_names = {m.__name__ for m in ordered}
    for name in sorted(models.keys() - seen_names):
        ordered.append(models[name])

    parts = [HEADER]
    parts.append("## Index\n")
    for m in ordered:
        parts.append(f"- [{m.__name__}](#{_model_anchor(m)})")
    parts.append("")
    parts.append("")
    for m in ordered:
        parts.append(_render_model(m))

    missing = _find_missing_descriptions(models)
    return "\n".join(parts).rstrip() + "\n", missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync config reference documentation.")
    parser.add_argument("--check", action="store_true", help="Fail if doc is out of sync or descriptions missing.")
    args = parser.parse_args()

    expected, missing = generate_doc()

    if args.check:
        if missing:
            print("Error: the following config fields are missing a description=:")
            for field_path in missing:
                print(f"  - {field_path}")
            print(
                "\nAdd `description=...` to each Field(...) declaration in inference_perf/config.py "
                "so it appears in the generated reference."
            )
            sys.exit(1)

        if not DOC_PATH.exists():
            print(f"Error: {DOC_PATH} does not exist. Run `pdm run update:config-docs`.")
            sys.exit(1)

        current = DOC_PATH.read_text()
        if current != expected:
            print(f"Error: {DOC_PATH} is out of sync with inference_perf/config.py.")
            diff = difflib.unified_diff(
                current.splitlines(keepends=True),
                expected.splitlines(keepends=True),
                fromfile="current",
                tofile="expected",
            )
            sys.stdout.writelines(diff)
            print("\nRun `pdm run update:config-docs` to regenerate.")
            sys.exit(1)
        print(f"{DOC_PATH} is in sync.")
    else:
        if missing:
            print("Warning: the following config fields are missing a description=:")
            for field_path in missing:
                print(f"  - {field_path}")
            print()
        DOC_PATH.write_text(expected)
        print(f"Updated {DOC_PATH}")


if __name__ == "__main__":
    main()
