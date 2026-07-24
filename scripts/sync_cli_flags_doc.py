import argparse
import inspect
import sys
import typing
from pathlib import Path
from pydantic import BaseModel
from inference_perf.utils.cli_parser import add_pydantic_args
from inference_perf.config import Config

HEADER = """# Inference-Perf CLI Flags

These command line flags are automatically generated from the internal `Config` schema. You can override any configuration directly from the CLI without using a yaml configuration file.

| Flag | Type | Description |
| --- | --- | --- |
"""


def generate_doc() -> str:
    parser = argparse.ArgumentParser()
    docs = []
    add_pydantic_args(parser, Config, docs=docs)
    return HEADER + "\n".join(docs) + "\n"


def collect_nested_models(annotation: typing.Any, models: set[type[BaseModel]]) -> None:
    """Collects every Pydantic model referenced by a type annotation, unwrapping Optional/Union/List/Dict."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        models.add(annotation)
        return
    for arg in typing.get_args(annotation):
        collect_nested_models(arg, models)


def find_missing_descriptions(model_cls: type[BaseModel], visited: typing.Optional[set[type[BaseModel]]] = None) -> list[str]:
    """
    Recursively walks every model reachable from model_cls and returns the fields
    that lack a `Field(description=...)`, as "path/to/file.py: Model.field" strings.
    """
    if visited is None:
        visited = set()
    if model_cls in visited:
        return []
    visited.add(model_cls)

    missing = []
    source_file = Path(inspect.getfile(model_cls)).resolve().relative_to(Path.cwd())
    for name, field in model_cls.model_fields.items():
        if not field.description:
            missing.append(f"{source_file}: {model_cls.__name__}.{name}")
        nested: set[type[BaseModel]] = set()
        collect_nested_models(field.annotation, nested)
        for nested_model in sorted(nested, key=lambda m: m.__name__):
            missing.extend(find_missing_descriptions(nested_model, visited))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync CLI flags documentation.")
    parser.add_argument("--check", action="store_true", help="Fail if doc is out of sync.")
    args = parser.parse_args()

    doc_path = Path("docs/cli_flags.md")

    if not doc_path.exists():
        print(f"Error: {doc_path} does not exist.")
        sys.exit(1)

    missing_descriptions = find_missing_descriptions(Config)
    if missing_descriptions:
        print(f"Error: {len(missing_descriptions)} config field(s) lack a `Field(description=...)`:")
        for entry in missing_descriptions:
            print(f"  {entry}")
        print("Every config field must have a description so docs/cli_flags.md and --help stay meaningful.")
        sys.exit(1)

    expected_content = generate_doc()

    if args.check:
        with open(doc_path, "r") as f:
            current_content = f.read()

        if current_content != expected_content:
            print("Error: docs/cli_flags.md is out of sync with Config.")
            import difflib

            diff = difflib.unified_diff(
                current_content.splitlines(keepends=True),
                expected_content.splitlines(keepends=True),
                fromfile="current",
                tofile="expected",
            )
            sys.stdout.writelines(diff)
            print("Run `pdm run update:cli-flags` to update it.")
            sys.exit(1)
        else:
            print("docs/cli_flags.md is in sync.")
    else:
        with open(doc_path, "w") as f:
            f.write(expected_content)
        print("Updated docs/cli_flags.md")


if __name__ == "__main__":
    main()
