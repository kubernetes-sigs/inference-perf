import argparse
import sys
from pathlib import Path
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync CLI flags documentation.")
    parser.add_argument("--check", action="store_true", help="Fail if doc is out of sync.")
    args = parser.parse_args()

    doc_path = Path("docs/cli_flags.md")

    if not doc_path.exists():
        print(f"Error: {doc_path} does not exist.")
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
