import argparse
import sys
from pathlib import Path
from typing import Any

from prometheus_client import Counter

from inference_perf.observability.metrics.registry import MetricSpec, always
from inference_perf.observability.metrics.sets import ALL_SPECS

HEADER = """# Inference-Perf Runtime Metrics

These are the Prometheus metrics inference-perf can export about its own runtime over an HTTP `/metrics` endpoint. They are distinct from the metrics inference-perf scrapes from the model server under test and from the benchmark result definitions in [metrics.md](../../../docs/metrics.md).

This document is automatically generated from the metric specs under `inference_perf/observability/metrics/sets/`. Do not edit it by hand; run `pdm run update:runtime-metrics` after changing the specs.

| Metric | Type | Labels | Exported | Description |
| --- | --- | --- | --- | --- |
"""


def exposition_name(spec: MetricSpec[Any]) -> str:
    # prometheus_client appends _total to Counter sample names.
    if spec.metric_type is Counter:
        return f"{spec.name}_total"
    return spec.name


def exported_when(spec: MetricSpec[Any]) -> str:
    if spec.enabled is always:
        return "Always"
    doc = spec.enabled.__doc__
    if not doc:
        print(
            f"Error: the enabled predicate for metric {spec.name!r} has no docstring. "
            "Conditional specs in ALL_SPECS must use a named predicate whose docstring "
            "describes the condition, so this doc can be generated."
        )
        sys.exit(1)
    return doc.strip().splitlines()[0]


def generate_doc() -> str:
    rows = []
    for spec in ALL_SPECS:
        labels = ", ".join(f"`{label}`" for label in spec.labelnames) or "none"
        rows.append(
            f"| `{exposition_name(spec)}` | {spec.metric_type.__name__} | {labels} "
            f"| {exported_when(spec)} | {spec.documentation} |"
        )
    return HEADER + "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync runtime metrics documentation.")
    parser.add_argument("--check", action="store_true", help="Fail if doc is out of sync.")
    args = parser.parse_args()

    doc_path = Path("inference_perf/observability/metrics/runtime_metrics.md")

    expected_content = generate_doc()

    if args.check:
        if not doc_path.exists():
            print(f"Error: {doc_path} does not exist. Run `pdm run update:runtime-metrics` to create it.")
            sys.exit(1)

        with open(doc_path, "r") as f:
            current_content = f.read()

        if current_content != expected_content:
            print(f"Error: {doc_path} is out of sync with the metric specs.")
            import difflib

            diff = difflib.unified_diff(
                current_content.splitlines(keepends=True),
                expected_content.splitlines(keepends=True),
                fromfile="current",
                tofile="expected",
            )
            sys.stdout.writelines(diff)
            print("Run `pdm run update:runtime-metrics` to update it.")
            sys.exit(1)
        else:
            print(f"{doc_path} is in sync.")
    else:
        with open(doc_path, "w") as f:
            f.write(expected_content)
        print(f"Updated {doc_path}")


if __name__ == "__main__":
    main()
