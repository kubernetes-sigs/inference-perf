import argparse
import sys
from pathlib import Path
from typing import Any

from inference_perf.observability.metrics.coverage import coverage_problems, emitted_metric_names
from inference_perf.observability.metrics.registry import MetricSpec, MetricStability, always, exposition_name
from inference_perf.observability.metrics.sets import ALL_SPECS

HEADER = """# Inference-Perf Runtime Metrics

These are the Prometheus metrics inference-perf can export about its own runtime over an HTTP `/metrics` endpoint. They are distinct from the metrics inference-perf scrapes from the model server under test and from the benchmark result definitions in [metrics.md](./metrics.md).

This document is automatically generated from the metric specs under `inference_perf/observability/metrics/sets/`. Do not edit it by hand; run `pdm run update:runtime-metrics` after changing the specs.

Every metric the endpoint can expose has a row below, and nothing else can be exposed: `pdm run check:runtime-metrics` scrapes a registry built with all config gating bypassed and fails if that exposition and this table disagree in either direction. It runs inside `pdm run validate`, which is merge-blocking.

## Stability

Every metric declares a stability level, and that level is prepended to the metric's HELP text, so a scrape says what is promised without anyone having to find this file:

{levels}

**Every metric below is `ALPHA` today, and the whole set stays `ALPHA` through v0.7.0.** These names, labels and buckets are a first cut that we expect to refine while the endpoint gets used; nothing is promoted before v1.0.0, and promotion is per metric, one `stability=` in its spec, not a blanket graduation of the set. The level appears only in the HELP text, never in a metric name and never in a label, so promoting a metric later does not break the queries or dashboards written against it.

## Metrics

| Metric | Type | Stability | Labels | Exported | Description |
| --- | --- | --- | --- | --- | --- |
""".format(levels="\n".join(f"- `{level.value}`: {level.promise}" for level in MetricStability))


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
            f"| `{exposition_name(spec)}` | {spec.metric_type.__name__} | `{spec.stability.value}` | {labels} "
            f"| {exported_when(spec)} | {spec.documentation} |"
        )
    return HEADER + "\n".join(rows) + "\n"


def check_every_emitted_metric_is_documented() -> None:
    """Fail unless the doc's metrics are exactly the ones a scrape can show.

    This is what makes a row in the table mean "actually exported" rather than
    "someone remembered to declare it".
    """
    problems = coverage_problems()
    if problems:
        for problem in problems:
            print(f"Error: {problem}")
        sys.exit(1)
    print(f"All {len(emitted_metric_names())} exported runtime metrics are documented.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync runtime metrics documentation.")
    parser.add_argument("--check", action="store_true", help="Fail if doc is out of sync.")
    args = parser.parse_args()

    doc_path = Path("docs/runtime_metrics.md")

    expected_content = generate_doc()

    if args.check:
        check_every_emitted_metric_is_documented()
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
