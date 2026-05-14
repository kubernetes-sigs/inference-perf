# Benchmark Report v0.2 (BR0.2)

inference-perf always emits its slice of a [BR0.2 report](https://github.com/llm-d/llm-d-benchmark/blob/main/docs/benchmark_report.md)
per load stage, alongside the native lifecycle reports. This is the canonical
output format for "where are the run's results"; downstream composers (the
llm-d-benchmark CLI, wrapper scripts, ad-hoc `yq` merges) layer their own
partials on top to produce a complete BR0.2 document.

inference-perf only writes the fields it can speak to truthfully from the
run itself: the schema `version`, the `run.uid` and `run.time` window, and
the entire `results` block. Everything else (`scenario.stack`,
`scenario.load`, `run.{eid, cid, pid, user, description, keywords}`,
observability beyond what inference-perf measures) is left absent so a
composer can merge another producer's partial on top without any
inference-perf field silently overwriting their data.

## Output

A run with two stages produces:

```
inference-perf.partial.stage_0.yaml
inference-perf.partial.stage_1.yaml
```

alongside the native reports. Each file is a valid BR0.2 document on its
own (required schema fields are populated; optional sections are absent)
and can be consumed directly or merged with other partials.

## What inference-perf fills

```yaml
version: "0.2"
run:
  uid: inference-perf-stage-0-a1b2c3d4
  time:
    start: "2026-05-20T14:30:12.123+00:00"
    end:   "2026-05-20T14:30:17.456+00:00"
    duration: "PT5.333S"
results:
  request_performance:
    aggregate:
      requests: { ... }
      latency:  { ... }
      throughput: { ... }
```

`results.request_performance.aggregate.{requests, latency, throughput}` is
populated with the BR0.2 percentile set (`p0p1, p1, p5, p10, p25, p50, p75,
p90, p95, p99, p99p9`) and the unit annotations the schema validators
require.

`run.uid` is generated per stage. A composer is free to overwrite it during
merge.

## Composing with other producers

The emitted partial is designed for clean deep-merge. `None`-valued fields
are stripped entirely so a merge never overwrites a real value with `null`;
datetimes are ISO-8601 strings; there are no YAML anchors, tags, or aliases.

The canonical merge with `yq`:

```bash
# Compose inference-perf's partial with a stack/scenario partial from
# another producer (e.g. llm-d-benchmark gathering the live cluster state).
yq '. * load("infra.partial.yaml")' inference-perf.partial.stage_0.yaml \
  > benchmark_report.yaml
```

Or in Python:

```python
import yaml

with open("inference-perf.partial.stage_0.yaml") as f:
    a = yaml.safe_load(f)
with open("infra.partial.yaml") as f:
    b = yaml.safe_load(f)

def deep_merge(x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        return {k: deep_merge(x.get(k), y[k]) if k in x else y[k] for k in y} \
            | {k: v for k, v in x.items() if k not in y}
    return y if y is not None else x

merged = deep_merge(a, b)
```

The merged document validates against `BenchmarkReportV02` in
`inference_perf/reportgen/br/v0_2/schema.py`.

## Schema source

The pydantic models in
`inference_perf/reportgen/br/v0_2/{base,schema_v0_2,schema_v0_2_components}.py`
are vendored from `llm-d/llm-d-benchmark`. The header of each file pins the
upstream commit SHA. To resync after a BR0.2 schema bump, replace those
three files from upstream and re-run
`tests/reportgen/br/v0_2/test_schema_fixture.py` to confirm round-trip
validation against the upstream example still holds.
