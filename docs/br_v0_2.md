# Benchmark Report v0.2 (BR0.2)

inference-perf always emits its slice of a [BR0.2 report](https://github.com/llm-d/llm-d-benchmark/blob/main/docs/benchmark_report.md)
per load stage, alongside the native lifecycle reports. This is the canonical
output format for "where are the run's results"; downstream composers (the
llm-d-benchmark CLI, wrapper scripts, ad-hoc `yq` merges) layer their own
partials on top to produce a complete BR0.2 document.

inference-perf only writes the fields it can speak to truthfully from the
run itself: the schema `version`, the `run.uid`, `run.eid`, and `run.time`
window, and the entire `results` block. Everything else (`scenario.stack`,
`scenario.load`, `run.{cid, pid, user, description, keywords}`,
observability beyond what inference-perf measures) is left absent so a
composer can merge another producer's partial on top without any
inference-perf field silently overwriting their data.

A composer can also hand those fields in before the run instead of merging
them afterwards. See [Building on a base report](#building-on-a-base-report).

## Output

A run with two stages produces:

```
inference-perf.partial.stage_0.yaml
inference-perf.partial.stage_1.yaml
```

alongside the native reports. Each file is a valid BR0.2 document on its
own (required schema fields are populated; optional sections are absent)
and can be consumed directly or merged with other partials. The emitted
`version` is `0.2.1`, the current point release of the BR0.2 schema family.

## What inference-perf fills

```yaml
version: "0.2.1"
run:
  uid: inference-perf-stage-0-a1b2c3d4
  eid: inference-perf-experiment-e5f6a7b8
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

The `requests` block also includes `request_size` in UTF-8 bytes and a `multimodal`
block for recorded image, video, and audio measurements. Counts are distributions
across successful requests, including zero for requests without that modality.
Filesize, pixels, aspect ratio, video frames, and audio duration describe individual
media instances. BR video pixels are summed across frames for each video, while
the native lifecycle report records pixels per frame. Failed requests are excluded
from these distributions.

The `throughput` block includes `image_rate`, `video_rate`, and `audio_rate` for
recorded modalities. Each rate is the successful media count divided by the same
first-send-to-last-completion window as the existing request and token rates,
including failed requests when determining the window. Modalities with no recorded
measurements are omitted; explicitly recorded zero counts remain zero. Rates are
omitted when there are no successful requests or the window has zero duration.

For workloads that ran sessions (OTel trace-replay agentic workloads),
`results.session_performance.sessions` is populated as well: session counts
(`total`, `succeeded`, `failed`), event totals (`total_events`,
`total_events_completed`, `total_events_cancelled`), `session_rate` (session
completions per second over the window spanning the sessions), and the
per-session distributions `session_duration`, `events_per_session`,
`events_cancelled_per_session`, `input_tokens_per_session`, and
`output_tokens_per_session`. These figures match the native session lifecycle
report for the same run; the token distributions depend on the same
request-level enrichment that report uses.

`total` counts every session the stage ran: sessions that ended without
recording a metric — still active or never dispatched when the stage was cut
short by `max_stage_duration` — are added to the ones that did, so a truncated
stage cannot read as a clean run. The remaining counts and distributions
describe the recorded sessions only.

The block is omitted entirely for a stage that ran no sessions, so partials from
non-session workloads are unchanged.

`run.uid` is generated per stage. A composer is free to overwrite it during
merge.

`run.eid` is generated once per inference-perf invocation and stamped on
every stage partial. It is BR0.2's own cross-report grouping mechanism
("common across benchmark reports from a particular experiment"), so the
per-stage files of a sweep are machine-readably one experiment rather than
being related only by filename. As with `uid`, a composer is free to
overwrite it during merge.

`run.time` is the stage's wall-clock window as recorded by the load
generator (stage start to stage end, including drain), not the span from
first request start to last request end.

## Building on a base report

`report.benchmark_report.base_file` points at a partial BR0.2 document that
every stage's partial is built on. The base carries what inference-perf cannot
measure, and inference-perf adds its own fields on top. The output keeps its
name, `inference-perf.partial.stage_<n>.yaml`, and the same base goes into
every stage.

```yaml
report:
  benchmark_report:
    base_file: base.yaml   # relative to the working directory
```

```yaml
# base.yaml
run:
  description: Nightly rate sweep against one vLLM replica.
  keywords: [nightly, vllm]
scenario:
  stack:
    - metadata:
        label: vllm-0
        cfg_id: vllm-gemma-270m
      standardized:
        kind: generic
        tool: vllm
        tool_version: 0.11.0
      native:
        args:
          --max-num-seqs: 256
```

Every leaf of the BR0.2 schema has one of four classes, listed in
`inference_perf/reportgen/br/v0_2/fillability.py`. A `Statistics` or
time-series block counts as one leaf.

| Class | Base may set it | inference-perf writes it | Examples |
|---|---|---|---|
| fillable | yes | never | `run.description`, `scenario.stack`, `results.component_health` |
| defaulted | yes | when the base does not | `run.eid` |
| written | no | always | `run.uid`, `run.time`, `results.request_performance.aggregate`, `results.session_performance` |
| reserved | no | not today | `results.request_performance.aggregate.requests.incomplete`, `results.request_performance.time_series`, `results.observability.drop_rate` |

`results.observability` takes keys the schema does not name, and those are
fillable. A leaf the table does not list, from a newer 0.2.x schema, is fillable
too, since inference-perf has no code that writes it.

The base is checked at config load, before any request is sent. Loading fails
when the base:

1. sets a written or reserved field (the error names each one),
2. has a key BR0.2 does not define,
3. has a null value,
4. is not valid BR0.2 once inference-perf's required fields are added, or
5. has a distribution with `min` above `max`, or a negative value.

Unquoted YAML timestamps in the base come out as ISO-8601 strings, like
inference-perf's own `run.time`. If the base stops loading during the run, every
stage emits its partial without it and the error is logged.

A field moves from fillable to written or reserved when inference-perf starts
measuring it. A base that sets it then fails at config load, and the release
notes say so. A field never moves back. Consumers read written and reserved
fields as measured by inference-perf, so a base can never set them.
`tests/reportgen/br/v0_2/fillability_ratchet.txt` lists every such field, and a
test fails if one becomes fillable.

Worked examples live in `tests/reportgen/br/v0_2/base_report_examples/`. Each
directory holds a `config.yaml` and the `base.yaml` it points at.

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

The merged document validates against `BenchmarkReportV021` in
`inference_perf/reportgen/br/v0_2/schema.py`.

## Schema source

The pydantic models come from the
[`llmd-benchmark-report`](https://pypi.org/project/llmd-benchmark-report/)
package, published from `llm-d/llm-d-benchmark`. inference-perf depends on
`>=0.2.1,<0.3`. Releases within a minor line only add optional fields, so a
partial that validates today keeps validating on every later 0.2.x.
