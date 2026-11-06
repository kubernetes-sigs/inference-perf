# BR0.2 report generation

Native emission of [llm-d-benchmark v0.2.1](https://github.com/llm-d/llm-d-benchmark/tree/main/benchmark-report) (BR0.2) partial reports alongside inference-perf's existing report formats. See [docs/br_v0_2.md](../../../../docs/br_v0_2.md) for user-facing documentation.

## Responsibility split

inference-perf writes only the fields it can speak to truthfully from the run itself: the schema `version`, the `run` block (a generated `uid`, an `eid` shared by every stage of the invocation, and the wall-clock `time` window of the stage), and the `results` block built from the actual request metrics — plus, for workloads that ran sessions (`session_lifecycle` reporting on), the session lifecycle metrics behind `results.session_performance`. Everything else (stack configuration, scenario, run metadata like `user`/`description`) is deliberately absent so a downstream composer (the llm-d-benchmark CLI, wrapper scripts, ad-hoc `yq` merges) can merge another producer's partial on top without any inference-perf field silently overwriting their data.

`results.session_performance` is omitted entirely for a stage that ran no sessions, so a non-session workload emits exactly the partial it emitted before that block was populated. `results.observability.components` carries one entry for the model server, labelled with the server type, from the stage's Prometheus summary; it needs `report.prometheus.per_stage`, and without it the block is absent.

Emission is unconditional: every run drops one `inference-perf.partial.stage_<n>.yaml` per stage, mirroring the existing per-stage lifecycle reports. The one config field, `report.benchmark_report.base_file`, names a partial BR0.2 document each stage's partial is built on. `fillability.py` decides which of its fields a base may set.

## File layout

| File | Owner | Purpose |
|------|-------|---------|
| `schema.py` | inference-perf | Facade that re-exports the BR0.2 models from the `llmd-benchmark-report` package. **Import from here**, not from the package directly, so a schema bump only touches this file. |
| `adapter.py` | inference-perf | `build_results(request_metrics, tokenizer, use_server_output_tokens, *, session_metrics, server_metrics)`: projects inference-perf `RequestLifecycleMetric`s — and, for session workloads, `SessionLifecycleMetric`s, and the stage's Prometheus summary — into a BR0.2 `Results` object. Pure function, no I/O. |
| `partial_report.py` | inference-perf | `build_partial_report` / `generate_run_uid` / `generate_experiment_eid`: assemble the per-stage partial dict (`version` + `run` + `results`) with `None` fields stripped so it deep-merges cleanly. |
| `fillability.py` | inference-perf | `FILLABILITY`: the class of every BR0.2 leaf (fillable, defaulted, written, reserved), plus the schema walk (`schema_leaves`, `document_leaves`) that tests hold it to. |
| `base_report.py` | inference-perf | `load_base_report` / `apply_base_report`: check a base report at config load and merge it under each stage's partial. |
| `__init__.py` | inference-perf | Re-exports the inference-perf-owned API surface (`build_results`, `build_partial_report`, `generate_run_uid`, `generate_experiment_eid`). |

## Schema dependency

The models come from [`llmd-benchmark-report`](https://pypi.org/project/llmd-benchmark-report/), pinned `>=0.2.1,<0.3` in `pyproject.toml`. Within a minor line the package only adds optional fields, so new 0.2.x releases need no change here. Moving to a new minor line means bumping the pin and adjusting `schema.py` for any renamed symbols.
