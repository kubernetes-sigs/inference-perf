# BR0.2 report generation

Native emission of [llm-d-benchmark v0.2.1](https://github.com/llm-d/llm-d-benchmark/tree/main/benchmark-report) (BR0.2) partial reports alongside inference-perf's existing report formats. See [docs/br_v0_2.md](../../../../docs/br_v0_2.md) for user-facing documentation.

## Responsibility split

inference-perf writes only the fields it can speak to truthfully from the run itself: the schema `version`, the `run` block (a generated `uid`, an `eid` shared by every stage of the invocation, and the wall-clock `time` window of the stage), and the `results` block built from the actual request metrics. Everything else (stack configuration, scenario, run metadata like `user`/`description`) is deliberately absent so a downstream composer (the llm-d-benchmark CLI, wrapper scripts, ad-hoc `yq` merges) can merge another producer's partial on top without any inference-perf field silently overwriting their data.

Emission is unconditional and has no config surface: every run drops one `inference-perf.partial.stage_<n>.yaml` per stage, mirroring the existing per-stage lifecycle reports.

## File layout

| File | Owner | Purpose |
|------|-------|---------|
| `schema.py` | inference-perf | Facade that re-exports the BR0.2 models from the `llmd-benchmark-report` package. **Import from here**, not from the package directly, so a schema bump only touches this file. |
| `adapter.py` | inference-perf | `build_results(request_metrics, tokenizer, use_server_output_tokens)`: projects inference-perf `RequestLifecycleMetric`s into a BR0.2 `Results` object. Pure function, no I/O. |
| `partial_report.py` | inference-perf | `build_partial_report` / `generate_run_uid` / `generate_experiment_eid`: assemble the per-stage partial dict (`version` + `run` + `results`) with `None` fields stripped so it deep-merges cleanly. |
| `__init__.py` | inference-perf | Re-exports the inference-perf-owned API surface (`build_results`, `build_partial_report`, `generate_run_uid`, `generate_experiment_eid`). |

## Schema dependency

The models come from [`llmd-benchmark-report`](https://pypi.org/project/llmd-benchmark-report/), pinned `>=0.2.1,<0.3` in `pyproject.toml`. Within a minor line the package only adds optional fields, so new 0.2.x releases need no change here. Moving to a new minor line means bumping the pin and adjusting `schema.py` for any renamed symbols.
