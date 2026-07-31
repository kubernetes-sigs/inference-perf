# BR0.2 report generation

Native emission of [llm-d-benchmark v0.2](https://github.com/llm-d/llm-d-benchmark/tree/main/llmdbenchmark/analysis/benchmark_report) (BR0.2) partial reports alongside inference-perf's existing report formats. See [docs/br_v0_2.md](../../../../docs/br_v0_2.md) for user-facing documentation.

## Responsibility split

inference-perf writes only the fields it can speak to truthfully from the run itself: the schema `version`, the `run` block (a generated `uid` plus the wall-clock `time` window of the stage), and the `results` block built from the actual request metrics. Everything else (stack configuration, scenario, run metadata like `eid`/`user`/`description`) is deliberately absent so a downstream composer (the llm-d-benchmark CLI, wrapper scripts, ad-hoc `yq` merges) can merge another producer's partial on top without any inference-perf field silently overwriting their data.

Emission is unconditional and has no config surface: every run drops one `inference-perf.partial.stage_<n>.yaml` per stage, mirroring the existing per-stage lifecycle reports.

## File layout

| File | Owner | Purpose |
|------|-------|---------|
| `base.py` | **Vendored** from upstream | `BenchmarkReport` base class, `Units` / `WorkloadGenerator` enums, unit-group constants. |
| `schema_v0_2.py` | **Vendored** from upstream | Top-level BR0.2 pydantic models (`Run`, `Scenario`, `Results`, `Statistics`, etc.). |
| `schema_v0_2_components.py` | **Vendored** from upstream | Component subtype hierarchy (`ComponentStandardizedBase` + concrete kinds). |
| `schema.py` | inference-perf | Facade that re-exports every public symbol from the vendored files. **Import from here**, not from the vendored files directly; a schema bump should only touch the vendored files. |
| `adapter.py` | inference-perf | `build_results(request_metrics, tokenizer, use_server_output_tokens)`: projects inference-perf `RequestLifecycleMetric`s into a BR0.2 `Results` object. Pure function, no I/O. |
| `partial_report.py` | inference-perf | `build_partial_report` / `generate_run_uid`: assemble the per-stage partial dict (`version` + `run` + `results`) with `None` fields stripped so it deep-merges cleanly. |
| `__init__.py` | inference-perf | Re-exports the inference-perf-owned API surface (`build_results`, `build_partial_report`, `generate_run_uid`). |

## Resyncing the vendored schema

The three vendored files map 1:1 to upstream files in `llmdbenchmark/analysis/benchmark_report/`. Each has a header pinning the upstream commit SHA. To bump the BR0.2 schema:

1. Copy the three upstream files over `base.py`, `schema_v0_2.py`, `schema_v0_2_components.py`.
2. Update the SHA in each header.
3. Adjust `schema.py` if new public symbols were added upstream.
4. Re-run `tests/reportgen/br/v0_2/`.

Keeping the file split matches the upstream layout, so a resync is a plain copy rather than a three-way merge.
