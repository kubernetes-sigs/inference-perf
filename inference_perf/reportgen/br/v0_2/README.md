# BR0.2 report generation

Native emission of [llm-d-benchmark v0.2.0](https://github.com/llm-d/llm-d-benchmark/tree/main/llmdbenchmark/analysis/benchmark_report) (BR0.2) reports alongside inference-perf's existing report formats.

## Responsibility split

inference-perf owns the BR0.2 `results` section only (the performance measurements derived from the run). Everything else — stack configuration, run/scenario metadata — is supplied by the user as a **partial report** (a BR0.2-shaped YAML/JSON file with `results` omitted). At report time, inference-perf loads the partial, fills `results`, validates the merged document against the BR0.2 schema, and emits one report per stage.

## File layout

| File | Owner | Purpose |
|------|-------|---------|
| `base.py` | **Vendored** from upstream | `BenchmarkReport` base class, `Units` / `WorkloadGenerator` enums, unit-group constants. |
| `schema_v0_2.py` | **Vendored** from upstream | Top-level BR0.2 pydantic models (`Run`, `Scenario`, `Results`, `Statistics`, etc.). |
| `schema_v0_2_components.py` | **Vendored** from upstream | Component subtype hierarchy (`ComponentStandardizedBase` + concrete kinds). |
| `schema.py` | inference-perf | Facade that re-exports every public symbol from the vendored files. **Import from here**, not from the vendored files directly — a schema bump should only touch the vendored files. |
| `adapter.py` | inference-perf | `build_results(request_metrics, tokenizer)` — projects inference-perf `RequestLifecycleMetric`s into a BR0.2 `Results` object. Pure function, no I/O. |
| `partial_report.py` | inference-perf | `load_partial_report` / `validate_partial_report` / `merge_results` — load a user-supplied partial (local or GCS), reject anything that already populates `results`, and merge with the adapter's output. |
| `__init__.py` | inference-perf | Re-exports the inference-perf-owned API surface (`build_results`, `merge_results`, `load_partial_report`, `validate_partial_report`, `PartialReportError`). |

## Resyncing the vendored schema

The three vendored files map 1:1 to upstream files in `llmdbenchmark/analysis/benchmark_report/`. Each has a header pinning the upstream commit SHA. To bump the BR0.2 schema:

1. Copy the three upstream files over `base.py`, `schema_v0_2.py`, `schema_v0_2_components.py`.
2. Update the SHA in each header.
3. Adjust `schema.py` if new public symbols were added upstream.
4. Re-run `tests/reportgen/br/v0_2/`.

Keeping the file split matches the upstream layout, so a resync is a plain copy rather than a three-way merge.
