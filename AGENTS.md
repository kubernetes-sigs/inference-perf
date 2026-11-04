# AGENTS.md — Contributor Guide for `inference-perf`

## Environment (always work inside the project env)

**Always run every command inside the project's PDM-managed virtual environment.** Never
install into or run against your system/global Python — the project pins Python `>= 3.12`
and an exact, locked dependency set (`pdm.lock`). `pdm run ...` automatically executes
inside this env, so prefer it over bare `python`/`pytest`.

**If the env doesn't exist yet, create it:**

```bash
# 1. Install PDM if you don't have it (see https://pdm-project.org)
pipx install pdm            # recommended
# or: pip install --user pdm
# or: curl -sSL https://pdm-project.org/install-pdm.py | python3 -

# 2. Point PDM at a Python 3.12+ interpreter and create the in-project venv
pdm use -f 3.12             # select the interpreter (installs .pdm-python / venv/)

# 3. Install all deps + dev/test groups and wire up git hooks
pdm run setup               # = pdm install + pre-commit install (pre-commit & pre-push)
```

`pdm install` creates the local `venv/` and writes `.pdm-python` (the pinned interpreter
path) if they're missing, so `pdm run setup` alone is usually enough on a fresh clone.

**Run things inside the env** (either form works):

```bash
pdm run <task>              # run a pyproject [tool.pdm.scripts] task, e.g. pdm run validate
pdm run python -m ...       # run an arbitrary command inside the env
eval "$(pdm venv activate)" # or activate the venv in your current shell
```

**Verify you're in the env** before running anything:

```bash
pdm info                    # shows the project interpreter & packages location
pdm run python -V           # must print Python 3.12.x or newer
```

> Nix users: `flake.nix` provides a dev shell (`nix develop`) with a matching toolchain;
> still run `pdm run setup` inside it to populate the venv.

## Commands (run these — in this order — before pushing)

All commands go through **PDM** and run **inside the project env** (see
[Environment](#environment-always-work-inside-the-project-env)). Run from the repo root.

```bash
pdm run setup          # one-time: install deps + pre-commit + pre-push hooks
pdm run test:picked    # run only tests for files you changed (fast inner loop)
pdm check              # fast: ruff format + ruff lint
pdm run validate       # full gate: format + lint + mypy --strict + CLI-flags doc sync
pdm run test           # full suite: pytest tests
pdm run check:cov      # coverage-regression check (fails if you lower coverage)
```

Other useful commands:

```bash
pdm run test:e2e             # end-to-end tests (e2e/ + tests/optional), parallel
pdm run update:cli-flags     # regenerate docs/cli_flags.md after adding a CLI flag
inference-perf --config_file config.yml           # run a benchmark from a config file
inference-perf --server.type vllm --server.base_url http://localhost:8000 \
  --data.type random --load.type constant \
  --load.stages '[{"rate": 10, "duration": 60}]' --api.streaming true
```

> `pdm run validate` runs `check:cli-flags` — if you added or renamed any config field,
> run `pdm run update:cli-flags` and commit the regenerated `docs/cli_flags.md`, or CI fails.

## Done means

Read `CONTRIBUTING.md` first. This section is the done-condition for a change.

1. `pdm run validate` is green: format, lint, strict mypy, generated-doc sync.
2. `pdm run test:picked` is green while iterating. `pdm run test` before opening a PR.
3. A change under `inference_perf/datagen`, `loadgen`, `client` or `reportgen` also passes `pdm run test:e2e` with `llm-d-inference-sim` on `PATH`. A skipped sim test is not a pass. `nix develop` provides the sim, Prometheus, pdm and Python. Without Nix, install the sim from its README.
4. A bug fix includes a test that fails on main before the fix. Quote the failing output in the PR.
5. `docs/cli_flags.md` and `docs/runtime_metrics.md` are generated. Run `pdm run update:cli-flags` or `pdm run update:runtime-metrics`. Do not edit them by hand.

## Tech Stack

Exact dependency versions live in `pyproject.toml` (locked in `pdm.lock`); check there, not here.

- **Python** `>= 3.12`.
- **PDM** — dependency & task runner (`pyproject.toml` `[tool.pdm.scripts]`). Not pip/poetry.
- **Pydantic** v2 — all config and payload models. v2 API only.
- **asyncio + uvloop** — request clients and load generation are fully async.
- **aiohttp** — HTTP model-server clients.
- **numpy** — metrics/distribution math.
- **datasets / transformers** — dataset loading & tokenization.
- **rich** — CLI summary tables.
- **prometheus-client**, **OpenTelemetry** (optional `otel` extra) — observability.
- **Tooling**: `ruff` (format + lint, line-length **127**, double quotes), `mypy --strict`,
  `pytest` + `pytest-asyncio` (`asyncio_mode = "auto"`), `hypothesis` (property tests).

## Repository Map

| Path | What lives here |
| :--- | :--- |
| `inference_perf/main.py` | **Entry point + factory.** Wires config → concrete clients/generators. Register new plugin types here. |
| `inference_perf/config/` | Pydantic config models & enums (`DataGenType`, `LoadType`, `ModelServerType`). Mirrors the runtime package layout. |
| `inference_perf/datagen/` | Data/trace generators. Extend `BaseGenerator`/`DataGenerator` in `datagen/base.py`. |
| `inference_perf/client/modelserver/` | Model-server clients (`vllm_client.py`, `sglang_client.py`, `tgi_client.py`, `openai_client.py`). Extend the base in `base.py`. |
| `inference_perf/client/filestorage/` | Report storage backends (local, GCS, S3). |
| `inference_perf/loadgen/` | Load patterns (constant, Poisson, concurrent) & multi-worker orchestration. |
| `inference_perf/metrics/` | Request collection & metrics aggregation. |
| `inference_perf/reportgen/` | Report generation (`reportgen/base.py`, `summarize_requests`). |
| `inference_perf/apis/` | Request/response lifecycle models (`InferenceInfo`, `RequestLifecycleMetric`, `ErrorResponseInfo`). |
| `inference_perf/payloads/` | Multimodal payload specs (text/image/video/audio). |
| `inference_perf/analysis/`, `observability/`, `circuit_breaker/`, `mediagen/` | Plotting, OTel/Prometheus, saturation triggers, media generation. |
| `docs/` | User-facing docs (`config.md`, `metrics.md`, `reports.md`, `design.md`, …). Update when behavior changes. |
| `tests/` | Test suite (see [Testing](#testing)). |
| `examples/`, `workload-catalog/`, `deploy/` | Sample configs, ready-made workloads, Kubernetes manifests. |
| `scripts/` | Repo automation (`sync_cli_flags_doc.py`, `check_coverage_regression.py`). |

## How Features Are Wired (the core pattern)

Adding a data generator, model-server client, or load type follows the same three-step
pattern. **Concrete implementation → enum member → factory branch in `main.py`.**

Example — the `DataGenType` enum (`inference_perf/config/datagen/config.py`) drives a
dispatch in `main.py`:

```python
# inference_perf/main.py  (abridged)
if config.data.type == DataGenType.ShareGPT:
    datagen = HFShareGPTDataGenerator(config.api, config.data, tokenizer)
elif config.data.type == DataGenType.Synthetic:
    datagen = SyntheticDataGenerator(config.api, config.data, tokenizer, seed=config.load.base_seed)
elif config.data.type == DataGenType.Random:
    datagen = RandomDataGenerator(config.api, config.data, tokenizer, seed=config.load.base_seed)
# ... one branch per DataGenType member
```

Every generator subclasses the base and declares its supported APIs:

```python
# inference_perf/datagen/base.py
class BaseGenerator(ABC):
    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        if api_config.type not in self.get_supported_apis():
            raise Exception(f"Unsupported API type {api_config}")
        ...

    @abstractmethod
    def get_supported_apis(self) -> List[APIType]:
        """Return list of supported API types (Chat, Completion, etc.)."""
        raise NotImplementedError
```

So to add a new data generator you:

1. Add a `DataGenType` member in `config/datagen/config.py` and any config fields it needs.
2. Implement the generator in `datagen/`, subclassing `DataGenerator`/`BaseGenerator`.
3. Add the dispatch branch in `main.py`.
4. Add tests under `tests/required/datagen/` and update `docs/config.md`.

Model-server clients (`ModelServerType` → `client/modelserver/`) and load patterns
(`LoadType` → `loadgen/`) follow the identical enum → factory-branch → base-class flow.

## Testing

Every contribution ships with tests. No exceptions for behavior changes.

- **Layout mirrors the source tree**: `tests/required/<package>/test_*.py`
  (e.g. `tests/required/reportgen/test_failure_reporting.py`,
  `tests/required/datagen/synthetic/test_synthetic_datagen.py`).
- **`tests/required/`** — always run in CI; fast, no network. Put your unit tests here.
- **`tests/optional/`** and **`e2e/`** — live/end-to-end tests needing a real server;
  run via `pdm run test:e2e`. Not part of the fast gate.
- **Naming** (`pyproject.toml`): files `test_*.py`, classes `Test*`, functions `test_*`.
- **Async tests need no decorator** — `asyncio_mode = "auto"` is set; just write `async def test_...`.
- Prefer **property-based tests with `hypothesis`** for generators/distributions
  (see `tests/required/datagen/synthetic/test_datagen_fuzz.py`).
- Use `unittest.mock.Mock` for clients/collectors so unit tests stay offline
  (see the `_make_generator()` helper in `test_failure_reporting.py`).
- **One test per condition in a boolean rule**: when code combines conditions (`a and b`, `a or b`),
  write a test for each condition, so deleting any one condition makes the suite fail.
- **No import-time state**: nothing a test imports may set state at import time — no `os.environ`
  writes and no cwd-relative paths at module level.
- **Comment every test**: every test function and helper gets a `#` block directly above the `def`, no blank line between:
  concrete inputs and expected outputs in one to three lines, written for someone who has not read the code.

Every test file starts with the Apache 2.0 license header (copy it from any existing file),
then a module docstring describing what is under test:

```python
# Copyright 2026 The Kubernetes Authors.
# ... (full Apache 2.0 header) ...
"""Unit tests for the error detail in the lifecycle reports."""
```

## Coding Conventions

- **License header**: every new `.py` file (source and test) begins with the Apache 2.0
  header — copy verbatim from an existing file.
- **Formatting/lint**: `ruff format` + `ruff check` are the source of truth. Line length
  **127**, double quotes, 4-space indent, respect magic trailing commas. Don't hand-format.
- **Typing**: fully type public functions; the code must pass `mypy --strict` over both
  `inference_perf/` and `tests/`. Follow the surrounding module's style for type annotations.
- **Config = Pydantic v2 models** under `inference_perf/config/`, kept in a package layout
  that mirrors the runtime code. New user-facing options are config fields, not ad-hoc args —
  they become CLI flags automatically (then run `pdm run update:cli-flags`).
- **Engine-agnostic**: keep changes engine-agnostic — the tool must produce apples-to-apples
  comparisons across vLLM, SGLang, TGI, and any OpenAI-compatible endpoint.
- **Match surrounding style**: comment density, naming, and idiom should look like the file
  you're editing.

## Boundaries

### Always do

- Run every command inside the project's PDM-managed env (`pdm run ...`); create it with `pdm run setup` if missing (see [Environment](#environment-always-work-inside-the-project-env)).
- Add tests under `tests/` for every behavior change and keep coverage from regressing (`pdm run check:cov`).
- Run `pdm run validate` and `pdm run test` before pushing.
- Keep the tool **engine-agnostic** — new features should work across model servers, or be clearly opt-in.
- Register new plugin types in **all three** places (enum, implementation, `main.py` factory).
- Update the relevant `docs/*.md` (and regenerate `docs/cli_flags.md`) when behavior or config changes.
- Base your branch on the latest `origin/main` (a pre-push hook enforces this).

### Ask first

- Adding or bumping a **dependency** in `pyproject.toml` (weigh the extra footprint; prefer optional extras).
- **Changing metric definitions or report JSON schema** — downstream consumers depend on these; discuss in an issue/PR first.
- **Breaking config changes** (renaming/removing fields, changing defaults) — coordinate and document the migration.
- Large refactors that cross package boundaries — open an issue and align before implementing.

### Never do

- Never merge a behavior change **without accompanying tests**.
- Never bypass the tooling — don't disable ruff/mypy rules, and don't add `# type: ignore`, `cast` or `Any` to get past mypy. Fix the type.
- Never edit or delete an existing test to make it pass. A test is pinned behavior. Say in the PR which behavior changed and why.
- Never regenerate a golden without saying which output changed.
- Never touch `scripts/`, `.github/`, `[tool.*]` in `pyproject.toml` or a golden directory in a PR about something else.
- Never hardcode engine-specific assumptions into shared code paths.
- Never commit generated run artifacts (the `reports-*/` directories) or secrets/credentials.
- Never edit `docs/cli_flags.md` or `docs/runtime_metrics.md` by hand — regenerate them with `pdm run update:cli-flags` /
  `pdm run update:runtime-metrics`.
- Never skip the CLA — Kubernetes projects require a signed [CLA](https://git.k8s.io/community/CLA.md).

## Contribution Flow (summary of CONTRIBUTING.md)

1. Fork; for a first contribution pick a **`good-first-issue`**.
2. Open an **issue** describing the change; get it assigned before opening a PR.
3. `pdm run setup` to install deps and hooks.
4. Implement + **add tests**; verify manually with a real benchmark run.
5. `pdm run test:picked` → `pdm check` → `pdm run validate` → `pdm run test` → `pdm run check:cov`.
6. Rebase onto latest `origin/main`, then open a PR (see [PR body](#pr-body)).

## PR body

`Part of #N` or `Fixes #N` on the first line. Old behavior, then new behavior. `Changes:` as bullets. Under 600 words.

Community: [#inference-perf](https://kubernetes.slack.com/?redir=%2Fmessages%2Finference-perf) on Kubernetes Slack. Governed by the [Kubernetes Code of Conduct](code-of-conduct.md).
