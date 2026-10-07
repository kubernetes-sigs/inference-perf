# Working in this repo

Read `CONTRIBUTING.md` first. This file is the done-condition for a change.

## Done means

1. `pdm run validate` is green: format, lint, strict mypy, generated-doc sync.
2. `pdm run test:picked` is green while iterating. `pdm run test` before opening a PR.
3. A change under `inference_perf/datagen`, `loadgen`, `client` or `reportgen` also passes `pdm run test:e2e` with `llm-d-inference-sim` on `PATH`. A skipped sim test is not a pass. `nix develop` provides the sim, Prometheus, pdm and Python. Without Nix, install the sim from its README.
4. A bug fix includes a test that fails on main before the fix. Quote the failing output in the PR.
5. `docs/cli_flags.md` and `docs/runtime_metrics.md` are generated. Run `pdm run update:cli-flags` or `pdm run update:runtime-metrics`. Do not edit them by hand.

## Do not

- Add `# type: ignore`, `cast` or `Any` to get past mypy. Fix the type.
- Edit or delete an existing test to make it pass. A test is pinned behavior. Say in the PR which behavior changed and why.
- Regenerate a golden without saying which output changed.
- Touch `scripts/`, `.github/`, `[tool.*]` in `pyproject.toml` or a golden directory in a PR about something else.

## Tests

Every test function and helper gets a `#` block directly above the `def`, no blank line between: concrete inputs and expected outputs in one to three lines, written for someone who has not read the code.

## PR body

`Part of #N` or `Fixes #N` on the first line. Old behavior, then new behavior. `Changes:` as bullets. Under 600 words.
