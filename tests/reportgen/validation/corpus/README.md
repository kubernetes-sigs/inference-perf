# Report-validation golden corpus

Each directory here that contains a `validation.json` is one test case: the
report files of a single inference-perf run, frozen at capture time, plus the
findings the validators are expected to produce for them. `test_corpus.py`
discovers every case, runs the full default validator stack over its report
files, and asserts the output matches `validation.json` exactly.

## Adding a case

A run already emits `validation.json` next to its other report files, so
capturing a case is copying the run's output directory:

1. Copy the run's output directory into `corpus/`, under a grouping directory
   if one fits (e.g. `per-stage-reports/` for cases that exercise per-stage
   properties, `full-runs/` for complete report sets).
2. Name the case directory `v{major}_{minor}_{patch}-{tag}`, where the version
   is the inference-perf version that produced the run and the tag says what
   the case demonstrates, e.g. `v0_5_0-negative-stage-rate`. The name is not
   parsed — it is documentation for humans, and it becomes the pytest ID.
3. Review the `validation.json` you are freezing: it is the golden, and
   committing it asserts "these findings are correct for these reports".
4. Run `pdm run test` to confirm the case passes.

Notes:

- Only `.json`/`.yaml`/`.yml` files are loaded as reports; anything else
  (e.g. a `notes.md`) is ignored, so a case can carry commentary.
- Partial sets are fine: a case with only stage files exercises the per-stage
  validator, and the validators for absent report families stay silent.
- Known-bad reports are the most valuable cases: when validation catches (or
  misses) something in a real run, freeze that run with its findings so the
  behavior can never regress silently.
- Prefer small runs. `per_request_lifecycle_metrics.json` grows with request
  count — trim or drop it when the case does not exercise it.
- For a hand-built case without a run-emitted golden, seed it with
  `echo '{}' > validation.json`, then run `pdm run update:goldens` and review
  the generated golden before committing.

## When validator behavior changes

Goldens assert exact output, so adding a check or rewording a message can
stale many cases at once. Regenerate them with:

    pdm run update:goldens

then review the git diff. For each changed case the question is binary: if
the new findings are correct for that frozen input, commit the updated
golden; if the case's report format is no longer supported, delete its
directory. Cases whose version prefix is far behind current are candidates
for removal rather than regeneration.

After the first regeneration, a case's `validation.json` means "what current
code is expected to say about this frozen input" — the run's own output is
only the seed golden, reviewed once at drop-in time.
