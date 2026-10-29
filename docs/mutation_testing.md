# Mutation testing: what the ratchet checks and how to pass it

**Coverage says a line ran. It does not say any test would fail if the line were wrong.** The `mutation` job in the Code Coverage workflow checks the second thing for the code a PR touches. It mutates each function in the PR's scope, once on the merge base and once on the PR, runs the tests that cover it, and fails when the PR leaves more mutants alive than the base did. It is merge-blocking, it runs only on PRs, and it is skipped when nothing under `inference_perf/` or `tests/` changes and neither `pdm.lock` nor `pyproject.toml` does.

Run it locally the way CI does, from a branch with `origin/main` fetched:

```
pdm sync -d -G analysis -G otel
pdm run check:mutation
```

`--base <ref>` changes the branch it compares against, `--keep` leaves the mutation trees in place for a look, `--budget-minutes` stops it with a partial report, and `mutation.json` holds every mutant's status on both sides.

## What is in scope

**Scope is the functions the diff touches plus the functions the tests it could have weakened run.** A function is touched when any line in its span changes, decorators included. A deleted function is in scope with its base-side survivors, so deleting tested code does not pay for anything. A test could have been weakened when it existed on the base and the PR edits, deletes or renames it, or edits a helper, fixture (autouse and `usefixtures` included), class, constant or import it reads in the same file, or a helper, `conftest.py` or package `__init__.py` it imports, directly or through another helper. Every function such a test runs on either side joins the scope. That second half is where a loosened assert shows: the source did not change, its survivors did. A test the PR adds can only add kills, so it widens nothing; comments and blank lines widen nothing; any other changed line outside a test, such as module-level code, widens to every test in the file.

**A function in scope only through changed tests runs against the changed tests first.** Its code is the same on both sides, and so is every test that did not change, so those tests catch a given mutant on both sides or on neither. Only the changed tests can make the sides differ. Each such mutant runs against just the changed tests on each side. When they catch it on both sides, or miss it on both, it is settled and counts on neither side. When they catch it on one side and miss it on the other, the rest of the suite runs it on the side that missed, and that result counts. A changed source function elsewhere that hides the mutant from an unchanged test is not seen. It is not seen when no test changes either.

Code that only runs in a multiprocessing child counts. The job loads a pytest plugin that carries a `Worker`'s trampoline hits and coverage back to the test that started it.

## The rules, and what each failure means

The job summary prints one table row per function in scope with its survivor count on the base and on the PR, then a `FAIL:` line naming every rule that tripped. For a function in scope only through changed tests the counts cover only the mutants the changed tests did not settle. Up to 10 failures also show as annotations on the line they are about, in the PR's Files changed tab: each new survivor with the change no test caught, each new function no test runs, and each new statement mutmut never mutates. In order of how often they come up:

1. **`Surviving mutants rose from N to M`.** One total over the whole scope, not per function, so moving code between functions compares like for like. Each survivor in a function whose count rose is rerun 3 times unless the identical mutation also survived on the base. A survivor every rerun kills is a fluke and does not count. One that only some reruns kill is flaky, is listed with the tests that killed it, and counts. Read the survivor diffs under the table: each is a change to your code that no test noticed. Write the test that notices it. A survivor somewhere else in scope pays for one that is an equivalent mutant, and so does a deleted function's survivor.
2. **`No test runs <function>`.** A new function no test executes. A survivor can be an equivalent mutant, code nobody runs cannot be, so nothing pays for it. Write a test that calls it.
3. **`Statements mutmut never mutates rose from N to M`.** mutmut mutates the bodies of module-level functions and of methods of module-level classes, and nothing else. Logic in a decorated function (pydantic validators, properties, `functools.cache`), at module or class level, in a lambda, in a nested function or class, inside a `len()` or `isinstance()` call, in a string passed to `exec` or `eval`, on a line matching `do_not_mutate_patterns` (log calls, `raise X(`), or in a file `do_not_mutate` excludes is never checked. The count of such statements across the changed files cannot rise. The summary lists each one with its reason. Move the logic into an undecorated module-level function or method and call it from where it was. In a file `do_not_mutate` excludes every statement counts, so the logic has to move to a file it does not exclude. A decorated function whose body is one plain delegation counts as zero.
4. **`<function> had N mutants on the base and none with this change`.** The function used to be mutated and now is not, usually because it gained a decorator. Same fix as rule 3.
5. **`Tests fail with an active mutant id that changes no code`.** A canary reruns the covering tests with a random mutant id that matches nothing, so every function runs unmutated. A test that fails there is reacting to the harness, not the code.
6. **`Code under inference_perf/ or tests/ names the mutation harness`.** Any line naming mutmut or a mutant fails outright, since code that can tell it runs under mutation testing can kill every mutant without checking anything. The script's own tests are the one exemption.
7. **`Code under inference_perf/ imports repo code outside it`** and **`New root modules shadow an installed or standard library module`.** Only `inference_perf/` is mutated. Logic moved out of it and imported back is never checked, and a new root module under a taken name wins over the installed one from the repo root.
8. **`mutmut is not the pinned release`.** The locked and installed mutmut must be the release this script pins, checked against the wheel hashes in `pdm.lock`. Operator changes between releases move survivor counts on main and PRs alike.
9. **`Ran out of the N minute budget while <stage>`.** The run stopped inside the job's own timeout and reported what it reached. The not-reached list names the functions whose verdict is missing. A smaller scope is the fix: split the PR, or keep edits to tests that run most of the package in their own PR.

Two errors stop the run before a verdict. **`The base branch ... fails its own unit tests in the mutation tree`** means main is red and names the failing tests. Every PR fails until main is fixed. There is no exit-0 path for this on purpose, since a red main would otherwise be a way through. **`Unit tests fail in the mutation tree <head>`** means the PR's own suite fails inside the mutation tree, which differs from a checkout in two ways: it is a copy, not a repository, so tests that shell out to `git` are excluded, and it holds only the paths `also_copy` lists in `[tool.mutmut]`.

## What is deliberately not here

**There is no opt-out in code.** `# pragma: no mutate` is stripped before mutating. `[tool.mutmut]` is read from the base branch, so a PR cannot change what gets mutated for its own run. CI runs the PR's own copy of the script and the workflow, so the code cannot stop a PR from editing its check. The script, the workflow and the config are owned separately (#876), so such an edit lands only after an owner reads it. Each of these closes a hole an automated contributor would find.

**Equivalent mutants are paid for, not allowlisted.** A mutant no test can kill because it does not change behaviour is real and does happen. The ratchet does not try to recognise one. Deleting dead code, killing a survivor in a neighbouring function in scope, or restructuring so the equivalent mutant is not generated are the ways out. An allowlist would be the opt-out the previous paragraph removes.

**Changes outside function bodies pass as nothing to mutate.** Constants, pydantic field defaults and class attributes are declarations. Module or class level code that does more than declare counts under rule 3.

## Wall time

**The floor is two serial full-suite runs.** mutmut's stats pass runs the whole unit suite once per side to learn which tests cover which function, so one changed function costs about 8 minutes on a 4-core runner before a mutant runs. Mutant runs are parallel and cost their covering tests' time each. A function in scope only through changed tests costs the changed tests' time per mutant, plus its covering tests' time for each mutant the changed tests did not settle. The job stops itself at 80 minutes and reports what it did not reach. Scope is what sets the wall time: the day the job landed, PRs with 54 to 66 functions in scope passed in 4 to 51 minutes and every PR above about 100 ran out of budget (#1002). An edit to one test in a file whose other tests drive the whole load generator used to bring all of it in; now only the edited test's functions come in.

## The weekly run

`pdm run test:mutation` mutates the whole package against the full suite and prints survivors per module. It takes hours and is the input for a scheduled run and a burn-down, neither of which exists yet (#872).

## When the call flips

The ratchet is blocking and has no opt-out because a report that does not block is a report that gets ignored, which is why #872 dropped its report-only phase. It should become report-only if the rate of failures on honest PRs that are not untested logic stays above the rate at which it catches untested logic. Every failure in the replays of merged PRs before it was turned on was untested logic (#722, #779, #820). If a month of PRs says otherwise, flip it.
