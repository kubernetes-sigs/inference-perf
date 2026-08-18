# Release Process

Releases are tag-driven. Pushing a `vX.Y.Z` tag runs
[`publish-on-release.yml`](.github/workflows/publish-on-release.yml), which creates the GitHub
Release, publishes the Python package to PyPI, pushes the container image to quay.io, and packages
the Helm chart. Everything below is what a maintainer does around that automation.

## Runbook

### 1. Before the cut

- [ ] The release milestone is groomed: every open item is either merged, moved to the next
      milestone, or closed with a reason. Close the milestone.
- [ ] `main` is green: `Python Linting and Type Checks`, `Unit Tests`, `Code Coverage`, and
      `E2E Test on change` all pass on the commit you are about to tag. The e2e job is the
      release-blocking suite; do not tag a commit it has not passed on.
- [ ] The commit you tag is a tide-merged commit on `main` (`git merge-base --is-ancestor
      <commit> upstream/main`). Every such commit already passed the deterministic suites at
      merge time via the merge gate, which is why the publish workflow runs no tests of its
      own (#643); tagging anything else forfeits that guarantee.
- [ ] Every PR that should appear in the changelog carries one of the labels in
      [`.github/changelog-config.json`](.github/changelog-config.json) (`feature`, `enhancement`,
      `bug`, `fix`, `documentation`, `docs`, `performance`, `perf`, `dependencies`, `deps`).
      Unlabelled PRs are dropped from the generated changelog (v0.6.1 shipped with 22 merged PRs
      and 0 categorized).
- [ ] Draft the written summary (see step 4). Past releases (v0.5.0, v0.6.0) put a short
      "Summary" of features, fixes, and improvements above the generated list.
- [ ] Optional: run [`test-release.yml`](.github/workflows/test-release.yml) (`workflow_dispatch`)
      to dry-run the package build against TestPyPI.

### 2. Cut

Tags are pushed directly by a maintainer with write access; there is no release PR.

```sh
git fetch upstream
git tag vX.Y.Z upstream/main            # the tested commit
git push upstream vX.Y.Z
git push upstream upstream/main:refs/heads/release-vX.Y.Z   # convention: one branch per release, at the tag
```

Creating the release through the GitHub UI (draft, then publish) also creates the tag and
triggers the same workflow; v0.6.0 and v0.6.1 were cut that way.

The version in `pyproject.toml` and `deploy/inference-perf/Chart.yaml` is not the source of truth:
the workflow rewrites `pyproject.toml` from the tag at publish time, and the checked-in values lag
(both read `0.5.0` at v0.6.1). Bumping them in a follow-up PR is tidy but not required to cut.

### 3. What the automation does

| Job | Action | Output |
|---|---|---|
| `build-and-publish` | changelog from labelled PRs since the previous tag; creates the GitHub Release | `https://github.com/kubernetes-sigs/inference-perf/releases/tag/vX.Y.Z` |
| `python-package` | rewrites `version` in `pyproject.toml` from the tag, `python -m build`, `twine upload` | `pip install inference-perf==X.Y.Z` |
| `docker` | `linux/amd64` image, tags `vX.Y.Z` and `latest` | `quay.io/inference-perf/inference-perf:vX.Y.Z` |
| `helm-chart` | packages `deploy/inference-perf` and pushes to `oci://quay.io/inference-perf/charts/inference-perf` | see "Known gaps": this job has failed on every release so far |

Watch the run under Actions, "Release Processing". A red `helm-chart` job does not block the
package or image; a red `python-package` or `docker` job means the release notes advertise
artifacts that do not exist, so fix and re-run before announcing.

### 4. After the cut

- [ ] Edit the release body: if the generated changelog is empty or thin, use "Generate release
      notes" in the release editor and put the written summary above it.
- [ ] Verify each artifact:
      `pip install inference-perf==X.Y.Z && python -c 'import importlib.metadata as m; print(m.version("inference-perf"))'`;
      `docker pull quay.io/inference-perf/inference-perf:vX.Y.Z`;
      `helm show chart oci://quay.io/inference-perf/charts/inference-perf --version X.Y.Z`.
      Remove any line from the release notes whose artifact did not publish.
- [ ] Announce in [#inference-perf](https://kubernetes.slack.com/?redir=%2Fmessages%2Finference-perf) on Kubernetes Slack; link the release page.
- [ ] Open the next milestone if it does not exist yet.

## Cadence and scope

- Minor releases are milestone-driven, not calendar-driven: a milestone opens with a theme and a
  target date, and the cut happens when its release-blocking items are done, with the rest
  explicitly moved out.
- Release-blocking means: the item is on the milestone and marked blocking in the milestone's
  tracking issue, or it is a correctness regression in a shipped code path. Everything else moves
  to the next milestone at cut time with one line of why.
- Patch releases (`vX.Y.Z+1`) ship fixes only, cut from the tested tip of `main` unless `main`
  carries unreleased features, in which case cherry-pick onto `release-vX.Y.Z` and tag there.
- The e2e suite (`E2E Test on change`) is the release gate; a nightly live-server run, when it
  exists, is release-blocking for the rows it covers.

## Versioning

Semantic versioning, `vX.Y.Z`, `v` prefix on tags and image tags, no prefix on PyPI or chart
versions.

- **Minor (`Y`)**: new features, new config fields, new report fields, new metrics, dependency
  bumps that change behaviour. Anything a user must read release notes to adopt.
- **Patch (`Z`)**: bug fixes, doc fixes, CI changes, dependency bumps with no behaviour change.
- **Major (`X`)**: reserved for v1.0.0 (#321); pre-1.0, minor releases may change config and
  report formats, and the release notes must call every such change out.

## Known gaps in the automation

Tracked so a release owner is not surprised; fixes are separate PRs.

1. `helm-chart` in `publish-on-release.yml` fails at `helm package`: `--app-version` reads
   `env.RELEASE_VERSION`, which is set in a different job, so the flag is empty. Failed on
   v0.5.0, v0.6.0, v0.6.1.
2. `helm push` in `publish-on-change.yml` fails with `401` against
   `quay.io/inference-perf/charts/inference-perf`; the robot account cannot write that repository.
   No chart has been published, so the "Helm Chart" section in every release body is not true.
3. The generated changelog only lists labelled PRs (item 1 above).
4. `pip install inference-perf==vX.Y.Z` in the release template keeps the `v`; pip accepts it, but
   the canonical form is `X.Y.Z`.
