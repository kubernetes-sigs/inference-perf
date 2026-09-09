# Release Process

Releases are tag-driven. Pushing a `vX.Y.Z` tag runs
[`publish-on-release.yml`](.github/workflows/publish-on-release.yml), which creates the GitHub
Release, publishes to PyPI, pushes the image to quay.io, and packages the Helm chart. This
document is what a maintainer does around that automation.

## 1. Before the cut

- Close the milestone. Every open item is merged, moved to the next milestone, or closed
      with a reason.
- Merge a PR bumping `version` in `pyproject.toml` and `version` and `appVersion` in
      `deploy/inference-perf/Chart.yaml` to `X.Y.Z`.
- Label every PR that belongs in the changelog with one of the categories in
      [`.github/changelog-config.json`](.github/changelog-config.json). Unlabelled PRs are
      dropped from the generated changelog.
- Confirm the commit you will tag is green on `main`: linting and type checks, unit tests,
      coverage, and `E2E Test on change`. Tag only a merged commit on `main`.
- Draft the summary of features, fixes, and improvements that goes above the generated
      changelog.
- Optional: run [`test-release.yml`](.github/workflows/test-release.yml) by
      `workflow_dispatch` to build the package against TestPyPI.

## 2. Cut

A maintainer with write access pushes the tag. There is no release PR.

```sh
git fetch upstream
git tag vX.Y.Z upstream/main
git push upstream vX.Y.Z
git push upstream upstream/main:refs/heads/release-vX.Y.Z   # one branch per release, at the tag
```

Drafting and publishing a release in the GitHub UI creates the same tag and triggers the same
workflow.

## 3. What the automation does

- `build-and-publish` builds the changelog from labelled PRs since the previous tag and creates
  the GitHub Release.
- `python-package` sets `version` from the tag, builds the package, and uploads it to PyPI.
- `docker` builds `linux/amd64` and tags it `vX.Y.Z` and `latest`.
- `helm-chart` packages `deploy/inference-perf` and pushes it to
  `oci://quay.io/inference-perf/charts/inference-perf`.

Watch the run under Actions, "Release Processing". The release notes advertise every artifact, so
fix and re-run any failed job before announcing.

## 4. After the cut

- Put the written summary above the generated changelog in the release body.
- Verify each artifact:
      `pip install inference-perf==X.Y.Z`,
      `docker pull quay.io/inference-perf/inference-perf:vX.Y.Z`,
      `helm show chart oci://quay.io/inference-perf/charts/inference-perf --version X.Y.Z`.
- Announce in [#inference-perf](https://kubernetes.slack.com/?redir=%2Fmessages%2Finference-perf)
      on Kubernetes Slack and link the release page.
- Open the next milestone.

## Cadence

Minor releases are milestone-driven. The cut happens when the milestone's release-blocking items
are done, and everything else moves to the next milestone with one line of why. Release-blocking
means the item is marked blocking in the milestone's tracking issue, or it is a correctness
regression in a shipped code path. Patch releases ship fixes only, cut from the tip of `main`, or
cherry-picked onto `release-vX.Y.Z` when `main` carries unreleased features.

## Versioning

Semantic versioning, `vX.Y.Z`, with the `v` prefix on tags and image tags and no prefix on PyPI
and chart versions.
