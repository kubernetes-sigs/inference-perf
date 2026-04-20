# Contributing Guidelines

Welcome to Kubernetes. We are excited about the prospect of you joining our [community](https://git.k8s.io/community)! The Kubernetes community abides by the CNCF [code of conduct](code-of-conduct.md). Here is an excerpt:

_As contributors and maintainers of this project, and in the interest of fostering an open and welcoming community, we pledge to respect all people who contribute through reporting issues, posting feature requests, updating documentation, submitting pull requests or patches, and other activities._

## Getting Started

We have full documentation on how to get started contributing here:

To contribute to this project, please adhere to the following procedure:

- **Fork the repository** and submit a Pull Request (PR) in accordance with the Kubernetes contribution guidelines below.
- For initial contributions, it is recommended to select an issue designated with the **`good-first-issue`** label.
- **Create an issue** detailing the proposed changes. Upon confirmation of assignment, you may proceed to submit a Pull Request.
- **Initial Setup**: Run `pdm run setup` to install dependencies and set up pre-commit hooks.
- Implement the required code modifications and perform **manual verification** by executing a benchmark run.
- **Include comprehensive unit tests** to validate the changes. You can use `pdm test:picked` to run tests only on modified files.
- Execute **validation** before pushing:
  - `pdm check`: Fast check (formatting and linting).
  - `pdm run validate`: Full check (formatting, linting, and strict type checking).
- Conduct **full test execution and code coverage analysis** using `pdm test` and `pdm check:cov`.
- **Branch Up-to-Date Requirement**: Your branch must be based on the latest `origin/main` before you can push. A pre-push hook enforces this by running `git fetch origin main` automatically. If it fails, you will need to rebase or merge `origin/main`.
- **Submit the Pull Request** utilizing the provided template.

### Resources
- [Contributor License Agreement](https://git.k8s.io/community/CLA.md) - Kubernetes projects require that you sign a Contributor License Agreement (CLA) before we can accept your pull requests
- [Kubernetes Contributor Guide](https://k8s.dev/guide) - Main contributor documentation, or you can just jump directly to the [contributing page](https://k8s.dev/docs/guide/contributing/)
- [Contributor Cheat Sheet](https://k8s.dev/cheatsheet) - Common resources for existing developers

## Mentorship

- [Mentoring Initiatives](https://k8s.dev/community/mentoring) - We have a diverse set of mentorship programs available that are always looking for volunteers!

## Contact Information

- [Slack](https://kubernetes.slack.com/?redir=%2Fmessages%2Finference-perf)
- [Mailing List](https://groups.google.com/forum/#!forum/kubernetes-sig-scale)
