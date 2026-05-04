# Contributing

Thank you for your interest in this project.

This repository accompanies a scientific publication and is provided primarily
as a reference implementation for reproducibility. The codebase is considered
**finalized for the associated publication** and is not under active feature
development.

## Issues

Bug reports and questions are welcome. Please open a GitHub issue and include:

- A clear description of the problem or question.
- Steps to reproduce (commands, input data shape, expected vs. actual output).
- Environment details (OS, Python version, key package versions, GPU/CPU).
- Relevant logs or stack traces.

The maintainer will triage issues on a best-effort basis.

## Pull requests

External pull requests are accepted on a limited basis, primarily for:

- Bug fixes that preserve the published behavior.
- Documentation improvements.
- Reproducibility or environment-setup fixes.

Substantive changes (new features, refactors, API changes) are unlikely to be
merged because the code is tied to a published study. If you would like to
extend the work, we encourage forking the repository.

### Pull request checklist

Before opening a PR, please ensure:

- The change is described clearly in the PR body, including motivation.
- Existing scripts still run on the example data in `Data/` and
  `Structure_data/`.
- No credentials, API keys, internal identifiers, or large/proprietary data
  files have been added.
- Commits are signed off (`git commit -s`) where possible.

## Security

If you believe you have found a security vulnerability, please do **not** open
a public issue. Contact the maintainer listed in
[AUTHORS.md](AUTHORS.md) directly.

## Code of conduct

Be respectful and constructive in all interactions. Reports of unacceptable
behavior may be sent to the maintainer.
