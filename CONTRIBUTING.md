# Contributing to MHA-GestureNet

Thanks for your interest in contributing to MHA-GestureNet. This document contains guidelines to make collaboration smooth and efficient.

## How to contribute
1. Fork the repository and create a branch for your change:
   - `git checkout -b feat/short-description` or `git checkout -b fix/short-description`
2. Make small, focused commits with descriptive messages.
3. Run the demo locally and ensure changes do not break existing behavior.
4. Open a pull request describing the change and why it's needed.

## Issues
- When filing an issue, include steps to reproduce, expected vs actual behaviour, and environment details (OS, Python version, streamlit version).
- Use labels: `bug`, `enhancement`, `documentation`, `question`.

## Pull request checklist
- [ ] PR targets `main` (or a feature branch if discussed)
- [ ] Descriptive title and summary
- [ ] Tests added/updated (if applicable)
- [ ] Documentation updated (README, DEPLOY_GUIDE.md, or docs)
- [ ] No secrets or large binaries committed

## Code style
- Follow Python conventions (PEP8). Use type hints where helpful.
- Keep functions small and single-purpose for testability.

## Model artifacts
- Do NOT commit large model weights into the repository. Use GitHub Releases or Git LFS and provide a download script.

## License
By contributing, you agree that your contributions will be licensed under the project's MIT license.
