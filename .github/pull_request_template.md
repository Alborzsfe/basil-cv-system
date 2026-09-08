## Summary

Describe the change and why it is needed.

## Verification

- [ ] `ruff check .`
- [ ] `ruff format --check .`
- [ ] `pytest --cov=src --cov-fail-under=70`
- [ ] Synthetic smoke test (when pipeline behavior changes)

## Data and experiment integrity

- [ ] No private data, checkpoints, credentials, or generated run outputs are committed.
- [ ] Group-aware split guarantees remain intact.
- [ ] Any reported metric links to a held-out evaluation artifact.
