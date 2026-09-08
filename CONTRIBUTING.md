# Contributing

Thank you for improving Basil CV System. Open an issue before a large behavioral
change so that data assumptions and experiment comparability can be agreed.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ml,dev]"
```

Run the same checks as CI before opening a pull request:

```bash
ruff check .
ruff format --check .
python -m build
pytest --cov=src --cov-report=term-missing --cov-fail-under=70
python scripts/run_smoke_demo.py
```

## Data and scientific claims

- Never commit greenhouse images, labels, plant identifiers, or checkpoints.
- Preserve plant/session group separation in every data split.
- Record data provenance, label definitions, seed, split, configuration, and environment.
- Do not report a metric without its held-out evaluation artifact.
- Keep the inference-only label `unknown` out of the training class list.

Small, focused pull requests with tests and documentation are easiest to review.
