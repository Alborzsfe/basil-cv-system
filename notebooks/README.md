# Notebooks

`01_dataset_analysis.ipynb` is a thin interactive wrapper for dataset inspection.
The reusable implementation lives in `src/`, and all reproducible operations have
command-line entry points in `scripts/`.

The notebook does not contain data, train a model, or report performance. Run it
only after following `docs/DATASET.md`, and keep private data outside version
control. Training, evaluation, inference, and experiment comparison are supported
by the Python scripts documented in the main README; notebook-only logic is
deliberately avoided.
