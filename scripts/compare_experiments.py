#!/usr/bin/env python
"""Create an auditable experiment comparison table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_utils import ensure_dir

METRICS = ("accuracy", "f1_macro", "f1_weighted", "roc_auc_macro_ovr", "pr_auc_macro")


def _run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=path/to/test_metrics.json.")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Experiment name and path are required.")
    return name.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare classification experiments.")
    parser.add_argument("--run", action="append", type=_run, required=True)
    parser.add_argument("--output-dir", default="experiments/comparison")
    args = parser.parse_args()
    if len(args.run) < 2:
        parser.error("Provide at least two --run values.")
    rows = []
    for name, path in args.run:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"experiment": name, **{key: metrics.get(key) for key in METRICS}})
    destination = ensure_dir(args.output_dir)
    with (destination / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("experiment", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Classification experiment comparison",
        "",
        "| Experiment | " + " | ".join(METRICS) + " |",
        "| --- | " + " | ".join("---" for _ in METRICS) + " |",
    ]
    for row in rows:
        values = [row["experiment"]] + [
            "n/a" if row[key] is None else f"{float(row[key]):.4f}" for key in METRICS
        ]
        lines.append("| " + " | ".join(values) + " |")
    (destination / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
