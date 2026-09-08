from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from src.utils.io_utils import load_json, save_json
from src.utils.manifest import build_run_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_workflow_actions_are_commit_pinned() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    references = re.findall(r"uses:\s*[^@\s]+@([^\s#]+)", workflow)
    assert references
    assert all(re.fullmatch(r"[0-9a-f]{40}", reference) for reference in references)
    assert "permissions:\n  contents: read" in workflow


def test_readme_local_links_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"!?\[[^]]*\]\(([^)]+)\)", readme)
    local = [link for link in links if "://" not in link and not link.startswith("#")]
    assert local
    for link in local:
        assert (ROOT / link).exists(), link


def test_smoke_evidence_is_explicitly_non_scientific() -> None:
    script = (ROOT / "scripts/run_smoke_demo.py").read_text(encoding="utf-8")
    results = (ROOT / "docs/RESULTS.md").read_text(encoding="utf-8")
    assert '"scientific_claim": False' in script
    assert "no real greenhouse results" in results.lower()
    assert "synthetic_" in script


def test_every_torch_load_uses_weights_only() -> None:
    for path in [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if (
                node.func.attr == "load"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            ):
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                assert "weights_only" in keywords, path


def test_notebook_code_cells_compile() -> None:
    notebook = json.loads((ROOT / "notebooks/01_dataset_analysis.ipynb").read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"cell-{index}", "exec")


def test_manifest_and_json_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    manifest = build_run_manifest(seed=42, device="cpu")
    assert manifest["schema_version"] == 1
    assert manifest["source_commit"] == "abc123"
    assert manifest["seed"] == 42
    assert "python" in manifest and "packages" in manifest
    path = tmp_path / "manifest.json"
    save_json(manifest, path)
    assert load_json(path)["device"]["requested_or_resolved"] == "cpu"
