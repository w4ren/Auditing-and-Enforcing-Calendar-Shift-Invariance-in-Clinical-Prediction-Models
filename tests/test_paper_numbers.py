"""End-to-end verification of public result tables and figures against the manuscript."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from verify_repository import run_verification


def test_all_manuscript_tables_and_figures_match() -> None:
    total_checks = run_verification(REPO_ROOT)
    assert total_checks >= 120
