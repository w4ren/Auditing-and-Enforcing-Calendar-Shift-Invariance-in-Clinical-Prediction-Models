"""Run synthetic invariance checks, the known-nuisance negative control, figure generation, and verification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_figures import build_all_figures
from verify_repository import run_verification

from calendar_shift_audit.cli import run_synthetic_audit
from calendar_shift_audit.negative_control import run_known_nuisance_control


def main() -> None:
    repo_root = SCRIPT_DIR.parent
    parser = argparse.ArgumentParser(
        description="Reproduce synthetic experiments, figures, and numerical verification."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    args = parser.parse_args()

    tables_dir = args.repo_root / "results" / "tables"
    figures_dir = args.repo_root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    synth = run_synthetic_audit()
    print("Completed synthetic 7-shift invariance audit:")
    for arm, delta in synth["neural_max_shift_delta"].items():
        print(f"  {arm:18s}: max 7-shift log-score delta = {delta:.3e}")

    nc_res = run_known_nuisance_control(
        output_csv=tables_dir / "negative_control_summary.csv"
    )
    print(
        f"Recomputed known-nuisance negative control ({len(nc_res['summary'])} summary rows -> "
        f"{tables_dir / 'negative_control_summary.csv'})."
    )

    figs = build_all_figures(tables_dir, figures_dir)
    print(f"Rendered {len(figs)} publication figure files into {figures_dir}.")

    n_checks = run_verification(args.repo_root)
    print(f"All {n_checks} verification checks passed.")


if __name__ == "__main__":
    main()
