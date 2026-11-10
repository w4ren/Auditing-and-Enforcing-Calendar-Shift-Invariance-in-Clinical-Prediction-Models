"""Evaluate 7-offset continuous glucose monitoring forecast sensitivity from patient prediction arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calendar_shift_audit.bootstrap import percentile_ci
from calendar_shift_audit.glucose import (
    GLUCOSE_METRICS,
    glucose_bootstrap_schedules,
    patient_glucose_statistics,
)


def _interval_summary(point: float, draws: np.ndarray) -> dict[str, float]:
    lo, hi = percentile_ci(draws)
    return {
        "estimate": float(point),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
    }


def evaluate_prediction_archive(
    npz_path: Path,
    dataset: str = "colas",
    bootstrap_draws: int = 20000,
) -> dict[str, object]:
    data = np.load(npz_path, allow_pickle=False)
    predictions = np.asarray(data["predictions"], dtype=np.float64)
    if predictions.ndim == 4:
        predictions = predictions[:, None, :, :, :]
    targets = np.asarray(data["targets"], dtype=np.float64)
    patient_ids = np.asarray(
        data["patient_ids"] if "patient_ids" in data else data["patient_index"]
    )
    n_seeds, n_models, _, n_windows, horizon = predictions.shape
    if "model_names" in data:
        model_names = [str(x) for x in np.asarray(data["model_names"]).tolist()]
    else:
        model_names = [f"model_{idx}" for idx in range(n_models)]
    if len(model_names) != n_models:
        raise ValueError(f"Expected {n_models} model_names, got {len(model_names)}")

    stats = patient_glucose_statistics(predictions, targets, patient_ids)
    cube = stats["cube"]
    sse = stats["sse"]
    counts = stats["point_counts"]
    n_patients = int(cube.shape[0])

    pw, sw, bootstrap_seed = glucose_bootstrap_schedules(
        dataset=dataset,
        n_patients=n_patients,
        n_seeds=n_seeds,
        draws=bootstrap_draws,
    )

    disp_idx = GLUCOSE_METRICS.index("typical_nonzero_displacement")
    worst_idx = GLUCOSE_METRICS.index("worst_consistent_pair_displacement")
    cat_idx = GLUCOSE_METRICS.index("typical_nonzero_category_change")
    any_cat_idx = GLUCOSE_METRICS.index("patient_has_any_category_change")

    models_out: dict[str, dict[str, object]] = {}
    for m_idx, name in enumerate(model_names):
        disp_mat = cube[:, :, m_idx, disp_idx]
        worst_mat = cube[:, :, m_idx, worst_idx]
        cat_mat = cube[:, :, m_idx, cat_idx]
        pat_cat_mat = cube[:, :, m_idx, any_cat_idx]

        disp_draws = np.einsum("bp,ps,bs->b", pw, disp_mat, sw)
        worst_draws = np.einsum("bp,ps,bs->b", pw, worst_mat, sw)
        cat_draws = np.einsum("bp,ps,bs->b", pw, cat_mat, sw)
        pat_cat_draws = np.einsum("bp,ps,bs->b", pw, pat_cat_mat, sw)

        sse_0 = sse[:, :, m_idx, 0]
        shift0_point = float(np.mean(np.sqrt(sse_0.sum(axis=0) / float(counts.sum()))))
        boot_sse = pw @ sse_0
        boot_cnt = np.maximum((pw @ counts)[:, None], 1.0)
        rmse_draws = np.sum(sw * np.sqrt(boot_sse / boot_cnt), axis=1)

        models_out[name] = {
            "shift0_pooled_rmse_mgdl": _interval_summary(shift0_point, rmse_draws),
            "typical_nonzero_displacement_mgdl": _interval_summary(
                float(disp_mat.mean()), disp_draws
            ),
            "worst_pair_displacement_mgdl": _interval_summary(
                float(worst_mat.mean()), worst_draws
            ),
            "typical_nonzero_category_change": _interval_summary(
                float(cat_mat.mean()), cat_draws
            ),
            "patient_any_category_change": _interval_summary(
                float(pat_cat_mat.mean()), pat_cat_draws
            ),
        }

    return {
        "dataset": dataset.lower(),
        "patients": n_patients,
        "windows": int(n_windows),
        "horizon_steps": int(horizon),
        "seeds": int(n_seeds),
        "bootstrap_draws": int(bootstrap_draws),
        "bootstrap_seed": int(bootstrap_seed),
        "resampling_protocol": (
            "paired_whole_patient_fixed_seeds"
            if dataset.lower() == "cgmacros"
            else "paired_crossed_whole_patient_and_seed"
        ),
        "models": models_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit 7-offset glucose forecast displacement and glycemic category crossings."
    )
    parser.add_argument(
        "--predictions-npz",
        type=Path,
        required=True,
        help="NPZ file containing predictions (S, [M,] 7, W, H), targets (W, H), and patient_ids (W,)",
    )
    parser.add_argument(
        "--dataset",
        choices=["colas", "cgmacros"],
        default="colas",
        help="Dataset bootstrap convention (colas: crossed patient-seed; cgmacros: patient-only)",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    report = evaluate_prediction_archive(
        args.predictions_npz,
        dataset=args.dataset,
        bootstrap_draws=args.bootstrap_draws,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote glucose shift audit summary to {args.output_json}")


if __name__ == "__main__":
    main()
