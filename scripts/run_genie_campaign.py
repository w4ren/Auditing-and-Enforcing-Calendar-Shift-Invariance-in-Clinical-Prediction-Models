"""Train and audit neural and boosted discrete-time survival models on local GENIE BPC v2.0 tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calendar_shift_audit.cohort import (
    DISEASE_SITES,
    build_frame,
    load_genie_release_tables,
    outer_split,
    shift_frame,
)
from calendar_shift_audit.models import (
    ALL_NEURAL_ARMS,
    TrainConfig,
    build_matched_model,
    build_sequences,
    score_patients,
    to_tensors,
    train_model,
)
from calendar_shift_audit.xalign import (
    build_rows,
    common8_design,
    estimate_phase_kappa,
    fit_xgb_discrete_hazard,
    phase_posterior,
    score_xgb_rows,
)


def run_neural_fold(
    frame: pd.DataFrame,
    held_out_site: str,
    seed: int,
    device: str,
    epochs: int = 400,
) -> list[dict[str, object]]:
    dev_df, ev_df = outer_split(frame, held_out_site)
    X_dev, X_ev, names = common8_design(dev_df, ev_df)
    dev_bundle = build_sequences(dev_df, X_dev, names)
    if dev_bundle.n_features != 19:
        raise AssertionError(f"Expected 19 neural input features, got {dev_bundle.n_features}")
    dev_tensors = to_tensors(dev_bundle, device=device)

    shifted_tensors = []
    for delta in range(7):
        s_df = shift_frame(ev_df, delta)
        _, X_s, _ = common8_design(dev_df, s_df)
        shifted_tensors.append(to_tensors(build_sequences(s_df, X_s, names), device=device))

    results = []
    for arm in ALL_NEURAL_ARMS:
        model = build_matched_model(arm, n_features=dev_bundle.n_features, seed=seed).to(device)
        info = train_model(model, dev_tensors, TrainConfig(epochs=epochs, seed=seed))
        shift_scores = [score_patients(model, st) for st in shifted_tensors]
        mat = np.stack(shift_scores, axis=1)
        results.append(
            {
                "family": "neural",
                "arm": arm,
                "site": held_out_site,
                "seed": seed,
                "n_features": dev_bundle.n_features,
                "parameters": model.n_parameters(),
                "epochs_run": info["epochs_run"],
                "shift0_patient_log_likelihood": float(np.mean(mat[:, 0])),
                "patient_shift_range": float(np.mean(np.max(mat, axis=1) - np.min(mat, axis=1))),
            }
        )
    return results


def run_xalign_fold(
    frame: pd.DataFrame,
    held_out_site: str,
    seed: int,
) -> list[dict[str, object]]:
    dev_df, ev_df = outer_split(frame, held_out_site)
    X_dev, X_ev, _ = common8_design(dev_df, ev_df)
    dev_rows = build_rows(dev_df)
    ev_rows = build_rows(ev_df)
    kappa = estimate_phase_kappa(dev_rows, X_dev)

    arm_specs: list[tuple[str, int | None]] = [("X-base", None), ("X-align", 0)] + [
        (f"X-align-rot{r}", r) for r in range(1, 7)
    ]

    shifted_eval = []
    for delta in range(7):
        s_df = shift_frame(ev_df, delta)
        _, X_s, _ = common8_design(dev_df, s_df)
        shifted_eval.append((build_rows(s_df), X_s))

    results = []
    for arm, rot in arm_specs:
        post_dev = (
            None
            if rot is None
            else phase_posterior(dev_rows, kappa, weight_mode="sqrt", rotate_fixed=rot)
        )
        booster, tail_rate = fit_xgb_discrete_hazard(dev_rows, X_dev, post_dev, seed=seed)

        shift_row_scores = []
        for s_rows, X_s in shifted_eval:
            post_ev = (
                None
                if rot is None
                else phase_posterior(s_rows, kappa, weight_mode="sqrt", rotate_fixed=rot)
            )
            shift_row_scores.append(score_xgb_rows(booster, tail_rate, s_rows, X_s, post_ev))

        score_mat = np.stack(shift_row_scores, axis=1)
        log_s0 = score_mat[:, 0]
        comp = ev_rows.event == 1

        pat_means = []
        for p in range(ev_rows.patients):
            mask = (ev_rows.patient_index == p) & comp
            if np.any(mask):
                pat_means.append(float(np.mean(log_s0[mask])))

        results.append(
            {
                "family": "boosted",
                "arm": arm,
                "site": held_out_site,
                "seed": seed,
                "fitted_kappa": kappa,
                "shift0_completed_gap_log_score": float(np.mean(pat_means)),
                "all_interval_max_spread": float(
                    np.max(np.max(score_mat, axis=1) - np.min(score_mat, axis=1))
                ),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leave-one-institution-out GENIE BPC survival training and shift audits."
    )
    parser.add_argument("--release-dir", type=Path, required=True, help="Upstream GENIE BPC release folder")
    parser.add_argument("--disease", choices=["nsclc", "crc"], default="nsclc")
    parser.add_argument("--site", type=str, default="ALL", help="Held-out institution or ALL")
    parser.add_argument("--family", choices=["neural", "xalign", "both"], default="both")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    patients, marks = load_genie_release_tables(args.release_dir)
    frame, audit, _ = build_frame(patients, marks)
    sites = list(DISEASE_SITES[args.disease]) if args.site == "ALL" else [args.site]

    rows: list[dict[str, object]] = []
    for site in sites:
        if args.family in ("neural", "both"):
            rows.extend(
                run_neural_fold(frame, site, seed=args.seed, device=args.device, epochs=args.epochs)
            )
        if args.family in ("xalign", "both"):
            rows.extend(run_xalign_fold(frame, site, seed=args.seed))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "disease": args.disease,
        "patients": int(frame["record_id"].nunique()),
        "intervals": int(len(frame)),
        "audit_rows": int(len(audit)),
        "results": rows,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} fold evaluation records to {args.output_json}")


if __name__ == "__main__":
    main()
