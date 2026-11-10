"""Command-line interface for calendar-shift invariance auditing and verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from .cohort import (
    build_frame,
    exclusion_summary,
    load_genie_release_tables,
    shift_frame,
    simple_numeric_categorical_design,
    summarize_cohort,
)
from .config import load_protocol_config
from .models import (
    ALL_NEURAL_ARMS,
    CALENDAR_BLIND,
    ABSOLUTE_PHASE,
    RELATIVE_PHASE,
    FLEXIBLE_ELAPSED,
    build_matched_model,
    build_sequences,
    component_parameter_counts,
    paired_initialization_summary,
    score_patients,
    to_tensors,
    verify_all_parameters_active,
)
from .negative_control import run_known_nuisance_control
from .xalign import (
    build_rows,
    common8_design,
    estimate_phase_kappa,
    fit_xgb_discrete_hazard,
    phase_posterior,
    score_xgb_rows,
)


def _synthetic_recurrent_frame(seed: int = 20260910) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sites = ("DFCI", "MSK", "UHN", "VICC")
    rows = []
    for p_idx in range(32):
        pid = f"SYN_{p_idx:03d}"
        site = sites[p_idx % len(sites)]
        anchor = int(rng.integers(0, 7))
        cur = float(anchor + 7 * int(rng.integers(10, 40)))
        n_gaps = int(rng.integers(4, 9))
        for j in range(n_gaps):
            weeks = int(rng.integers(1, 7))
            jitter = int(rng.choice([-1, 0, 0, 0, 1]))
            gap = max(1, 7 * weeks + ((anchor - int(cur) % 7 + jitter) % 7))
            nxt = cur + float(gap)
            rows.append(
                {
                    "record_id": pid,
                    "interval_start": cur,
                    "interval_stop": nxt,
                    "exposure_days": float(gap),
                    "event": 1,
                    "order_in_patient": j,
                    "start_is_imaging": int(rng.integers(0, 2)),
                    "institution": site,
                    "cov_a": float(rng.normal()),
                    "cov_b": float(rng.normal()),
                    "cov_c": float(rng.normal()),
                    "cov_d": float(rng.normal()),
                    "cov_e": float(rng.normal()),
                    "cov_f": float(rng.normal()),
                    "cov_g": float(rng.normal()),
                }
            )
            cur = nxt
        if p_idx % 3 == 0:
            censor_gap = float(rng.integers(5, 35))
            rows.append(
                {
                    "record_id": pid,
                    "interval_start": cur,
                    "interval_stop": cur + censor_gap,
                    "exposure_days": censor_gap,
                    "event": 0,
                    "order_in_patient": n_gaps,
                    "start_is_imaging": 0,
                    "institution": site,
                    "cov_a": 0.0,
                    "cov_b": 0.0,
                    "cov_c": 0.0,
                    "cov_d": 0.0,
                    "cov_e": 0.0,
                    "cov_f": 0.0,
                    "cov_g": 0.0,
                }
            )
    return pd.DataFrame(rows)


def run_synthetic_audit() -> Dict[str, Any]:
    """Run parameter-matching and 7-shift invariance checks on synthetic recurrent data."""
    cfg = load_protocol_config()
    frame = _synthetic_recurrent_frame()
    dev = frame.loc[frame["institution"] != "VICC"].reset_index(drop=True)
    ev = frame.loc[frame["institution"] == "VICC"].reset_index(drop=True)

    cov_cols = [
        "start_is_imaging",
        "cov_a",
        "cov_b",
        "cov_c",
        "cov_d",
        "cov_e",
        "cov_f",
        "cov_g",
    ]
    _, X_ev, names = simple_numeric_categorical_design(dev, ev, cov_cols, [])
    bundle_0 = build_sequences(ev, X_ev, names)
    assert bundle_0.n_features == cfg["neural_audit"]["n_features"] == 19

    models = {arm: build_matched_model(arm, n_features=19, seed=3) for arm in ALL_NEURAL_ARMS}
    with __import__("torch").no_grad():
        models[ABSOLUTE_PHASE].abs_class.copy_(
            __import__("torch").linspace(-0.25, 0.25, 49).reshape(7, 7)
        )

    param_counts = {arm: m.n_parameters() for arm, m in models.items()}
    components = {arm: component_parameter_counts(m) for arm, m in models.items()}
    active_checks = {arm: verify_all_parameters_active(m)["all_active"] for arm, m in models.items()}
    paired_init = paired_initialization_summary(
        {k: models[k] for k in (CALENDAR_BLIND, ABSOLUTE_PHASE, RELATIVE_PHASE)}
    )

    neural_shift_max_delta: Dict[str, float] = {}
    base_scores = {arm: score_patients(m, to_tensors(bundle_0)) for arm, m in models.items()}
    for arm, model in models.items():
        max_delta = 0.0
        for shift in range(1, 7):
            ev_s = shift_frame(ev, shift)
            _, X_ev_s, _ = simple_numeric_categorical_design(dev, ev_s, cov_cols, [])
            b_s = build_sequences(ev_s, X_ev_s, names)
            sc = score_patients(model, to_tensors(b_s))
            max_delta = max(max_delta, float(np.max(np.abs(sc - base_scores[arm]))))
        neural_shift_max_delta[arm] = max_delta

    X_dev_8, X_ev_8, _ = common8_design(dev, ev)
    dev_rows = build_rows(dev)
    ev_rows = build_rows(ev)
    kappa = estimate_phase_kappa(dev_rows, X_dev_8, iterations=40)
    p_dev = phase_posterior(dev_rows, kappa, weight_mode="sqrt")
    p_ev = phase_posterior(ev_rows, kappa, weight_mode="sqrt")
    booster_align, tail_align = fit_xgb_discrete_hazard(
        dev_rows,
        X_dev_8,
        p_dev,
        seed=11,
        xgb_params={"eta": 0.05, "max_depth": 3, "min_child_weight": 5.0, "num_boost_round": 40},
        days=60,
    )
    score_0 = score_xgb_rows(
        booster_align, tail_align, ev_rows, X_ev_8, p_ev, days=60
    )
    xalign_max_delta = 0.0
    for shift in range(1, 7):
        ev_s = shift_frame(ev, shift)
        _, X_ev_s8, _ = common8_design(dev, ev_s)
        rows_s = build_rows(ev_s)
        p_s = phase_posterior(rows_s, kappa, weight_mode="sqrt")
        sc_s = score_xgb_rows(
            booster_align, tail_align, rows_s, X_ev_s8, p_s, days=60
        )
        xalign_max_delta = max(xalign_max_delta, float(np.max(np.abs(sc_s - score_0))))

    assert all(c == 4356 for c in param_counts.values())
    assert all(active_checks.values())
    assert paired_init["all_shared_tensors_byte_identical"]
    assert neural_shift_max_delta[CALENDAR_BLIND] == 0.0
    assert neural_shift_max_delta[FLEXIBLE_ELAPSED] == 0.0
    assert neural_shift_max_delta[RELATIVE_PHASE] < 1e-5
    assert neural_shift_max_delta[ABSOLUTE_PHASE] > 1e-3
    assert xalign_max_delta < 1e-10

    return {
        "parameter_counts": param_counts,
        "component_counts": components,
        "all_parameters_active": active_checks,
        "paired_initialization_identical": paired_init["all_shared_tensors_byte_identical"],
        "neural_max_shift_delta": neural_shift_max_delta,
        "xalign_max_shift_delta": xalign_max_delta,
        "estimated_synthetic_kappa": kappa,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calendar-shift-audit",
        description="Calendar-shift invariance auditing and enforcement utilities.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "synthetic-audit",
        help="Run synthetic parameter-matching and 7-shift invariance checks.",
    )

    nc = sub.add_parser(
        "negative-control",
        help="Run the 200-replicate known-nuisance calendar-phase negative control.",
    )
    nc.add_argument("--output-csv", type=Path, default=None)

    bg = sub.add_parser(
        "build-genie-cohort",
        help="Construct recurrent assessment intervals from a local GENIE BPC release directory.",
    )
    bg.add_argument("--release-dir", type=Path, required=True)
    bg.add_argument("--output-parquet", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "synthetic-audit":
        print(json.dumps(run_synthetic_audit(), indent=2, sort_keys=True))
        return 0
    if args.command == "negative-control":
        result = run_known_nuisance_control(output_csv=args.output_csv)
        print(
            json.dumps(
                {
                    "replicates": result["replicates"],
                    "training_sizes": result["training_sizes"],
                    "population_entropy_nats": result["population_entropy_nats"],
                    "summary_rows": len(result["summary"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-genie-cohort":
        patients, marks = load_genie_release_tables(args.release_dir)
        frame, audit, source_counts = build_frame(patients, marks)
        if args.output_parquet is not None:
            args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(args.output_parquet, index=False)
        print(
            json.dumps(
                {
                    "counts": summarize_cohort(frame),
                    "exclusions": exclusion_summary(audit),
                    "source_row_processing": source_counts,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
