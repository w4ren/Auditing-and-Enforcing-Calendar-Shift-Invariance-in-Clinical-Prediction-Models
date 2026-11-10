"""Tests for GENIE and glucose campaign runners and edge-case numerical guards."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_genie_campaign import run_neural_fold, run_xalign_fold
from run_glucose_campaign import evaluate_prediction_archive

from calendar_shift_audit.cohort import build_frame, load_genie_release_tables
from calendar_shift_audit.models import (
    ABSOLUTE_PHASE,
    CALENDAR_BLIND,
    RELATIVE_PHASE,
    build_matched_model,
    build_sequences,
    paired_initialization_summary,
    to_tensors,
)
from calendar_shift_audit.tv import tta_mixture_log_scores
from calendar_shift_audit.xalign import (
    build_rows,
    common8_design,
    common8_history_columns,
    fit_xgb_discrete_hazard,
    score_xgb_rows,
)


def test_genie_runner_uses_19_features_and_audits_8_boosted_arms(tmp_path: Path) -> None:
    rng = np.random.default_rng(515)
    sites = ["DFCI", "MSK", "UHN", "VICC"]
    pat_rows = []
    img_rows = []
    onc_rows = []
    for i in range(24):
        pid = f"GENIE-SYN-{i:03d}"
        site = sites[i % 4]
        pat_rows.append(
            {
                "record_id": pid,
                "institution": site,
                "dob_lastalive_int": 400,
                "hybrid_death_int": np.nan,
            }
        )
        cur = int(rng.integers(5, 25))
        for step in range(4):
            cur += int(rng.integers(14, 42))
            if step % 2 == 0:
                img_rows.append({"record_id": pid, "image_scan_int": cur})
            else:
                onc_rows.append({"record_id": pid, "md_onc_visit_int": cur})

    pd.DataFrame(pat_rows).to_csv(tmp_path / "patient_level_dataset.csv", index=False)
    pd.DataFrame(img_rows).to_csv(tmp_path / "imaging_level_dataset.csv", index=False)
    pd.DataFrame(onc_rows).to_csv(tmp_path / "med_onc_note_level_dataset.csv", index=False)

    patients, marks = load_genie_release_tables(tmp_path)
    frame, _, _ = build_frame(patients, marks)

    neural_res = run_neural_fold(frame, "VICC", seed=3, device="cpu", epochs=3)
    assert len(neural_res) == 4
    for rec in neural_res:
        assert rec["n_features"] == 19
        assert rec["parameters"] == 4356

    xalign_res = run_xalign_fold(frame, "VICC", seed=3)
    assert len(xalign_res) == 8
    for rec in xalign_res:
        assert rec["all_interval_max_spread"] < 1e-10


def test_glucose_runner_preserves_all_models_and_seed_uncertainty(tmp_path: Path) -> None:
    pred = np.zeros((2, 2, 7, 4, 2), dtype=np.float64)
    pred[:, 1, 1:, :, :] = 100.0
    npz_two_models = tmp_path / "two_models.npz"
    np.savez(
        npz_two_models,
        predictions=pred,
        targets=np.zeros((4, 2)),
        patient_ids=np.array(["P1", "P2", "P3", "P4"]),
        model_names=np.array(["invariant", "sensitive"]),
    )
    out = evaluate_prediction_archive(npz_two_models, dataset="colas", bootstrap_draws=500)
    assert out["models"]["invariant"]["typical_nonzero_displacement_mgdl"]["estimate"] == 0.0
    assert out["models"]["sensitive"]["typical_nonzero_displacement_mgdl"]["estimate"] == 100.0

    pred_seed = np.zeros((2, 1, 7, 4, 2), dtype=np.float64)
    pred_seed[1, 0, 1:, :, :] = 100.0
    npz_seed = tmp_path / "two_seeds.npz"
    np.savez(
        npz_seed,
        predictions=pred_seed,
        targets=np.zeros((4, 2)),
        patient_ids=np.array(["P1", "P2", "P3", "P4"]),
    )
    out_colas = evaluate_prediction_archive(npz_seed, dataset="colas", bootstrap_draws=1000)
    ci = out_colas["models"]["model_0"]["typical_nonzero_displacement_mgdl"]
    assert ci["estimate"] == 50.0
    assert ci["ci_lower"] == 0.0
    assert ci["ci_upper"] == 100.0


def test_numerical_and_indexing_edge_cases() -> None:
    df = pd.DataFrame(
        [
            {"record_id": "B", "institution": "MSK", "interval_start": 50.0, "interval_stop": 80.0, "exposure_days": 30.0, "event": 1, "start_is_imaging": 1},
            {"record_id": "A", "institution": "DFCI", "interval_start": 40.0, "interval_stop": 70.0, "exposure_days": 30.0, "event": 1, "start_is_imaging": 0},
            {"record_id": "A", "institution": "DFCI", "interval_start": 10.0, "interval_stop": 40.0, "exposure_days": 30.0, "event": 1, "start_is_imaging": 1},
        ],
        index=[10, 25, 42],
    )
    hist = common8_history_columns(df)
    assert list(hist.index) == [10, 25, 42]
    assert hist.loc[42, "order_index"] == 0.0
    assert hist.loc[25, "order_index"] == 0.1

    rows = build_rows(df)
    X_dev, _, _ = common8_design(df, df)
    assert list(rows.record_id) == ["A", "A", "B"]
    assert X_dev.shape == (3, 8)

    long_df = pd.DataFrame(
        [
            {"record_id": f"P{i}", "institution": "DFCI", "interval_start": 0.0, "interval_stop": 120.0, "exposure_days": 120.0, "event": i % 2, "start_is_imaging": 1}
            for i in range(8)
        ]
    )
    l_rows = build_rows(long_df)
    X_l, _, _ = common8_design(long_df, long_df)
    booster, tail = fit_xgb_discrete_hazard(
        l_rows, X_l, None, xgb_params={"eta": 0.3, "max_depth": 2, "min_child_weight": 1.0, "num_boost_round": 20}
    )
    scores = score_xgb_rows(booster, tail, l_rows, X_l, None)
    assert np.all(np.isfinite(scores))
    assert np.all(scores > -500.0)

    drop_models = {
        arm: build_matched_model(arm, n_features=19, seed=9, dropout=0.1)
        for arm in (CALENDAR_BLIND, ABSOLUTE_PHASE, RELATIVE_PHASE)
    }
    assert paired_initialization_summary(drop_models)["all_shared_tensors_byte_identical"] is True

    torch.set_default_dtype(torch.float64)
    try:
        m_rot = build_matched_model(RELATIVE_PHASE, n_features=19, seed=4)
        m_rot.eval_rotation = 3
        b_l = build_sequences(long_df, X_l, tuple(f"f{i}" for i in range(8)))
        t_l = to_tensors(b_l, dtype="float64")
        row_ll = m_rot.predictive_rows(t_l)[:, 0]
        pmf_120 = m_rot.predictive_gap_pmf(t_l, interval=0, days=120)
        ev_mask = long_df["event"].to_numpy() == 1
        assert np.allclose(
            row_ll[ev_mask].detach().numpy(),
            np.log(pmf_120[ev_mask, 119].detach().numpy()),
            atol=1e-12,
        )
    finally:
        torch.set_default_dtype(torch.float32)

    all_neg_inf = np.full((2, 7), -np.inf)
    tta = tta_mixture_log_scores(all_neg_inf)
    assert np.all(np.isneginf(tta))
