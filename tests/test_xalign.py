"""Tests for phase-posterior equivariance, rotated controls, and X-align shift invariance."""

from __future__ import annotations

import numpy as np

from calendar_shift_audit.cli import _synthetic_recurrent_frame
from calendar_shift_audit.cohort import shift_frame
from calendar_shift_audit.xalign import (
    build_rows,
    common8_design,
    estimate_phase_kappa,
    fit_xgb_discrete_hazard,
    phase_alignment_features,
    phase_posterior,
    score_xgb_rows,
)


def test_phase_posterior_equivariance_and_feature_invariance() -> None:
    frame = _synthetic_recurrent_frame(seed=202)
    rows_0 = build_rows(frame)
    kappa = 0.95
    post_0 = phase_posterior(rows_0, kappa, weight_mode="sqrt")
    assert np.allclose(post_0.sum(axis=1), 1.0)
    first_rows = rows_0.order_in_patient == 0
    start_mod_0 = np.mod(np.round(rows_0.start[first_rows]).astype(int), 7)
    expected_peak = np.exp(kappa) / (np.exp(kappa) + 6.0)
    assert np.allclose(post_0[first_rows, start_mod_0], expected_peak, atol=1e-12)

    candidate_days = np.arange(1, 29)
    for delta in range(1, 7):
        rows_s = build_rows(shift_frame(frame, delta))
        post_s = phase_posterior(rows_s, kappa, weight_mode="sqrt")
        expected = np.roll(post_0, shift=delta, axis=1)
        assert np.allclose(post_s, expected, atol=1e-14)

        for pos in (0, 3, 10):
            f0 = phase_alignment_features(rows_0, post_0, candidate_days, pos)
            fs = phase_alignment_features(rows_s, post_s, candidate_days, pos)
            assert np.allclose(f0, fs, atol=1e-14)

        for rot in range(1, 7):
            p_rot0 = phase_posterior(rows_0, kappa, weight_mode="sqrt", rotate_fixed=rot)
            p_rots = phase_posterior(rows_s, kappa, weight_mode="sqrt", rotate_fixed=rot)
            f_r0 = phase_alignment_features(rows_0, p_rot0, candidate_days, 5)
            f_rs = phase_alignment_features(rows_s, p_rots, candidate_days, 5)
            assert np.allclose(f_r0, f_rs, atol=1e-14)


def test_xalign_booster_is_invariant_to_per_patient_shifts() -> None:
    frame = _synthetic_recurrent_frame(seed=303)
    dev = frame[frame["institution"] != "DFCI"].reset_index(drop=True)
    ev = frame[frame["institution"] == "DFCI"].reset_index(drop=True)

    X_dev, X_ev, _ = common8_design(dev, ev)
    dev_rows = build_rows(dev)
    ev_rows = build_rows(ev)
    kappa = estimate_phase_kappa(dev_rows, X_dev, iterations=30)
    assert kappa > 0.0

    p_dev = phase_posterior(dev_rows, kappa)
    p_ev = phase_posterior(ev_rows, kappa)
    params = {"eta": 0.05, "max_depth": 3, "min_child_weight": 5.0, "num_boost_round": 25}
    booster, tail = fit_xgb_discrete_hazard(dev_rows, X_dev, p_dev, seed=3, xgb_params=params, days=42)
    s0 = score_xgb_rows(booster, tail, ev_rows, X_ev, p_ev, days=42)

    patient_shifts = {pid: (i * 5 + 2) % 7 for i, pid in enumerate(sorted(ev["record_id"].unique()))}
    ev_shifted = shift_frame(ev, patient_shifts)
    _, X_ev_s, _ = common8_design(dev, ev_shifted)
    ev_rows_s = build_rows(ev_shifted)
    p_ev_s = phase_posterior(ev_rows_s, kappa)
    ss = score_xgb_rows(booster, tail, ev_rows_s, X_ev_s, p_ev_s, days=42)
    assert np.allclose(s0, ss, atol=1e-12)
