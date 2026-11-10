"""Tests for glucose timestamp canonicalization, glycemic categories, and cluster bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calendar_shift_audit.bootstrap import (
    ClusterBootstrap,
    crossed_patient_seed_bootstrap,
    weighted_ecdf_quantile,
)
from calendar_shift_audit.glucose import (
    GLUCOSE_METRICS,
    calendar_day_fields,
    canonical_relative_dates,
    consensus_glucose_category,
    patient_anchor_schedule,
    patient_glucose_statistics,
)


def test_consensus_categories_and_canonical_relative_timestamps() -> None:
    vals = np.array([55.0, 69.99, 70.0, 125.0, 180.0, 180.01, 250.0])
    cats = consensus_glucose_category(vals)
    assert list(cats) == [-1, -1, 0, 0, 0, 1, 1]

    ts_base = pd.date_range("2023-05-10 08:15:00", periods=6, freq="5min")
    anchor_base = int(ts_base[0].value)
    canon_0 = canonical_relative_dates(ts_base, anchor_base)

    for shift_days in (1, 3, 6, 19, 120):
        ts_shifted = ts_base + pd.Timedelta(days=shift_days)
        anchor_shifted = int(ts_shifted[0].value)
        canon_s = canonical_relative_dates(ts_shifted, anchor_shifted)
        assert (canon_0 == canon_s).all()
        assert np.array_equal(calendar_day_fields(canon_0), calendar_day_fields(canon_s))


def test_anchor_schedule_and_patient_glucose_statistics() -> None:
    pids, sched, _ = patient_anchor_schedule([10, 20, 30, 40], seed=3, epochs=16)
    assert list(pids) == [10, 20, 30, 40]
    assert sched.shape == (16, 4)
    for col in range(4):
        assert len(set(sched[:8, col].tolist())) == 8

    truth = np.full((4, 3), 100.0)
    invariant_pred = np.broadcast_to(np.array([100.0, 120.0, 140.0]), (2, 1, 7, 4, 3)).copy()
    stats_inv = patient_glucose_statistics(invariant_pred, truth, ["A", "A", "B", "B"])
    assert np.allclose(stats_inv["cube"], 0.0)

    shifted_pred = invariant_pred.copy()
    shifted_pred[:, 0, 1:, :, :] = 195.0
    stats_shift = patient_glucose_statistics(shifted_pred, truth, ["A", "A", "B", "B"])
    any_cat_idx = GLUCOSE_METRICS.index("patient_has_any_category_change")
    assert np.allclose(stats_shift["cube"][:, :, 0, any_cat_idx], 1.0)


def test_cluster_and_crossed_bootstrap() -> None:
    q50 = weighted_ecdf_quantile(np.array([1.0, 2.0, 10.0]), np.array([0.25, 0.50, 0.25]), 0.5)
    assert q50 == 2.0

    pids = [f"P{i}" for i in range(20)]
    sites = ["DFCI"] * 10 + ["MSK"] * 10
    delta = np.linspace(0.05, 0.15, 20)
    boot = ClusterBootstrap(pids, sites, draws=1000, seed=123)
    res = boot.contrast(delta)
    assert res["ci_lower"] > 0.0
    assert res["positive_sites"] == 2

    crossed = crossed_patient_seed_bootstrap(
        {"DFCI": np.ones((8, 5)) * 0.1, "MSK": np.ones((6, 5)) * 0.2},
        draws=500,
        seed=123,
    )
    assert np.isclose(crossed["estimate"], 0.15)
    assert np.isclose(crossed["ci_lower"], 0.15)
    assert np.isclose(crossed["ci_upper"], 0.15)
