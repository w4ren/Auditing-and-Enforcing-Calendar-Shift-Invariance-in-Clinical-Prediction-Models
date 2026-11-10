"""Tests for exact Total Variation distance on geometric-tail mixtures and 7-shift TTA."""

from __future__ import annotations

import numpy as np

from calendar_shift_audit.tv import (
    ensemble_tv,
    mixture_tail_tv,
    single_checkpoint_427_tv,
    tta_mixture_log_scores,
)


def test_mixture_tail_tv_matches_truncated_series() -> None:
    taus = np.array([0.04, 0.11, 0.25], dtype=np.float64)
    rng = np.random.default_rng(42)
    weights_a = rng.dirichlet(np.ones(7), size=(2, 3)) * 0.3
    weights_b = rng.dirichlet(np.ones(7), size=(2, 3)) * 0.3
    delta = (weights_a - weights_b) / 3.0

    cert = mixture_tail_tv(delta, taus, atol=1e-11)
    assert np.all(cert["error_bound"] <= 1e-11)

    w = np.arange(15000, dtype=np.float64)[:, None]
    geom = taus[None, :] * np.power(1.0 - taus[None, :], w)
    brute = np.zeros(2, dtype=np.float64)
    for i in range(2):
        acc = 0.0
        for r in range(7):
            diff_w = geom @ delta[i, :, r]
            acc += 0.5 * np.sum(np.abs(diff_w))
        brute[i] = acc
    assert np.allclose(cert["tv"], brute, atol=1e-9)


def test_ensemble_tv_and_tta_invariance() -> None:
    rng = np.random.default_rng(77)
    pmf_a = rng.dirichlet(np.ones(427), size=(4, 3))
    pmf_b = rng.dirichlet(np.ones(427), size=(4, 3))
    taus = np.array([0.05, 0.10, 0.20], dtype=np.float64)

    zero_res = ensemble_tv(pmf_a, pmf_a, taus, atol=1e-10)
    assert np.allclose(zero_res["tv"], 0.0, atol=1e-10)

    ab = ensemble_tv(pmf_a, pmf_b, taus, atol=1e-10)
    ba = ensemble_tv(pmf_b, pmf_a, taus, atol=1e-10)
    assert np.allclose(ab["tv"], ba["tv"], atol=1e-10)

    single = single_checkpoint_427_tv(pmf_a[:, 0, :], pmf_b[:, 0, :])
    single_ens = ensemble_tv(pmf_a[:, :1, :], pmf_b[:, :1, :], taus[:1], atol=1e-10)["tv"]
    assert np.allclose(single, single_ens, atol=1e-10)

    log_scores = rng.normal(loc=-4.4, scale=0.3, size=(12, 7))
    tta_0 = tta_mixture_log_scores(log_scores)
    for shift in range(1, 7):
        tta_s = tta_mixture_log_scores(np.roll(log_scores, shift=shift, axis=-1))
        assert np.allclose(tta_0, tta_s, atol=1e-14)
