"""Tests for parameter matching, active gradients, PMF normalization, and 7-shift invariance."""

from __future__ import annotations

import numpy as np
import torch

from calendar_shift_audit.cli import _synthetic_recurrent_frame
from calendar_shift_audit.cohort import shift_frame, simple_numeric_categorical_design
from calendar_shift_audit.models import (
    ABSOLUTE_PHASE,
    ALL_NEURAL_ARMS,
    CALENDAR_BLIND,
    FLEXIBLE_ELAPSED,
    RELATIVE_PHASE,
    MatchedNeuralConfig,
    assert_shift_invariant_bundle,
    build_matched_model,
    build_sequences,
    component_parameter_counts,
    expected_parameter_count,
    paired_initialization_summary,
    score_patients,
    to_tensors,
    verify_all_parameters_active,
)


def test_matched_parameter_counts_and_active_gradients() -> None:
    cfg = MatchedNeuralConfig(arm=CALENDAR_BLIND, hidden=64, embed=16, weeks=60)
    assert expected_parameter_count(19, cfg) == 4356

    models = {arm: build_matched_model(arm, n_features=19, seed=7) for arm in ALL_NEURAL_ARMS}
    for arm, model in models.items():
        assert model.n_parameters() == 4356
        comp = component_parameter_counts(model)
        assert comp["total"] == 4356
        assert comp["encoder"] == 19 * 64 + 64 + 64 * 16 + 16
        active = verify_all_parameters_active(model, seed=7)
        assert active["all_active"] is True

    paired = paired_initialization_summary(
        {k: models[k] for k in (CALENDAR_BLIND, ABSOLUTE_PHASE, RELATIVE_PHASE)}
    )
    assert paired["all_shared_tensors_byte_identical"] is True


def test_pmf_normalization_and_seven_shift_invariance() -> None:
    torch.set_default_dtype(torch.float64)
    try:
        frame = _synthetic_recurrent_frame(seed=101)
        dev = frame[frame["institution"] != "VICC"].reset_index(drop=True)
        ev = frame[frame["institution"] == "VICC"].reset_index(drop=True)
        covs = ["start_is_imaging", "cov_a", "cov_b", "cov_c", "cov_d", "cov_e", "cov_f", "cov_g"]

        _, X0, names = simple_numeric_categorical_design(dev, ev, covs, ())
        b0 = build_sequences(ev, X0, names)
        t0 = to_tensors(b0, dtype="float64")

        models = {arm: build_matched_model(arm, n_features=19, seed=5) for arm in ALL_NEURAL_ARMS}
        with torch.no_grad():
            models[ABSOLUTE_PHASE].abs_class.copy_(torch.linspace(-0.3, 0.3, 49).reshape(7, 7))

        for arm, model in models.items():
            pmf = model.predictive_gap_pmf(t0, interval=0, days=2500)
            total_mass = pmf.sum(dim=-1)
            assert torch.all((pmf >= 0.0) & torch.isfinite(pmf))
            assert torch.allclose(total_mass, torch.ones_like(total_mass), atol=1e-5)

        base_scores = {arm: score_patients(m, t0) for arm, m in models.items()}
        patient_shifts = {pid: (idx * 3 + 1) % 7 for idx, pid in enumerate(b0.patient_ids)}
        ev_shifted = shift_frame(ev, patient_shifts)
        _, Xs, _ = simple_numeric_categorical_design(dev, ev_shifted, covs, ())
        bs = build_sequences(ev_shifted, Xs, names)
        assert_shift_invariant_bundle(b0, bs)
        ts = to_tensors(bs, dtype="float64")

        for arm in (CALENDAR_BLIND, FLEXIBLE_ELAPSED, RELATIVE_PHASE):
            sc = score_patients(models[arm], ts)
            assert np.max(np.abs(sc - base_scores[arm])) < 1e-12

        sc_abs = score_patients(models[ABSOLUTE_PHASE], ts)
        assert np.max(np.abs(sc_abs - base_scores[ABSOLUTE_PHASE])) > 1e-3
    finally:
        torch.set_default_dtype(torch.float32)
