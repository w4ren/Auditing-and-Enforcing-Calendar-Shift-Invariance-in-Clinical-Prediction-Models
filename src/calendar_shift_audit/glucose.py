"""External glucose forecasting calendar-shift audit metrics and timestamp remedies."""

from __future__ import annotations

import hashlib
import itertools
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd


SHIFT_PAIRS: Tuple[Tuple[int, int], ...] = tuple(itertools.combinations(range(7), 2))

GLUCOSE_METRICS: Tuple[str, ...] = (
    "typical_nonzero_displacement",
    "typical_including_identity_displacement",
    "worst_consistent_pair_displacement",
    "mean_pointwise_prediction_range",
    "typical_nonzero_category_change",
    "typical_nonzero_low_indicator_change",
    "typical_nonzero_high_indicator_change",
    "any_offset_category_change",
    "any_offset_low_indicator_change",
    "any_offset_high_indicator_change",
    "patient_has_any_category_change",
)

OFFSET_METRICS: Tuple[str, ...] = (
    "absolute_displacement",
    "category_change",
    "low_indicator_change",
    "high_indicator_change",
)


def consensus_glucose_category(
    values_mgdl: np.ndarray,
    low_threshold: float = 70.0,
    high_threshold: float = 180.0,
) -> np.ndarray:
    """Classify glucose predictions into <70 (-1), 70-180 (0), and >180 (+1) mg/dL."""
    x = np.asarray(values_mgdl, dtype=np.float64)
    return np.where(x < low_threshold, -1, np.where(x > high_threshold, 1, 0)).astype(
        np.int8
    )


def canonical_relative_dates(
    timestamps: pd.DatetimeIndex,
    anchor_ns: int,
    origin_date: str = "2000-01-03",
) -> pd.DatetimeIndex:
    """Map participant timestamps to a fixed relative calendar origin preserving time-of-day."""
    idx = pd.DatetimeIndex(timestamps)
    return idx - pd.Timestamp(anchor_ns).normalize() + pd.Timestamp(origin_date)


def calendar_day_fields(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Extract (day_of_year, day_of_month, weekday) integer covariates."""
    dates = pd.DatetimeIndex(timestamps)
    return np.column_stack([dates.dayofyear, dates.day, dates.dayofweek]).astype(
        np.int64
    )


def patient_anchor_schedule(
    patients: Sequence[int],
    seed: int,
    epochs: int = 100,
    train_offsets: Sequence[int] = range(8),
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Generate balanced per-patient synthetic-anchor offsets across 8-epoch cycles."""
    patient_arr = np.asarray(sorted(set(map(int, patients))), dtype=np.int64)
    offsets = np.asarray(list(train_offsets), dtype=np.int8)
    n_off = len(offsets)
    token = f"colas_random_anchor_v1/calendar_schedule/{seed}"
    random_seed = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(random_seed)
    schedule = np.empty((epochs, len(patient_arr)), dtype=np.int8)

    for first in range(0, epochs, n_off):
        order = rng.permutation(len(patient_arr))
        labels = rng.permutation(offsets)
        for phase in range(min(n_off, epochs - first)):
            schedule[first + phase, order] = labels[
                (np.arange(len(patient_arr)) + phase) % n_off
            ]
    return patient_arr, schedule, random_seed


def patient_glucose_statistics(
    predictions: np.ndarray,
    truth: np.ndarray,
    patient_ids: Sequence[Any],
    *,
    low_threshold: float = 70.0,
    high_threshold: float = 180.0,
    preserve_patient_order: bool = False,
) -> Dict[str, Any]:
    """Compute per-patient displacement, category-change, and SSE summaries across 7 shifts.

    Parameters
    ----------
    predictions : ndarray of shape (S, M, 7, W, H)
        Predicted glucose trajectories across S seeds, M model arms, 7 shifts,
        W windows, and H forecast steps (e.g. 12 steps = 60 minutes).
    truth : ndarray of shape (W, H)
        Ground-truth glucose targets in mg/dL.
    patient_ids : sequence of length W
        Participant identifier for each evaluation window.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    y_true = np.asarray(truth, dtype=np.float64)
    ids = np.asarray(patient_ids)
    n_seeds, n_models, n_shifts, n_windows, horizon = pred.shape
    if n_shifts != 7:
        raise ValueError(f"Expected 7 shifts along axis 2, got {n_shifts}")
    if y_true.shape != (n_windows, horizon) or len(ids) != n_windows:
        raise ValueError("Mismatched truth or patient_ids dimensions")

    roster = pd.unique(ids) if preserve_patient_order else np.unique(ids)
    groups = [np.flatnonzero(ids == p) for p in roster]
    n_patients = len(groups)

    cube = np.empty((n_patients, n_seeds, n_models, len(GLUCOSE_METRICS)), dtype=np.float64)
    offsets = np.empty((n_patients, n_seeds, n_models, 7, 4), dtype=np.float64)
    sse = np.empty((n_patients, n_seeds, n_models, 7), dtype=np.float64)
    point_counts = np.asarray([len(g) * horizon for g in groups], dtype=np.int64)
    maximum_ranges = np.empty((n_seeds, n_models), dtype=np.float64)
    maximum_from_zero = np.empty((n_seeds, n_models), dtype=np.float64)

    for a in range(n_models):
        for s in range(n_seeds):
            values = pred[s, a]
            labels = consensus_glucose_category(
                values, low_threshold=low_threshold, high_threshold=high_threshold
            )
            point_range = values.max(0) - values.min(0)
            zero_abs = np.abs(values - values[0])
            zero_cat = labels != labels[0]
            zero_low = (labels == -1) != (labels[0] == -1)
            zero_high = (labels == 1) != (labels[0] == 1)
            any_cat = labels.max(0) != labels.min(0)
            any_low = (labels == -1).any(0) & ~(labels == -1).all(0)
            any_high = (labels == 1).any(0) & ~(labels == 1).all(0)

            maximum_ranges[s, a] = float(point_range.max())
            maximum_from_zero[s, a] = float(zero_abs.max())
            errors = np.square(values - y_true)
            pair_displacements = np.asarray(
                [
                    [
                        np.abs(values[left, g] - values[right, g]).mean()
                        for left, right in SHIFT_PAIRS
                    ]
                    for g in groups
                ]
            )
            for p_idx, g in enumerate(groups):
                typical = float(zero_abs[1:, g].mean())
                worst = float(pair_displacements[p_idx].max())
                range_mean = float(point_range[g].mean())
                cube[p_idx, s, a] = [
                    typical,
                    typical * 6.0 / 7.0,
                    worst,
                    range_mean,
                    float(zero_cat[1:, g].mean()),
                    float(zero_low[1:, g].mean()),
                    float(zero_high[1:, g].mean()),
                    float(any_cat[g].mean()),
                    float(any_low[g].mean()),
                    float(any_high[g].mean()),
                    float(any_cat[g].any()),
                ]
                offsets[p_idx, s, a] = np.column_stack(
                    [
                        v[:, g].mean(axis=(1, 2))
                        for v in (zero_abs, zero_cat, zero_low, zero_high)
                    ]
                )
                sse[p_idx, s, a] = errors[:, g].sum(axis=(1, 2))

    return {
        "cube": cube,
        "offsets": offsets,
        "sse": sse,
        "point_counts": point_counts,
        "roster": roster,
        "maximum_ranges": maximum_ranges,
        "maximum_from_zero": maximum_from_zero,
    }


def glucose_bootstrap_schedules(
    dataset: str, n_patients: int, n_seeds: int = 3, draws: int = 20000
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Generate paired participant (and optional crossed seed) bootstrap weight matrices."""
    is_cgmacros = dataset.lower() == "cgmacros"
    seed = 20260913 if is_cgmacros else 20260909
    rng = np.random.default_rng(seed)
    if is_cgmacros:
        positions = rng.integers(0, n_patients, size=(draws, n_patients))
        pw = np.zeros((draws, n_patients), dtype=np.float64)
        np.add.at(pw, (np.arange(draws)[:, None], positions), 1.0 / n_patients)
        sw = np.full((draws, n_seeds), 1.0 / n_seeds, dtype=np.float64)
    else:
        pw = rng.multinomial(n_patients, np.full(n_patients, 1.0 / n_patients), size=draws) / float(n_patients)
        sw = rng.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=draws) / float(n_seeds)
    return pw, sw, seed
