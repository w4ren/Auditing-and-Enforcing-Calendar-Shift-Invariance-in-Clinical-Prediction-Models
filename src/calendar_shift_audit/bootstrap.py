"""Paired hospital-cluster and crossed patient-seed bootstrap inference."""

from __future__ import annotations

import hashlib
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_DRAWS = 20000
ALPHA_ORDINARY = 0.05
ALPHA_ROTATION_FWER = 0.05 / 6.0


def percentile_ci(values: np.ndarray, alpha: float = ALPHA_ORDINARY) -> Tuple[float, float]:
    half = 100.0 * float(alpha) / 2.0
    return (
        float(np.percentile(values, half)),
        float(np.percentile(values, 100.0 - half)),
    )


def weighted_ecdf_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float = 0.5
) -> float:
    """Inverse weighted ECDF at q in (0, 1) with nonnegative weights."""
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.shape != w.shape or x.ndim != 1 or len(x) == 0:
        raise ValueError("Expected nonempty 1D values and weights of equal length")
    if (w < 0).any() or float(w.sum()) <= 0.0:
        raise ValueError("Weights must be nonnegative with positive sum")
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ws = w[order] / float(w.sum())
    cdf = np.cumsum(ws)
    idx = int(np.searchsorted(cdf, float(quantile), side="left"))
    idx = min(max(idx, 0), len(xs) - 1)
    return float(xs[idx])


class ClusterBootstrap:
    """Hospital-stratified whole-patient cluster bootstrap for paired contrasts."""

    def __init__(
        self,
        patient_ids: Sequence[str],
        sites: Sequence[str],
        draws: int = DEFAULT_DRAWS,
        seed: int = 0,
    ):
        self.patient_ids = np.asarray(patient_ids, dtype=object)
        self.sites = np.asarray(sites, dtype=object)
        self.draws = int(draws)
        self.seed = int(seed)
        self.site_names: Tuple[str, ...] = tuple(sorted(set(self.sites.tolist())))
        self.position: Dict[str, np.ndarray] = {
            s: np.flatnonzero(self.sites == s) for s in self.site_names
        }

        rng = np.random.default_rng(self.seed)
        self.picks: Dict[str, np.ndarray] = {}
        for site in self.site_names:
            idx = self.position[site]
            choice = rng.integers(0, idx.size, size=(self.draws, idx.size))
            self.picks[site] = idx[choice].astype(np.int32)

    def replicate_site_means(self, delta: np.ndarray) -> Dict[str, np.ndarray]:
        values = np.asarray(delta, dtype=float)
        return {s: values[self.picks[s]].mean(axis=1) for s in self.site_names}

    def replicate_site_equal(self, delta: np.ndarray) -> np.ndarray:
        means = self.replicate_site_means(delta)
        return np.mean(np.column_stack([means[s] for s in self.site_names]), axis=1)

    def replicate_patient_equal(self, delta: np.ndarray) -> np.ndarray:
        values = np.asarray(delta, dtype=float)
        total = np.zeros(self.draws, dtype=float)
        count = 0
        for site in self.site_names:
            total += values[self.picks[site]].sum(axis=1)
            count += self.position[site].size
        return total / float(count)

    def point_site_equal(self, delta: np.ndarray) -> float:
        values = np.asarray(delta, dtype=float)
        return float(
            np.mean([values[self.position[s]].mean() for s in self.site_names])
        )

    def point_patient_equal(self, delta: np.ndarray) -> float:
        return float(np.asarray(delta, dtype=float).mean())

    def contrast(
        self, delta: np.ndarray, alpha: float = ALPHA_ORDINARY
    ) -> Dict[str, object]:
        values = np.asarray(delta, dtype=float)
        site_means = self.replicate_site_means(values)
        site_equal = np.mean(
            np.column_stack([site_means[s] for s in self.site_names]), axis=1
        )
        lo, hi = percentile_ci(site_equal, alpha)
        point = self.point_site_equal(values)

        per_site: Dict[str, Dict[str, object]] = {}
        positive = 0
        min_site = float("inf")
        for site in self.site_names:
            s_lo, s_hi = percentile_ci(site_means[site], alpha)
            est = float(values[self.position[site]].mean())
            per_site[site] = {
                "point": est,
                "ci_lower": s_lo,
                "ci_upper": s_hi,
                "n_patients": int(self.position[site].size),
            }
            positive += int(est > 0.0)
            min_site = min(min_site, est)

        pooled = self.replicate_patient_equal(values)
        p_lo, p_hi = percentile_ci(pooled, alpha)
        return {
            "estimate": point,
            "ci_lower": lo,
            "ci_upper": hi,
            "patient_equal": {
                "estimate": self.point_patient_equal(values),
                "ci_lower": p_lo,
                "ci_upper": p_hi,
            },
            "per_site": per_site,
            "positive_sites": int(positive),
            "n_sites": len(self.site_names),
            "meets_criterion": bool(lo > 0.0 and min_site >= -0.02),
            "bootstrap_replicates": int(self.draws),
            "alpha": float(alpha),
            "n_patients": int(self.patient_ids.size),
            "seed": int(self.seed),
        }

    def index_digest(self) -> str:
        digest = hashlib.sha256()
        for site in self.site_names:
            digest.update(np.ascontiguousarray(self.picks[site], dtype=np.int32).tobytes())
        return digest.hexdigest()


def crossed_patient_seed_bootstrap(
    site_matrices: Mapping[str, np.ndarray],
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = 20260828,
    alpha: float = ALPHA_ORDINARY,
) -> Dict[str, object]:
    """Paired crossed whole-patient and training-seed bootstrap across hospitals.

    Each entry in ``site_matrices`` has shape ``(P_h, S, ...)`` for hospital ``h``,
    where ``P_h`` is the number of patients at hospital ``h`` and ``S`` is the
    number of paired training seeds. Patients are resampled independently within
    each hospital and paired seeds are resampled jointly across hospitals.
    """
    site_names = sorted(site_matrices.keys())
    if not site_names:
        raise ValueError("site_matrices cannot be empty")
    first = np.asarray(site_matrices[site_names[0]], dtype=np.float64)
    if first.ndim < 2:
        raise ValueError("Each hospital matrix must have shape (P_h, S, ...)")
    n_seeds = first.shape[1]
    trailing = first.shape[2:]

    arrays = {}
    for s in site_names:
        arr = np.asarray(site_matrices[s], dtype=np.float64)
        if arr.shape[1:] != (n_seeds,) + trailing:
            raise ValueError(f"Mismatched seed or metric shape for hospital {s}")
        arrays[s] = arr

    rng = np.random.default_rng(int(seed))
    seed_picks = rng.integers(0, n_seeds, size=(int(draws), n_seeds))
    seed_weights = np.zeros((int(draws), n_seeds), dtype=np.float64)
    np.add.at(
        seed_weights,
        (np.arange(int(draws))[:, None], seed_picks),
        1.0 / float(n_seeds),
    )

    per_site_draws: Dict[str, np.ndarray] = {}
    per_site_point: Dict[str, np.ndarray] = {}
    for s in site_names:
        arr = arrays[s]
        n_p = arr.shape[0]
        p_picks = rng.integers(0, n_p, size=(int(draws), n_p))
        p_weights = np.zeros((int(draws), n_p), dtype=np.float64)
        np.add.at(
            p_weights,
            (np.arange(int(draws))[:, None], p_picks),
            1.0 / float(n_p),
        )
        flat = arr.reshape(n_p, -1)
        p_avg = (p_weights @ flat).reshape((int(draws), n_seeds) + trailing)
        s_avg = np.einsum("bs...,bs->b...", p_avg, seed_weights)
        per_site_draws[s] = s_avg
        per_site_point[s] = arr.mean(axis=(0, 1))

    overall_point = np.mean([per_site_point[s] for s in site_names], axis=0)
    overall_draws = np.mean([per_site_draws[s] for s in site_names], axis=0)
    half = 100.0 * float(alpha) / 2.0
    lo = np.percentile(overall_draws, half, axis=0)
    hi = np.percentile(overall_draws, 100.0 - half, axis=0)
    return {
        "estimate": overall_point,
        "ci_lower": lo,
        "ci_upper": hi,
        "bootstrap_sd": overall_draws.std(axis=0, ddof=0),
        "per_site_estimate": per_site_point,
        "draws": int(draws),
        "seed": int(seed),
    }


def align_patient_series(
    frames: Mapping[str, pd.Series],
) -> Tuple[np.ndarray, pd.Index]:
    common: Optional[pd.Index] = None
    for series in frames.values():
        idx = pd.Index(series.index)
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) == 0:
        raise ValueError("No common patients across model arms")
    ordered = pd.Index(sorted(common.tolist()))
    return np.asarray(ordered.tolist(), dtype=object), ordered
