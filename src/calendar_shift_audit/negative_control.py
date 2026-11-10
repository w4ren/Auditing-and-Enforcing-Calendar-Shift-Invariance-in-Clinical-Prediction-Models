"""Synthetic known-nuisance calendar-phase negative control simulation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_TRAINING_SIZES: Tuple[int, ...] = (200, 2000, 20000)
DEFAULT_REPLICATES = 200
DEFAULT_SEED = 2026100801
DEFAULT_ALPHA = 0.5
DEFAULT_TARGET_PMF: Tuple[float, ...] = (0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

METRICS: Tuple[str, ...] = (
    "mean_nonzero_shift_tv",
    "maximum_pair_tv",
    "expected_log_loss",
    "excess_expected_log_loss",
)


def _table_metrics(q: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    if q.shape != (7, 7) or not np.isfinite(q).all() or not (q > 0).all():
        raise ValueError("Expected positive finite 7x7 conditional probability table")
    if not np.allclose(q.sum(axis=1), 1.0, rtol=0.0, atol=1e-14):
        raise ValueError("Rows of conditional probability table must sum to 1")
    tv_matrix = 0.5 * np.abs(q[:, None, :] - q[None, :, :]).sum(axis=2)
    pairs = tv_matrix[np.triu_indices(7, k=1)]
    cyclic_tv = float(
        np.mean([tv_matrix[u, (u + k) % 7] for u in range(7) for k in range(1, 7)])
    )
    if abs(float(pairs.mean()) - cyclic_tv) >= 1e-14:
        raise AssertionError("Pairwise and cyclic nonzero TV disagree")
    entropy = -float(np.dot(p, np.log(p)))
    loss = -float(np.mean(np.sum(p[None, :] * np.log(q), axis=1)))
    excess = float(np.mean(np.sum(p[None, :] * np.log(p[None, :] / q), axis=1)))
    if abs(excess - (loss - entropy)) >= 1e-12 or excess < -1e-12:
        raise AssertionError("KL divergence and excess log loss identity violated")
    return {
        "mean_nonzero_shift_tv": float(pairs.mean()),
        "maximum_pair_tv": float(pairs.max()),
        "expected_log_loss": loss,
        "excess_expected_log_loss": excess,
    }


def _describe_replicates(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    return {
        "mean": float(arr.mean()),
        "sample_sd": sd,
        "mc_se_mean": sd / math.sqrt(max(n, 1)),
        "fit_percentile_2_5": float(np.quantile(arr, 0.025)),
        "fit_percentile_97_5": float(np.quantile(arr, 0.975)),
    }


def run_known_nuisance_control(
    *,
    training_sizes: Sequence[int] = DEFAULT_TRAINING_SIZES,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    target_pmf: Sequence[float] = DEFAULT_TARGET_PMF,
    output_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    """Simulate unrestricted vs. pooled and shift-averaged estimators under an independent phase."""
    sizes = tuple(int(n) for n in training_sizes)
    p = np.asarray(target_pmf, dtype=float)
    if p.shape != (7,) or not np.isclose(p.sum(), 1.0):
        raise ValueError("target_pmf must be a normalized 7-category distribution")

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    rows: List[Dict[str, Any]] = []
    paired: List[Dict[str, Any]] = []
    max_n = max(sizes)

    for rep in range(int(replicates)):
        u = rng.integers(0, 7, size=max_n)
        y = rng.choice(7, size=max_n, p=p)
        for n in sizes:
            counts = np.bincount(u[:n] * 7 + y[:n], minlength=49).reshape(7, 7)
            abs_q = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + 7.0 * alpha)
            pool_q = (counts.sum(axis=0) + alpha) / (float(n) + 7.0 * alpha)
            avg_q = abs_q.mean(axis=0)
            for k in range(7):
                if not np.allclose(
                    np.roll(abs_q, k, axis=0).mean(axis=0), avg_q, rtol=0.0, atol=1e-14
                ):
                    raise AssertionError("Shift-average failed cyclic invariance")
            matrices = {
                "absolute_table": abs_q,
                "pooled_shared": np.tile(pool_q, (7, 1)),
                "shift_average": np.tile(avg_q, (7, 1)),
                "population_oracle": np.tile(p, (7, 1)),
            }
            metrics = {name: _table_metrics(mat, p) for name, mat in matrices.items()}
            for name, vals in metrics.items():
                if name != "absolute_table":
                    if vals["mean_nonzero_shift_tv"] != 0.0 or vals["maximum_pair_tv"] != 0.0:
                        raise AssertionError(f"Nonzero shift TV in invariant estimator {name}")
                rows.append({"replicate": rep, "training_n": n, "model": name, **vals})
            if (
                metrics["shift_average"]["expected_log_loss"]
                > metrics["absolute_table"]["expected_log_loss"] + 1e-12
            ):
                raise AssertionError("Jensen inequality violated for shift_average")
            for left, right in (
                ("absolute_table", "pooled_shared"),
                ("absolute_table", "shift_average"),
                ("pooled_shared", "shift_average"),
            ):
                paired.append(
                    {
                        "replicate": rep,
                        "training_n": n,
                        "left": left,
                        "right": right,
                        **{k: metrics[left][k] - metrics[right][k] for k in METRICS},
                    }
                )

    summary: List[Dict[str, Any]] = []
    paired_summary: List[Dict[str, Any]] = []
    model_order = ("absolute_table", "pooled_shared", "shift_average", "population_oracle")
    for n in sizes:
        for model in model_order:
            sel = [r for r in rows if r["training_n"] == n and r["model"] == model]
            for metric in METRICS:
                summary.append(
                    {
                        "training_n": n,
                        "model": model,
                        "metric": metric,
                        **_describe_replicates([r[metric] for r in sel]),
                    }
                )
        for left, right in (
            ("absolute_table", "pooled_shared"),
            ("absolute_table", "shift_average"),
            ("pooled_shared", "shift_average"),
        ):
            sel = [
                r
                for r in paired
                if r["training_n"] == n and r["left"] == left and r["right"] == right
            ]
            for metric in METRICS:
                paired_summary.append(
                    {
                        "training_n": n,
                        "left": left,
                        "right": right,
                        "metric": metric,
                        **_describe_replicates([r[metric] for r in sel]),
                    }
                )

    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    return {
        "replicates": int(replicates),
        "training_sizes": list(sizes),
        "seed": int(seed),
        "population_entropy_nats": -float(np.dot(p, np.log(p))),
        "summary": summary,
        "paired_summary": paired_summary,
    }
