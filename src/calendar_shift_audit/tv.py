"""Exact and certified Total Variation (TV) distance and seven-origin mixture scoring."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import numpy as np


_EPS_GUARD = 256 * np.finfo(np.float64).eps


def _outward(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.maximum(0.0, np.nextafter(lower, -np.inf))
    upper = np.nextafter(upper, np.inf)
    return lower, upper


def _signed_integral_bounds(
    terms: np.ndarray, pad: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    signed = np.abs(terms.sum(axis=1))
    return _outward(np.maximum(0.0, signed - pad), signed + pad)


def _abel_sign(
    coefficients: np.ndarray,
    taus: np.ndarray,
    coefficient_error: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Summation-by-parts sign certificate on [0, inf) for rates sorted slow-to-fast."""
    terms = coefficients * taus[None, :]
    prefix = np.cumsum(terms, axis=1)
    error = _EPS_GUARD * np.cumsum(np.abs(terms), axis=1)
    if coefficient_error is not None:
        error = error + np.cumsum(coefficient_error * taus[None, :], axis=1)
    return np.all(prefix >= error, axis=1) | np.all(prefix <= -error, axis=1)


def _powers(log_rates: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore"):
        exponent = positions[:, None] * log_rates[None, :]
    exponent = np.where(positions[:, None] == 0, 0.0, exponent)
    return np.exp(exponent), exponent


def _block_bounds(
    coefficients: np.ndarray,
    taus: np.ndarray,
    log_rates: np.ndarray,
    left: np.ndarray,
    width: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    power, exponent = _powers(log_rates, left)
    remaining = coefficients * power
    with np.errstate(invalid="ignore"):
        fraction = -np.expm1(width[:, None] * log_rates[None, :])
    terms = remaining * fraction
    pad = _EPS_GUARD * np.sum(
        np.abs(remaining) * (3 + np.minimum(np.abs(exponent), 750)), axis=1
    )
    lo, signed_hi = _signed_integral_bounds(terms, pad)
    hi = np.abs(terms).sum(axis=1) + pad

    fleft = remaining * taus[None, :]
    with np.errstate(invalid="ignore"):
        ratio_exponent = (width[:, None] - 1) * (log_rates - log_rates[0])[None, :]
    ratio_exponent = np.where(width[:, None] == 1, 0.0, ratio_exponent)
    ratio = np.exp(ratio_exponent)
    low_f = np.where(fleft >= 0, fleft * ratio, fleft).sum(axis=1)
    high_f = np.where(fleft >= 0, fleft, fleft * ratio).sum(axis=1)
    sign_guard = _EPS_GUARD * np.sum(
        np.abs(fleft)
        * (
            4
            + np.minimum(np.abs(exponent), 750)
            + np.minimum(np.abs(ratio_exponent), 750)
        ),
        axis=1,
    )
    sign = (low_f >= sign_guard) | (high_f <= -sign_guard)
    sign |= _abel_sign(
        remaining,
        taus,
        _EPS_GUARD * np.abs(remaining) * (3 + np.minimum(np.abs(exponent), 750)),
    )
    sign |= width == 1
    hi = np.where(sign, signed_hi, hi)
    lo, hi = _outward(lo, hi)
    return lo, hi, sign


def mixture_tail_tv(
    tail_delta: np.ndarray,
    taus: np.ndarray,
    atol: float = 1e-9,
    max_nodes: int = 2_000_000,
    batch_size: int = 512,
) -> Dict[str, Any]:
    """Certified Total Variation contribution over seven infinite geometric tail channels."""
    started = time.perf_counter()
    delta = np.asarray(tail_delta, dtype=np.float64)
    tau = np.asarray(taus, dtype=np.float64)
    if (
        delta.ndim < 2
        or delta.shape[-1] != 7
        or tau.ndim != 1
        or delta.shape[-2] != len(tau)
    ):
        raise ValueError(
            "Expected (..., K, 7) signed tail masses and length-K hazard vector"
        )
    if (
        not np.isfinite(delta).all()
        or not np.isfinite(tau).all()
        or not ((tau > 0) & (tau <= 1)).all()
    ):
        raise ValueError("Invalid signed tail mass or geometric hazard")
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive")

    leading = delta.shape[:-2]
    count = int(np.prod(leading)) if leading else 1
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if count > batch_size:
        flat = delta.reshape(count, len(tau), 7)
        pieces = [
            mixture_tail_tv(
                flat[start : start + batch_size], tau, atol, max_nodes, batch_size
            )
            for start in range(0, count, batch_size)
        ]
        answer: Dict[str, Any] = {
            name: np.concatenate([p[name] for p in pieces]).reshape(leading)
            for name in ("tv", "error_bound", "lower", "upper")
        }
        answer["stats"] = {
            "forecasts": count,
            "checkpoints": len(tau),
            "adaptive_nodes": sum(p["stats"]["adaptive_nodes"] for p in pieces),
            "refinement_levels": max(p["stats"]["refinement_levels"] for p in pieces),
            "cutoff_weeks": max(p["stats"]["cutoff_weeks"] for p in pieces),
            "maximum_queue": max(p["stats"]["maximum_queue"] for p in pieces),
            "max_error_bound": float(answer["error_bound"].max()),
            "seconds": time.perf_counter() - started,
        }
        return answer

    order = np.argsort(tau, kind="stable")
    tau = tau[order]
    coeff = (
        delta.reshape(count, len(tau), 7)
        .transpose(0, 2, 1)
        .reshape(count * 7, len(tau))[:, order]
    )
    mass = np.abs(coeff).sum(1)
    total_mass = mass.reshape(count, 7).sum(1)
    global_guard = _EPS_GUARD * np.maximum(total_mass, np.finfo(float).tiny)
    if np.any(global_guard > atol * 0.05):
        raise ValueError("Tolerance is below floating-point error budget")

    budget = atol * 3.2 / 7.0
    pad = _EPS_GUARD * mass
    lower, upper = _signed_integral_bounds(coeff, pad)
    upper_triangle = mass + pad
    same_sign = np.all(coeff >= 0, axis=1) | np.all(coeff <= 0, axis=1)
    if np.all(tau == tau[0]):
        certified = np.ones(len(coeff), dtype=bool)
    else:
        certified = same_sign | _abel_sign(coeff, tau)
    upper = np.where(certified, upper, upper_triangle)
    active = np.flatnonzero((upper - lower) > budget)

    nodes = 0
    levels = 0
    cutoff = 0
    max_queue = len(active)
    if len(active):
        with np.errstate(divide="ignore"):
            log_rates = np.log1p(-tau)
        needed = math.log(
            (budget * 0.0625) / max(float(mass[active].max()), np.finfo(float).tiny)
        ) / float(log_rates[0])
        exponent = max(0, int(math.ceil(math.log2(max(1.0, needed)))))
        if exponent >= 62:
            raise ValueError("Hazard requires integer blocks beyond 62-bit range")
        cutoff = 1 << exponent
        power, exponents = _powers(log_rates, np.full(len(active), cutoff))
        remaining = coeff[active] * power
        residual_pad = _EPS_GUARD * np.sum(
            np.abs(remaining) * (3 + np.minimum(np.abs(exponents), 750)), axis=1
        )
        residual_lo, residual_hi = _signed_integral_bounds(remaining, residual_pad)
        residual_sign = _abel_sign(
            remaining,
            tau,
            _EPS_GUARD * np.abs(remaining) * (3 + np.minimum(np.abs(exponents), 750)),
        )
        residual_hi = np.where(
            residual_sign, residual_hi, np.abs(remaining).sum(1) + residual_pad
        )
        closed_lo = np.zeros(len(coeff))
        closed_hi = np.zeros(len(coeff))
        closed_lo[active] = residual_lo
        closed_hi[active] = residual_hi
        index = active
        left = np.zeros(len(index), dtype=np.int64)
        width = np.full(len(index), cutoff, dtype=np.int64)

        while len(index):
            nodes += len(index)
            levels += 1
            max_queue = max(max_queue, len(index))
            if nodes > max_nodes:
                raise RuntimeError("Tail refinement exceeded max_nodes")
            lo, hi, sign = _block_bounds(coeff[index], tau, log_rates, left, width)
            if sign.any():
                closed_lo += np.bincount(
                    index[sign], weights=lo[sign], minlength=len(coeff)
                )
                closed_hi += np.bincount(
                    index[sign], weights=hi[sign], minlength=len(coeff)
                )
            unresolved = ~sign
            if not unresolved.any():
                lower[active], upper[active] = closed_lo[active], closed_hi[active]
                break
            idx, lft, wid = index[unresolved], left[unresolved], width[unresolved]
            ulo, uhi = lo[unresolved], hi[unresolved]
            trial_lo = closed_lo + np.bincount(idx, weights=ulo, minlength=len(coeff))
            trial_hi = closed_hi + np.bincount(idx, weights=uhi, minlength=len(coeff))
            keep = (trial_hi - trial_lo)[idx] > budget
            finalized = np.setdiff1d(active, np.unique(idx[keep]), assume_unique=True)
            lower[finalized], upper[finalized] = trial_lo[finalized], trial_hi[finalized]
            if not keep.any():
                break
            idx, lft, half = idx[keep], lft[keep], wid[keep] // 2
            active = np.unique(idx)
            index = np.repeat(idx, 2)
            left = np.column_stack((lft, lft + half)).reshape(-1)
            width = np.repeat(half, 2)

    lower = lower.reshape(count, 7).sum(1) / 2.0
    upper = upper.reshape(count, 7).sum(1) / 2.0
    lower, upper = _outward(
        np.maximum(0.0, lower - global_guard), upper + global_guard
    )
    midpoint = (lower + upper) / 2.0
    error = (upper - lower) / 2.0
    if not np.isfinite(midpoint).all() or np.any(error > atol):
        raise RuntimeError("Requested TV error tolerance was not certified")
    return {
        "tv": midpoint.reshape(leading),
        "error_bound": error.reshape(leading),
        "lower": lower.reshape(leading),
        "upper": upper.reshape(leading),
        "stats": {
            "forecasts": count,
            "checkpoints": len(tau),
            "adaptive_nodes": nodes,
            "refinement_levels": levels,
            "cutoff_weeks": cutoff,
            "maximum_queue": max_queue,
            "max_error_bound": float(error.max()) if error.size else 0.0,
            "seconds": time.perf_counter() - started,
        },
    }


def ensemble_tv(
    a: np.ndarray,
    b: np.ndarray,
    taus: np.ndarray,
    weights_a: Optional[np.ndarray] = None,
    weights_b: Optional[np.ndarray] = None,
    atol: float = 1e-9,
    max_nodes: int = 2_000_000,
    batch_size: int = 512,
) -> Dict[str, Any]:
    """Compute exact certified Total Variation between 427-component mixture PMFs.

    Components 0..419 represent explicit day probabilities on days 1..420;
    components 420..426 store the seven total geometric-tail channel masses.
    """
    a_arr, b_arr = np.broadcast_arrays(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    )
    if (
        a_arr.ndim < 2
        or a_arr.shape[-1] != 427
        or not np.isfinite(a_arr).all()
        or not np.isfinite(b_arr).all()
        or min(float(a_arr.min()), float(b_arr.min())) < 0.0
    ):
        raise ValueError("Expected finite nonnegative (..., K, 427) checkpoint PMFs")
    if (
        max(
            float(np.abs(a_arr.sum(-1) - 1.0).max()),
            float(np.abs(b_arr.sum(-1) - 1.0).max()),
        )
        > 1e-11
    ):
        raise ValueError("Checkpoint PMFs must sum to 1")

    shape, k = a_arr.shape[:-1], a_arr.shape[-2]

    def _resolve_weights(value: Optional[np.ndarray]) -> np.ndarray:
        base = (
            np.full(k, 1.0 / k)
            if value is None
            else np.asarray(value, dtype=float)
        )
        result = np.broadcast_to(base, shape)
        if (
            not np.isfinite(result).all()
            or (result < 0).any()
            or np.abs(result.sum(-1) - 1.0).max() > 1e-13
        ):
            raise ValueError("Mixture weights must be finite, nonnegative, and sum to 1")
        return result

    wa = _resolve_weights(weights_a)
    wb = wa if weights_b is None else _resolve_weights(weights_b)
    weighted_a = wa[..., None] * a_arr
    weighted_b = wb[..., None] * b_arr
    difference = weighted_a - weighted_b
    finite = 0.5 * np.abs(difference[..., :420].sum(axis=-2)).sum(axis=-1)
    finite_guard = _EPS_GUARD * (
        weighted_a.sum(axis=(-2, -1)) + weighted_b.sum(axis=(-2, -1))
    )
    if np.any(finite_guard >= atol * 0.1):
        raise ValueError("Requested tolerance is below finite-mixture precision")

    result = mixture_tail_tv(
        difference[..., 420:],
        taus,
        atol=atol * 0.9,
        max_nodes=max_nodes,
        batch_size=batch_size,
    )
    lower, upper = _outward(
        np.maximum(0.0, finite + result["lower"] - finite_guard),
        finite + result["upper"] + finite_guard,
    )
    midpoint = (lower + upper) / 2.0
    error = (upper - lower) / 2.0
    if np.any(error > atol):
        raise RuntimeError("Full-mixture TV tolerance was not certified")
    result.update(
        tv=midpoint,
        error_bound=error,
        lower=lower,
        upper=upper,
        finite_day_tv=finite,
        tail_tv=result["tv"],
    )
    return result


def single_checkpoint_427_tv(p_427: np.ndarray, q_427: np.ndarray) -> np.ndarray:
    """Exact TV between two PMFs sharing the same geometric tail hazard."""
    p = np.asarray(p_427, dtype=np.float64)
    q = np.asarray(q_427, dtype=np.float64)
    if p.shape != q.shape or p.shape[-1] != 427:
        raise ValueError("Expected matching (..., 427) PMF arrays")
    return 0.5 * np.abs(p - q).sum(axis=-1)


def tta_mixture_log_scores(shift_log_scores: np.ndarray) -> np.ndarray:
    """Compute seven-origin equal-weight probability mixture log scores.

    Given log-probabilities L[..., 7] across shifts m = 0..6, returns
    log((1/7) * sum_{m=0}^6 exp(L[..., m])), which is strictly shift-invariant.
    """
    arr = np.asarray(shift_log_scores, dtype=np.float64)
    if arr.shape[-1] != 7:
        raise ValueError("Expected trailing axis of length 7 for shifts 0..6")
    top = np.max(arr, axis=-1, keepdims=True)
    finite_top = np.isfinite(top)
    safe_top = np.where(finite_top, top, 0.0)
    with np.errstate(divide="ignore"):
        out = safe_top[..., 0] + np.log(np.mean(np.exp(arr - safe_top), axis=-1))
    return np.where(finite_top[..., 0], out, -np.inf)
