"""Relative-phase alignment feature (X-align) and boosted discrete-time hazard models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .cohort import simple_numeric_categorical_design


COMMON8_HISTORY_COLUMNS: Tuple[str, ...] = (
    "order_index",
    "log_prev_gap",
    "log_mean_prior_gap",
    "prior_count",
    "prev_gap_is_week_multiple",
    "share_prior_week_multiple",
    "log1p_time_since_start",
)

COMMON8_FEATURES: Tuple[str, ...] = ("start_is_imaging",) + COMMON8_HISTORY_COLUMNS

DEFAULT_XGB_PARAMS: Dict[str, float] = {
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 20.0,
    "reg_lambda": 10.0,
    "num_boost_round": 400,
}

HORIZON_DAYS = 120
FEATURE_SETS = {1: (0,), 2: (0, 2), 4: (0, 1, 2, 3)}


@dataclass(frozen=True)
class RowBundle:
    """Sorted interval-level arrays for person-period discrete-time hazard models."""

    record_id: np.ndarray
    institution: np.ndarray
    start: np.ndarray
    stop: np.ndarray
    duration: np.ndarray
    event: np.ndarray
    patient_index: np.ndarray
    patient_ids: Tuple[str, ...]
    order_in_patient: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.duration.shape[0])

    @property
    def patients(self) -> int:
        return len(self.patient_ids)


def _sort_frame_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["record_id", "interval_start", "interval_stop"]
    present = [c for c in cols if c in frame.columns]
    return frame.sort_values(present, kind="mergesort").reset_index(drop=True)


def build_rows(frame: pd.DataFrame) -> RowBundle:
    ordered = _sort_frame_canonical(frame)
    ids = ordered["record_id"].astype(str).to_numpy()
    unique_ids = list(dict.fromkeys(ids.tolist()))
    lookup = {value: pos for pos, value in enumerate(unique_ids)}
    patient_index = np.array([lookup[v] for v in ids], dtype=int)
    if "order_in_patient" in ordered.columns:
        order_col = ordered["order_in_patient"].to_numpy(dtype=int)
    else:
        order_col = ordered.groupby("record_id", sort=False).cumcount().to_numpy(dtype=int)
    return RowBundle(
        record_id=ids,
        institution=ordered["institution"].astype(str).to_numpy(),
        start=ordered["interval_start"].to_numpy(dtype=float),
        stop=ordered["interval_stop"].to_numpy(dtype=float),
        duration=ordered["exposure_days"].to_numpy(dtype=float),
        event=ordered["event"].to_numpy(dtype=int),
        patient_index=patient_index,
        patient_ids=tuple(unique_ids),
        order_in_patient=order_col,
    )


def common8_history_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly-prior gap summary predictors shared across NSCLC and CRC."""
    work = frame.reset_index(drop=True)
    out = {name: np.zeros(len(work), dtype=float) for name in COMMON8_HISTORY_COLUMNS}
    sort_cols = [c for c in ("record_id", "interval_start", "interval_stop") if c in work.columns]
    order = work.sort_values(sort_cols, kind="mergesort").index.to_numpy()
    gaps = work["exposure_days"].to_numpy(dtype=float)
    ids = work["record_id"].astype(str).to_numpy()
    prior: List[float] = []
    elapsed = 0.0
    current = None
    for row in order:
        if ids[row] != current:
            current, prior, elapsed = ids[row], [], 0.0
        n = len(prior)
        arr = np.asarray(prior, dtype=float) if n else np.zeros(0)
        prev = prior[-1] if n else 0.0
        week = (np.mod(np.round(arr).astype(int), 7) == 0) if n else np.zeros(0, bool)
        out["order_index"][row] = n / 10.0
        out["log_prev_gap"][row] = np.log(max(prev, 1.0)) if n else 0.0
        out["log_mean_prior_gap"][row] = np.log(max(arr.mean(), 1.0)) if n else 0.0
        out["prior_count"][row] = n / 10.0
        out["prev_gap_is_week_multiple"][row] = (
            1.0 if (n and int(round(prev)) % 7 == 0) else 0.0
        )
        out["share_prior_week_multiple"][row] = float(week.mean()) if n else 0.0
        out["log1p_time_since_start"][row] = np.log1p(elapsed)
        prior.append(float(gaps[row]))
        elapsed += float(gaps[row])
    return pd.DataFrame(out, index=frame.index)


def common8_design(
    dev_frame: pd.DataFrame, ev_frame: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    """Construct standardized 8-column history design matrices aligned to canonical row order."""
    dev_sorted = _sort_frame_canonical(dev_frame)
    ev_sorted = _sort_frame_canonical(ev_frame)
    dev = pd.concat([dev_sorted, common8_history_columns(dev_sorted)], axis=1)
    ev = pd.concat([ev_sorted, common8_history_columns(ev_sorted)], axis=1)
    X_dev, X_ev, names = simple_numeric_categorical_design(
        dev, ev, list(COMMON8_FEATURES), []
    )
    if X_dev.shape[1] != 8:
        raise AssertionError(f"Expected 8 common history features, got {X_dev.shape[1]}")
    return X_dev, X_ev, names


def phase_counts(
    rows: RowBundle, weight_mode: str = "sqrt", half_life: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute modulo-7 opening-day counts from the current start and strictly prior visits."""
    n = rows.rows
    counts = np.zeros((n, 7), dtype=float)
    start_mod = np.mod(np.round(rows.start).astype(int), 7)
    by_patient: Dict[int, List[int]] = {}
    order = np.lexsort((np.asarray(rows.order_in_patient), np.asarray(rows.patient_index)))
    decay = 0.5 ** (1.0 / float(half_life)) if weight_mode == "recency" else 1.0

    for idx in order:
        pat = int(rows.patient_index[idx])
        prior = by_patient.setdefault(pat, [])
        if weight_mode == "recency":
            for age, r in enumerate(reversed(prior), start=1):
                counts[idx, r] += decay ** age
        else:
            for r in prior:
                counts[idx, r] += 1.0
        counts[idx, start_mod[idx]] += 1.0
        prior.append(int(start_mod[idx]))

    if weight_mode in ("raw", "recency"):
        return counts, start_mod
    if weight_mode == "sqrt":
        total = counts.sum(axis=1, keepdims=True)
        return counts / np.sqrt(np.maximum(total, 1.0)), start_mod
    raise ValueError(f"Unknown phase weight_mode: {weight_mode}")


def phase_posterior(
    rows: RowBundle,
    kappa: float,
    weight_mode: str = "sqrt",
    half_life: float = 5.0,
    rotate_fixed: int = 0,
) -> np.ndarray:
    """Compute patient-specific modulo-7 booking-phase weights (Eq. 6)."""
    counts, _ = phase_counts(rows, weight_mode=weight_mode, half_life=half_life)
    z = float(kappa) * counts
    z = z - z.max(axis=1, keepdims=True)
    w = np.exp(z)
    w = w / w.sum(axis=1, keepdims=True)
    if rotate_fixed:
        w = np.roll(w, int(rotate_fixed) % 7, axis=1)
    return w


def phase_alignment_features(
    rows: RowBundle, posterior: np.ndarray, day: np.ndarray, position: int
) -> np.ndarray:
    """Evaluate alignment features for candidate elapsed days u = 1..span (Eq. 7)."""
    start = int(np.round(rows.start[position]))
    residue = np.mod(start + day.astype(int), 7)
    w = posterior[position]
    align = w[residue]
    align_pm1 = w[(residue - 1) % 7] + align + w[(residue + 1) % 7]
    conc = np.full(day.shape[0], float(w.max()))
    offsets = np.arange(7)
    diff = np.abs(offsets[None, :] - residue[:, None])
    circ = np.minimum(diff, 7 - diff)
    dist = (circ * w[None, :]).sum(axis=1)
    return np.column_stack([align, align_pm1, conc, dist])


def estimate_phase_kappa(
    rows: RowBundle,
    X: np.ndarray,
    *,
    smooth: float = 10.0,
    ridge: float = 0.1,
    weight_mode: str = "sqrt",
    half_life: float = 5.0,
    iterations: int = 150,
) -> float:
    """Estimate phase concentration kappa on development rows via L-BFGS."""
    X = np.asarray(X, dtype=float)
    n, pdim = X.shape
    counts_np, start_mod_np = phase_counts(rows, weight_mode=weight_mode, half_life=half_life)
    d = np.round(rows.duration).clip(1.0).astype(int)
    din = np.minimum(d, HORIZON_DAYS)
    ev = torch.tensor(rows.event.astype(float), dtype=torch.float64)
    inside_event = torch.tensor(
        ((rows.event == 1) & (d <= HORIZON_DAYS)).astype(float), dtype=torch.float64
    )
    didx = torch.tensor(din - 1, dtype=torch.long)
    Xt = torch.tensor(X, dtype=torch.float64)
    counts = torch.tensor(counts_np, dtype=torch.float64)
    start_mod = torch.tensor(start_mod_np, dtype=torch.long)
    u_np = np.arange(1, HORIZON_DAYS + 1)

    b = torch.full((HORIZON_DAYS,), -3.0, dtype=torch.float64, requires_grad=True)
    beta = torch.zeros(pdim, dtype=torch.float64, requires_grad=True)
    phi = torch.zeros(7, dtype=torch.float64, requires_grad=True)
    kappa = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    prof_index = torch.tensor(
        np.stack([(u_np + c) % 7 for c in range(7)]), dtype=torch.long
    )
    delta_of_c = torch.stack(
        [torch.remainder(start_mod - c, 7) for c in range(7)], dim=1
    )

    beyond = rows.duration > float(HORIZON_DAYS)
    tail_rate = 0.01
    if int(beyond.sum()) > 5:
        excess = rows.duration[beyond] - float(HORIZON_DAYS)
        observed = rows.event[beyond].astype(float)
        tail_rate = float(np.clip(observed.sum() / max(excess.sum(), 1.0), 1e-4, 1.0))

    def negloglik() -> torch.Tensor:
        phi_c = phi - phi.mean()
        eta_x = torch.clamp(Xt.mv(beta), -15.0, 15.0)
        w = torch.softmax(kappa * counts, dim=1)
        wc = torch.gather(w, 1, delta_of_c)
        lik_ev = torch.zeros(n, dtype=torch.float64)
        surv_at = torch.zeros(n, dtype=torch.float64)
        for c in range(7):
            logits = (b + phi_c[prof_index[c]]).reshape(1, -1) + eta_x.reshape(-1, 1)
            h = torch.sigmoid(torch.clamp(logits, -30.0, 30.0)).clamp(1e-9, 1.0 - 1e-9)
            log_s = torch.cumsum(torch.log1p(-h), dim=1)
            s_at = torch.exp(log_s[torch.arange(n), didx])
            s_prev = torch.where(
                didx > 0,
                torch.exp(log_s[torch.arange(n), torch.clamp(didx - 1, min=0)]),
                torch.ones(n, dtype=torch.float64),
            )
            h_at = h[torch.arange(n), didx]
            lik_ev = lik_ev + wc[:, c] * (s_prev * h_at)
            surv_at = surv_at + wc[:, c] * s_at
        over = torch.tensor(
            np.maximum(rows.duration - float(HORIZON_DAYS), 0.0), dtype=torch.float64
        )
        tail_factor = torch.exp(-tail_rate * over)
        tail_prev_factor = torch.exp(-tail_rate * torch.clamp(over - 1.0, min=0.0))
        tail_pmf = surv_at * tail_prev_factor * (-math.expm1(-tail_rate))
        p_event = inside_event * lik_ev + (ev - inside_event) * tail_pmf
        p_cens = surv_at * tail_factor
        ll = ev * torch.log(torch.clamp(p_event, min=1e-300)) + (
            1.0 - ev
        ) * torch.log(torch.clamp(p_cens, min=1e-300))
        diff = b[1:] - b[:-1]
        penalty = smooth * diff.pow(2).sum() + ridge * beta.pow(2).sum() + phi_c.pow(2).sum()
        return -(ll.sum()) + penalty

    optimizer = torch.optim.LBFGS(
        [b, beta, phi, kappa],
        max_iter=int(iterations),
        tolerance_grad=1e-7,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = negloglik()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(kappa.detach().cpu().item())


def expand_person_period(
    rows: RowBundle,
    X: np.ndarray,
    posterior: Optional[np.ndarray],
    keep_features: int = 1,
    days: int = HORIZON_DAYS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand intervals into daily person-period rows for u = 1..min(duration, days)."""
    duration = np.clip(np.round(rows.duration).astype(int), 1, None)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for pos in range(rows.rows):
        span = int(min(duration[pos], days))
        if span < 1:
            continue
        day = np.arange(1, span + 1, dtype=float)
        base = np.hstack(
            (
                np.tile(X[pos], (span, 1)),
                day.reshape(-1, 1),
                (day % 7.0).reshape(-1, 1),
                np.log(day).reshape(-1, 1),
            )
        )
        if posterior is not None:
            cols = phase_alignment_features(rows, posterior, day, pos)
            base = np.hstack((base, cols[:, list(FEATURE_SETS[keep_features])]))
        target = np.zeros(span, dtype=float)
        if rows.event[pos] == 1 and duration[pos] <= days:
            target[-1] = 1.0
        xs.append(base)
        ys.append(target)
    return np.vstack(xs), np.concatenate(ys)


def fit_xgb_discrete_hazard(
    rows: RowBundle,
    X: np.ndarray,
    posterior: Optional[np.ndarray],
    *,
    keep_features: int = 1,
    seed: int = 0,
    nthread: int = 2,
    xgb_params: Optional[Dict[str, float]] = None,
    days: int = HORIZON_DAYS,
):
    """Fit XGBoost discrete-time daily hazard model and constant exponential tail rate."""
    import xgboost as xgb

    cfg = dict(DEFAULT_XGB_PARAMS)
    if xgb_params:
        cfg.update(xgb_params)

    design, target = expand_person_period(
        rows, X, posterior, keep_features=keep_features, days=days
    )
    matrix = xgb.DMatrix(design, label=target)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "learning_rate": float(cfg["eta"]),
        "max_depth": int(cfg["max_depth"]),
        "min_child_weight": float(cfg["min_child_weight"]),
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": float(cfg.get("reg_lambda", 10.0)),
        "nthread": int(nthread),
        "seed": int(seed),
    }
    booster = xgb.train(params, matrix, num_boost_round=int(cfg["num_boost_round"]))
    beyond = rows.duration > float(days)
    tail_rate = 0.01
    if int(beyond.sum()) > 5:
        excess = rows.duration[beyond] - float(days)
        observed = rows.event[beyond].astype(float)
        tail_rate = float(np.clip(observed.sum() / max(excess.sum(), 1.0), 1e-4, 1.0))
    return booster, tail_rate


def score_xgb_rows(
    booster,
    tail_rate: float,
    rows: RowBundle,
    X: np.ndarray,
    posterior: Optional[np.ndarray],
    *,
    keep_features: int = 1,
    days: int = HORIZON_DAYS,
) -> np.ndarray:
    """Evaluate per-interval daily-scale log-likelihood for completed and censored rows."""
    import xgboost as xgb

    n = X.shape[0]
    day_grid = np.arange(1, days + 1, dtype=float)
    repeated = np.repeat(np.asarray(X, dtype=float), days, axis=0)
    day_column = np.tile(day_grid, n).reshape(-1, 1)
    base = np.hstack((repeated, day_column, day_column % 7.0, np.log(day_column)))

    if posterior is not None:
        start = np.round(rows.start).astype(int).reshape(-1, 1)
        residue = np.mod(start + day_grid.astype(int).reshape(1, -1), 7)
        sel = np.arange(n).reshape(-1, 1)
        align = posterior[sel, residue]
        align_pm1 = (
            posterior[sel, (residue - 1) % 7]
            + align
            + posterior[sel, (residue + 1) % 7]
        )
        conc = np.tile(posterior.max(axis=1).reshape(-1, 1), (1, days))
        offsets = np.arange(7).reshape(1, 1, 7)
        diff = np.abs(offsets - residue[:, :, None])
        circ = np.minimum(diff, 7 - diff)
        dist = (circ * posterior[:, None, :]).sum(axis=2)
        cols = np.column_stack(
            [align.reshape(-1), align_pm1.reshape(-1), conc.reshape(-1), dist.reshape(-1)]
        )
        base = np.hstack((base, cols[:, list(FEATURE_SETS[keep_features])]))

    raw = np.asarray(
        booster.predict(xgb.DMatrix(base), output_margin=True), dtype=np.float64
    ).reshape(n, days)
    clipped = np.clip(raw, -30.0, 30.0)
    log_h = -np.logaddexp(0.0, -clipped)
    log_1m_h = -np.logaddexp(0.0, clipped)
    log_surv = np.cumsum(log_1m_h, axis=1)

    duration = np.clip(np.round(rows.duration).astype(int), 1, None)
    idx = np.clip(duration, 1, days) - 1
    row_arange = np.arange(n)
    ls_at = log_surv[row_arange, idx]
    ls_prev = np.where(
        idx > 0, log_surv[row_arange, np.maximum(idx - 1, 0)], 0.0
    )
    lh_at = log_h[row_arange, idx]

    over = np.maximum(duration - days, 0).astype(np.float64)
    inside = duration <= days
    ls_horizon = log_surv[:, days - 1]
    log_tail_event_prob = math.log(max(-math.expm1(-float(tail_rate)), 1e-300))

    log_cens = np.where(inside, ls_at, ls_horizon - float(tail_rate) * over)
    log_event = np.where(
        inside,
        ls_prev + lh_at,
        ls_horizon - float(tail_rate) * np.maximum(over - 1.0, 0.0) + log_tail_event_prob,
    )
    return np.where(rows.event == 1, log_event, log_cens)
