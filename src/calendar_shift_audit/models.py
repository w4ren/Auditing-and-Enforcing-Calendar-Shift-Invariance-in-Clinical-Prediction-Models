"""Matched 4,356-parameter neural recurrent-event models and invariance checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


CALENDAR_BLIND = "Calendar-blind"
ABSOLUTE_PHASE = "Absolute-phase"
RELATIVE_PHASE = "Relative-phase"
FLEXIBLE_ELAPSED = "Flexible-elapsed"

ARM_ALIASES: Dict[str, str] = {
    "Calendar-blind": CALENDAR_BLIND,
    "CM-blind": CALENDAR_BLIND,
    "Absolute-phase": ABSOLUTE_PHASE,
    "CM-absolute": ABSOLUTE_PHASE,
    "Absolute-weekday": ABSOLUTE_PHASE,
    "Relative-phase": RELATIVE_PHASE,
    "CM-equivariant": RELATIVE_PHASE,
    "Relative-weekday": RELATIVE_PHASE,
    "Flexible-elapsed": FLEXIBLE_ELAPSED,
}

PRIMARY_NEURAL_ARMS: Tuple[str, ...] = (
    CALENDAR_BLIND,
    ABSOLUTE_PHASE,
    RELATIVE_PHASE,
)
ALL_NEURAL_ARMS: Tuple[str, ...] = (
    CALENDAR_BLIND,
    ABSOLUTE_PHASE,
    RELATIVE_PHASE,
    FLEXIBLE_ELAPSED,
)

SEQUENCE_HISTORY_NAMES: Tuple[str, ...] = (
    "order_index_scaled",
    "log1p_order_index",
    "is_first_interval",
    "log_prev_gap",
    "log_prev_gap2",
    "log_mean_prior_gap",
    "sd_prior_gap_scaled",
    "prev_gap_is_week_multiple",
    "share_prior_gaps_week_multiple",
    "log1p_time_since_start",
    "n_prior_intervals_scaled",
)


def canonical_arm_name(arm: str) -> str:
    if arm not in ARM_ALIASES:
        raise ValueError(f"Unknown neural model arm: {arm}")
    return ARM_ALIASES[arm]


@dataclass
class SeqBundle:
    """Patient-major padded sequence bundle with separate modulo-7 opening residues."""

    patient_ids: Tuple[str, ...]
    institution: np.ndarray
    gap: np.ndarray
    event: np.ndarray
    mask: np.ndarray
    resid: np.ndarray
    feat: np.ndarray
    feature_names: Tuple[str, ...]
    row_index: np.ndarray

    @property
    def patients(self) -> int:
        return len(self.patient_ids)

    @property
    def max_intervals(self) -> int:
        return int(self.gap.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.feat.shape[2])

    @property
    def rows(self) -> int:
        return int(self.mask.sum())


def _sequence_history(gap: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute 11 strictly-prior gap summary features without future leakage."""
    n_patients, max_j = gap.shape
    out = np.zeros((n_patients, max_j, len(SEQUENCE_HISTORY_NAMES)), dtype=float)
    for p in range(n_patients):
        prior: List[float] = []
        elapsed = 0.0
        for j in range(max_j):
            if mask[p, j] <= 0:
                break
            n = len(prior)
            arr = np.asarray(prior, dtype=float) if n else np.zeros(0)
            prev = prior[-1] if n >= 1 else 0.0
            prev2 = prior[-2] if n >= 2 else prev
            week = (np.mod(np.round(arr).astype(int), 7) == 0) if n else np.zeros(0, bool)
            out[p, j] = (
                j / 10.0,
                np.log1p(j),
                1.0 if j == 0 else 0.0,
                np.log(max(prev, 1.0)) if n else 0.0,
                np.log(max(prev2, 1.0)) if n else 0.0,
                np.log(max(arr.mean(), 1.0)) if n else 0.0,
                (arr.std() / 10.0) if n >= 2 else 0.0,
                1.0 if (n and int(round(prev)) % 7 == 0) else 0.0,
                float(week.mean()) if n else 0.0,
                np.log1p(elapsed),
                n / 10.0,
            )
            prior.append(float(gap[p, j]))
            elapsed += float(gap[p, j])
    return out


def build_sequences(
    frame: pd.DataFrame, X: np.ndarray, feature_names: Sequence[str]
) -> SeqBundle:
    """Convert a sorted interval frame and covariate matrix into padded patient sequences."""
    ids = frame["record_id"].astype(str).to_numpy()
    unique = list(dict.fromkeys(ids.tolist()))
    lookup = {v: i for i, v in enumerate(unique)}
    pidx = np.array([lookup[v] for v in ids], dtype=int)
    counts = np.bincount(pidx, minlength=len(unique))
    max_j = int(counts.max())
    n_patients = len(unique)

    gap = np.zeros((n_patients, max_j), dtype=float)
    event = np.zeros((n_patients, max_j), dtype=float)
    mask = np.zeros((n_patients, max_j), dtype=float)
    resid = np.zeros((n_patients, max_j), dtype=int)
    rowix = np.full((n_patients, max_j), -1, dtype=int)
    cov = np.zeros((n_patients, max_j, X.shape[1]), dtype=float)
    inst = np.empty(n_patients, dtype=object)

    start = frame["interval_start"].to_numpy(dtype=float)
    dur = np.clip(np.round(frame["exposure_days"].to_numpy(dtype=float)), 1.0, None)
    ev = frame["event"].to_numpy(dtype=int)
    institution = frame["institution"].astype(str).to_numpy()

    seen = np.zeros(n_patients, dtype=int)
    for row in range(len(frame)):
        p = int(pidx[row])
        j = int(seen[p])
        seen[p] = j + 1
        gap[p, j] = dur[row]
        event[p, j] = ev[row]
        mask[p, j] = 1.0
        resid[p, j] = int(np.mod(np.round(start[row]).astype(np.int64), 7))
        rowix[p, j] = row
        cov[p, j] = X[row]
        inst[p] = institution[row]

    hist = _sequence_history(gap, mask)
    feat = np.concatenate([cov, hist], axis=2)
    names = tuple(list(feature_names) + list(SEQUENCE_HISTORY_NAMES))
    return SeqBundle(
        patient_ids=tuple(unique),
        institution=inst,
        gap=gap,
        event=event,
        mask=mask,
        resid=resid,
        feat=feat,
        feature_names=names,
        row_index=rowix,
    )


def assert_shift_invariant_bundle(
    bundle_a: SeqBundle, bundle_b: SeqBundle, tol: float = 0.0
) -> None:
    """Verify that gaps, events, masks, and shift-invariant features are unchanged."""
    for field in ("gap", "event", "mask", "feat"):
        a = getattr(bundle_a, field)
        b = getattr(bundle_b, field)
        if a.shape != b.shape or not np.allclose(a, b, atol=tol, rtol=0.0):
            raise AssertionError(f"Sequence field '{field}' changed under a pure calendar shift")


def _mlp(sizes: Sequence[int], out: int, dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(a, b))
        layers.append(nn.Tanh())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(sizes[-1], out))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class MatchedNeuralConfig:
    arm: str
    hidden: int = 64
    embed: int = 16
    weeks: int = 60
    dropout: float = 0.0
    drift_temp: float = 1.0
    l2: float = 1e-4
    seed: int = 0

    def __post_init__(self) -> None:
        canonical = canonical_arm_name(self.arm)
        object.__setattr__(self, "arm", canonical)
        if self.hidden <= 0 or self.embed < 7 or self.weeks < 7:
            raise ValueError(f"Invalid neural model dimensions: {asdict(self)}")


class PhaseFreeWeekAdapter(nn.Module):
    """History-conditioned elapsed-week hazard adapter with zero phase inputs."""

    def __init__(self, embed: int, width: int, weeks: int):
        super().__init__()
        self.embed = int(embed)
        self.width = int(width)
        self.weeks = int(weeks)
        if self.width <= 0:
            raise ValueError("Adapter width must be positive")
        self.adapter_in = nn.Linear(self.embed, self.width)
        self.adapter_out = nn.Linear(self.width, 1)
        self.week_basis_free = nn.Parameter(
            torch.tensor(
                [-0.030, -0.020, -0.010, 0.010, 0.020, 0.030],
                dtype=torch.get_default_dtype(),
            )
        )
        week = torch.arange(self.weeks, dtype=torch.long)
        group = torch.div(7 * week, self.weeks, rounding_mode="floor")
        self.register_buffer("week_group", torch.clamp(group, max=6), persistent=True)

    def full_basis(self) -> torch.Tensor:
        seventh = -self.week_basis_free.sum().reshape(1)
        return torch.cat((self.week_basis_free, seventh), dim=0)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        amplitude = self.adapter_out(torch.tanh(self.adapter_in(embed)))
        selected = self.full_basis()[self.week_group]
        return amplitude * selected.reshape(*([1] * (embed.dim() - 1)), self.weeks)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


class MatchedNeuralSurvivalModel(nn.Module):
    """Parameter-matched recurrent-event model implementing Calendar-blind,
    Absolute-phase, and Relative-phase (4,356 active parameters at F=19, h=64, e=16, K=60).
    """

    eval_rotation: int = 0

    def __init__(self, n_features: int, config: MatchedNeuralConfig):
        super().__init__()
        self.n_features = int(n_features)
        self.cfg = config
        self.arm = config.arm
        torch.manual_seed(int(config.seed))

        h, e, K = int(config.hidden), int(config.embed), int(config.weeks)
        self.encoder = _mlp([self.n_features, h], e, dropout=config.dropout)
        self.pi_head: Optional[nn.Linear] = nn.Linear(e, 7)
        self.week_base = nn.Parameter(torch.full((K,), -1.0))
        self.class_shift: Optional[nn.Parameter] = nn.Parameter(torch.zeros(7))
        self.week_load = nn.Parameter(torch.zeros(K, e))
        self.tail_logit = nn.Parameter(torch.tensor(-1.0))
        self.init_logits: Optional[nn.Parameter] = nn.Parameter(torch.zeros(7))

        if self.arm == ABSOLUTE_PHASE:
            self.drift_head: Optional[nn.Linear] = nn.Linear(e, 49)
            self.abs_class: Optional[nn.Parameter] = nn.Parameter(torch.zeros(7, 7))
        else:
            self.drift_head = nn.Linear(e, 7)
            self.register_parameter("abs_class", None)

        if self.arm == CALENDAR_BLIND:
            self.pi_head = None
            self.class_shift = None
            self.init_logits = None
            self.drift_head = None
            self.abs_class = None

        width: Optional[int]
        if self.arm == CALENDAR_BLIND:
            width = 56
        elif self.arm == RELATIVE_PHASE:
            width = 42
        else:
            width = None

        self.capacity_adapter: Optional[PhaseFreeWeekAdapter] = (
            PhaseFreeWeekAdapter(e, width, K) if width is not None else None
        )

        expected = expected_parameter_count(self.n_features, config)
        actual = self.n_parameters()
        if actual != expected:
            raise AssertionError(
                f"{self.arm} parameter count {actual} != expected {expected}"
            )

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def _tail(self) -> Tuple[torch.Tensor, torch.Tensor]:
        logit = torch.clamp(self.tail_logit, -18.0, 18.0)
        return (
            torch.nn.functional.logsigmoid(logit),
            torch.nn.functional.logsigmoid(-logit),
        )

    def _shared_week_logits(self, embed: torch.Tensor) -> torch.Tensor:
        K = int(self.cfg.weeks)
        shared = self.week_base.reshape(*([1] * (embed.dim() - 1)), K) + embed.matmul(
            self.week_load.t()
        )
        if self.capacity_adapter is not None:
            shared = shared + self.capacity_adapter(embed)
        return shared

    def _week_log_terms(
        self, embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self._shared_week_logits(embed)
        if self.arm == CALENDAR_BLIND or self.class_shift is None:
            logits = shared.unsqueeze(-2).expand(
                *shared.shape[:-1], 7, shared.shape[-1]
            )
        else:
            logits = shared.unsqueeze(-2) + self.class_shift.reshape(
                *([1] * (embed.dim() - 1)), 7, 1
            )
        logits = torch.clamp(logits, -18.0, 18.0)
        log_haz = torch.nn.functional.logsigmoid(logits)
        log_one_minus = torch.nn.functional.logsigmoid(-logits)
        zero = torch.zeros_like(log_one_minus[..., :1])
        prefix = torch.cat((zero, torch.cumsum(log_one_minus, dim=-1)), dim=-1)
        return log_haz, prefix

    def _log_pi(
        self, embed: torch.Tensor, resid: Optional[torch.Tensor]
    ) -> torch.Tensor:
        assert self.pi_head is not None
        logits = self.pi_head(embed)
        if self.arm == ABSOLUTE_PHASE and resid is not None and self.abs_class is not None:
            logits = logits + self.abs_class[resid]
        return torch.log_softmax(logits, dim=-1)

    def _blind_observed_log_prob(
        self, embed: torch.Tensor, gap: torch.Tensor, event: torch.Tensor
    ) -> torch.Tensor:
        K = int(self.cfg.weeks)
        log_haz_7, prefix_7 = self._week_log_terms(embed)
        log_haz = log_haz_7[..., 0, :]
        prefix = prefix_7[..., 0, :]
        log_tail_h, log_tail_s = self._tail()
        t = gap.long()
        week = torch.div(t - 1, 7, rounding_mode="floor")
        week_in = torch.clamp(week, 0, K - 1)
        inside = week < K
        event_q = torch.gather(log_haz, -1, week_in.unsqueeze(-1)).squeeze(-1) + torch.gather(
            prefix, -1, week_in.unsqueeze(-1)
        ).squeeze(-1)
        event_q = torch.where(
            inside,
            event_q,
            prefix[..., K]
            + torch.clamp(week - K, min=0).to(prefix.dtype) * log_tail_s
            + log_tail_h,
        )
        log_seven = torch.log(torch.tensor(7.0, dtype=embed.dtype, device=embed.device))
        log_event = event_q - log_seven

        m = torch.arange(7, device=gap.device).reshape(*([1] * gap.dim()), 7)
        first = torch.remainder(m - 1, 7) + 1
        expanded_t = t.unsqueeze(-1)
        k_min = torch.where(
            expanded_t >= first,
            torch.div(expanded_t - first, 7, rounding_mode="floor") + 1,
            torch.zeros_like(expanded_t - first),
        )
        k_in = torch.clamp(k_min, 0, K)
        expanded_prefix = prefix.unsqueeze(-2).expand(*prefix.shape[:-1], 7, K + 1)
        log_qbar = torch.gather(expanded_prefix, -1, k_in.unsqueeze(-1)).squeeze(-1)
        log_qbar = log_qbar + torch.clamp(k_min - K, min=0).to(prefix.dtype) * log_tail_s
        log_censored = torch.logsumexp(log_qbar - log_seven, dim=-1)
        return torch.where(event > 0.5, log_event, log_censored)

    def _censored_log_prob(
        self,
        log_pi: torch.Tensor,
        prefix: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        log_tail_s: torch.Tensor,
        K: int,
        n_total: int,
        index: torch.Tensor,
    ) -> torch.Tensor:
        m_all = torch.arange(7, device=t.device).reshape(1, 1, 7)
        u = torch.remainder(m_all + c.unsqueeze(-1) - 1, 7) + 1
        t_exp = t.reshape(-1, 1, 1)
        k_min = torch.where(
            t_exp >= u,
            torch.div(t_exp - u, 7, rounding_mode="floor") + 1,
            torch.zeros_like(u),
        )
        k_in = torch.clamp(k_min, 0, K)
        pre = prefix.unsqueeze(1).expand(-1, 7, -1, -1)
        log_qbar = torch.gather(pre, 3, k_in.unsqueeze(-1)).squeeze(-1)
        log_qbar = log_qbar + torch.clamp(k_min - K, min=0).to(prefix.dtype) * log_tail_s
        value = torch.logsumexp(log_pi.unsqueeze(1) + log_qbar, dim=-1)
        out = torch.zeros(n_total, 7, dtype=value.dtype, device=value.device)
        return out.index_copy(0, index, value)

    def emission_log_prob(
        self,
        embed: torch.Tensor,
        gap: torch.Tensor,
        event: torch.Tensor,
        resid: torch.Tensor,
    ) -> torch.Tensor:
        if self.arm in (CALENDAR_BLIND, FLEXIBLE_ELAPSED):
            row = self._blind_observed_log_prob(embed, gap, event)
            return row.unsqueeze(-1).expand(*row.shape, 7)

        K = int(self.cfg.weeks)
        log_pi = self._log_pi(embed, resid)
        log_haz, prefix = self._week_log_terms(embed)
        log_tail_h, log_tail_s = self._tail()

        t = gap.long()
        k_obs = torch.div(t - 1, 7, rounding_mode="floor")
        z = torch.arange(7, device=gap.device).reshape(1, 1, 7)
        c = torch.remainder(z + int(self.eval_rotation) - resid.unsqueeze(-1), 7)
        m_obs = torch.remainder(t.unsqueeze(-1) - c, 7)

        k_clamped = torch.clamp(k_obs, 0, K - 1)
        inside = k_obs < K
        gather_m = m_obs.unsqueeze(-1)
        haz_at = (
            torch.gather(log_haz, 2, gather_m.expand(-1, -1, -1, K))
            .gather(3, k_clamped.reshape(*k_obs.shape, 1, 1).expand(-1, -1, 7, 1))
            .squeeze(-1)
        )
        pre_at = (
            torch.gather(prefix, 2, gather_m.expand(-1, -1, -1, K + 1))
            .gather(3, k_clamped.reshape(*k_obs.shape, 1, 1).expand(-1, -1, 7, 1))
            .squeeze(-1)
        )
        pre_full = torch.gather(prefix, 2, gather_m.expand(-1, -1, -1, K + 1))[..., K]
        beyond = torch.clamp(k_obs - K, min=0).unsqueeze(-1).to(prefix.dtype)
        log_q_event = torch.where(
            inside.unsqueeze(-1),
            pre_at + haz_at,
            pre_full + beyond * log_tail_s + log_tail_h,
        )
        log_event = torch.gather(log_pi, 2, m_obs) + log_q_event

        flat_ev = event.reshape(-1)
        cens_idx = torch.nonzero(
            (flat_ev < 0.5) & (gap.reshape(-1) > 0), as_tuple=False
        ).squeeze(-1)
        log_cens_flat = torch.zeros(
            flat_ev.shape[0], 7, dtype=log_pi.dtype, device=log_pi.device
        )
        if cens_idx.numel() > 0:
            log_cens_flat = self._censored_log_prob(
                log_pi.reshape(-1, 7)[cens_idx],
                prefix.reshape(-1, 7, K + 1)[cens_idx],
                t.reshape(-1)[cens_idx],
                c.reshape(-1, 7)[cens_idx],
                log_tail_s,
                K,
                flat_ev.shape[0],
                cens_idx,
            )
        ev = event.unsqueeze(-1)
        return ev * log_event + (1.0 - ev) * log_cens_flat.reshape(*event.shape, 7)

    def _log_transition(self, embed: torch.Tensor) -> torch.Tensor:
        assert self.drift_head is not None
        n_p, max_j = embed.shape[0], embed.shape[1]
        if self.arm == ABSOLUTE_PHASE:
            logits = self.drift_head(embed).reshape(n_p, max_j, 7, 7)
            return torch.log_softmax(logits, dim=-1)
        drift = torch.log_softmax(
            self.drift_head(embed) / float(self.cfg.drift_temp), dim=-1
        )
        z = torch.arange(7, device=embed.device)
        offset = torch.remainder(z.reshape(1, 7) - z.reshape(7, 1), 7)
        return drift[..., offset]

    def patient_log_likelihood(
        self, bundle_t: Dict[str, torch.Tensor], return_posterior: bool = False
    ):
        embed = self.encoder(bundle_t["feat"])
        if self.arm in (CALENDAR_BLIND, FLEXIBLE_ELAPSED):
            row = self._blind_observed_log_prob(
                embed, bundle_t["gap"], bundle_t["event"]
            )
            total = (row * bundle_t["mask"]).sum(dim=1)
            return (total, None) if return_posterior else total

        assert self.init_logits is not None
        feat, gap = bundle_t["feat"], bundle_t["gap"]
        event, mask, resid = bundle_t["event"], bundle_t["mask"], bundle_t["resid"]
        emis = self.emission_log_prob(embed, gap, event, resid) * mask.unsqueeze(-1)
        log_trans = self._log_transition(embed)
        z = torch.arange(7, device=feat.device).reshape(1, 7)
        alpha = torch.log_softmax(
            self.init_logits[torch.remainder(z - resid[:, :1], 7)], dim=-1
        )
        max_j = int(gap.shape[1])
        posteriors: List[torch.Tensor] = []
        alpha = alpha + emis[:, 0]
        if return_posterior:
            posteriors.append(torch.log_softmax(alpha, dim=-1))
        for j in range(1, max_j):
            moved = torch.logsumexp(alpha.unsqueeze(-1) + log_trans[:, j], dim=1)
            live = mask[:, j].unsqueeze(-1)
            alpha = torch.where(live > 0, moved + emis[:, j], alpha)
            if return_posterior:
                posteriors.append(torch.log_softmax(alpha, dim=-1))
        total = torch.logsumexp(alpha, dim=-1)
        if return_posterior:
            return total, torch.stack(posteriors, dim=1)
        return total

    def _predictive_log_beliefs(self, bundle_t: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.arm in (CALENDAR_BLIND, FLEXIBLE_ELAPSED) or self.init_logits is None:
            raise RuntimeError(f"{self.arm} has no latent phase belief")
        feat, gap = bundle_t["feat"], bundle_t["gap"]
        event, mask, resid = bundle_t["event"], bundle_t["mask"], bundle_t["resid"]
        embed = self.encoder(feat)
        emission = self.emission_log_prob(embed, gap, event, resid) * mask.unsqueeze(-1)
        transition = self._log_transition(embed)
        z = torch.arange(7, device=feat.device).reshape(1, 7)
        alpha = torch.log_softmax(
            self.init_logits[torch.remainder(z - resid[:, :1], 7)], dim=-1
        )
        beliefs = [alpha]
        for j in range(1, int(gap.shape[1])):
            alpha = alpha + emission[:, j - 1]
            moved = torch.logsumexp(alpha.unsqueeze(-1) + transition[:, j], dim=1)
            beliefs.append(torch.log_softmax(moved, dim=-1))
            alpha = moved
        return torch.stack(beliefs, dim=1)

    def predictive_rows(self, bundle_t: Dict[str, torch.Tensor]) -> torch.Tensor:
        """One-step-ahead observed log score for each patient and interval."""
        embed = self.encoder(bundle_t["feat"])
        if self.arm in (CALENDAR_BLIND, FLEXIBLE_ELAPSED):
            return self._blind_observed_log_prob(
                embed, bundle_t["gap"], bundle_t["event"]
            )
        emission = self.emission_log_prob(
            embed, bundle_t["gap"], bundle_t["event"], bundle_t["resid"]
        )
        belief = self._predictive_log_beliefs(bundle_t)
        return torch.logsumexp(belief + emission, dim=-1)

    def predictive_gap_pmf(
        self, bundle_t: Dict[str, torch.Tensor], interval: int, days: int = 420
    ) -> torch.Tensor:
        """One-step-ahead day PMF on 1..days given strictly prior intervals."""
        embed_all = self.encoder(bundle_t["feat"])
        embed = embed_all[:, int(interval)]
        K = int(self.cfg.weeks)
        log_haz, prefix = self._week_log_terms(embed)
        log_tail_h, log_tail_s = self._tail()
        day = torch.arange(1, int(days) + 1, device=embed.device)
        week = torch.div(day - 1, 7, rounding_mode="floor")

        if self.arm == CALENDAR_BLIND:
            week_in = torch.clamp(week, 0, K - 1)
            haz = log_haz[:, 0].gather(1, week_in.reshape(1, -1).expand(embed.shape[0], -1))
            pre = prefix[:, 0].gather(1, week_in.reshape(1, -1).expand(embed.shape[0], -1))
            beyond = torch.clamp(week - K, min=0).to(prefix.dtype).reshape(1, -1)
            log_q = torch.where(
                (week < K).reshape(1, -1),
                pre + haz,
                prefix[:, 0, K].unsqueeze(-1) + beyond * log_tail_s + log_tail_h,
            )
            return torch.exp(log_q) / 7.0

        belief = self._predictive_log_beliefs(bundle_t)[:, int(interval)]
        r_j = bundle_t["resid"][:, int(interval)]
        log_pi = self._log_pi(embed, r_j)
        k_in = torch.clamp(week, 0, K - 1)
        n_p = embed.shape[0]
        z = torch.arange(7, device=embed.device).reshape(1, 7)
        c = torch.remainder(z + int(self.eval_rotation) - r_j.unsqueeze(-1), 7)
        m = torch.remainder(day.reshape(1, 1, -1) - c.unsqueeze(-1), 7)

        haz = log_haz.gather(2, k_in.reshape(1, 1, -1).expand(n_p, 7, int(days)))
        pre = prefix.gather(2, k_in.reshape(1, 1, -1).expand(n_p, 7, int(days)))
        full = prefix[..., K].unsqueeze(-1)
        beyond = torch.clamp(week - K, min=0).to(prefix.dtype).reshape(1, 1, -1)
        log_q = torch.where(
            (week < K).reshape(1, 1, -1),
            pre + haz,
            full + beyond * log_tail_s + log_tail_h,
        )
        log_q_z = torch.gather(
            log_q.unsqueeze(1).expand(-1, 7, -1, -1), 2, m.unsqueeze(2)
        ).squeeze(2)
        log_pi_z = torch.gather(log_pi.unsqueeze(1).expand(-1, 7, -1), 2, m)
        return torch.logsumexp(belief.unsqueeze(-1) + log_pi_z + log_q_z, dim=1).exp()


class FlexibleElapsedSurvivalModel(MatchedNeuralSurvivalModel):
    """Elapsed-day weekly-heaping control with 4,356 active parameters and zero phase inputs."""

    def __init__(
        self,
        n_features: int = 19,
        seed: int = 0,
        hidden: int = 64,
        embed: int = 16,
        weeks: int = 60,
        l2: float = 1e-4,
        dropout: float = 0.0,
    ):
        base_cfg = MatchedNeuralConfig(
            arm=CALENDAR_BLIND,
            hidden=hidden,
            embed=embed,
            weeks=weeks,
            dropout=dropout,
            l2=l2,
            seed=seed,
        )
        super().__init__(n_features, base_cfg)
        self.arm = FLEXIBLE_ELAPSED
        object.__setattr__(self.cfg, "arm", FLEXIBLE_ELAPSED)
        torch.manual_seed(int(seed) + 901)
        self.elapsed_pi = nn.Linear(int(embed), 7)
        self.elapsed_class_shift = nn.Parameter(torch.zeros(7))
        self.capacity_adapter = PhaseFreeWeekAdapter(int(embed), 49, int(weeks))
        expected = expected_parameter_count(n_features, base_cfg)
        if self.n_parameters() != expected:
            raise AssertionError(
                f"Flexible-elapsed parameter count {self.n_parameters()} != {expected}"
            )

    def _week_log_terms(
        self, embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self._shared_week_logits(embed).unsqueeze(-2)
        logits = logits + self.elapsed_class_shift.reshape(
            *([1] * (embed.dim() - 1)), 7, 1
        )
        logits = logits.clamp(-18.0, 18.0)
        haz = torch.nn.functional.logsigmoid(logits)
        surv = torch.nn.functional.logsigmoid(-logits)
        return haz, torch.cat((torch.zeros_like(surv[..., :1]), surv.cumsum(-1)), -1)

    def _blind_observed_log_prob(
        self, embed: torch.Tensor, gap: torch.Tensor, event: torch.Tensor
    ) -> torch.Tensor:
        K = int(self.cfg.weeks)
        log_pi = self.elapsed_pi(embed).log_softmax(-1)
        haz, prefix = self._week_log_terms(embed)
        tail_h, tail_s = self._tail()
        t = gap.long()
        k = torch.div(t - 1, 7, rounding_mode="floor")
        r = (t - 1).remainder(7)
        k_in = k.clamp(0, K - 1)
        haz_k = haz.gather(-1, k_in[..., None, None].expand(*k.shape, 7, 1)).squeeze(-1)
        pre_k = prefix.gather(-1, k_in[..., None, None].expand(*k.shape, 7, 1)).squeeze(-1)
        log_q = torch.where(
            (k < K)[..., None],
            haz_k + pre_k,
            prefix[..., K] + (k - K).clamp(min=0)[..., None] * tail_s + tail_h,
        )
        log_event = (log_pi + log_q).gather(-1, r[..., None]).squeeze(-1)
        first = torch.arange(1, 8, device=t.device)
        k_min = torch.div(t[..., None] - first, 7, rounding_mode="floor").add(1).clamp(min=0)
        log_surv = prefix.gather(-1, k_min.clamp(max=K)[..., None]).squeeze(-1)
        log_surv = log_surv + (k_min - K).clamp(min=0) * tail_s
        log_censor = torch.logsumexp(log_pi + log_surv, dim=-1)
        return torch.where(event > 0.5, log_event, log_censor)

    def predictive_gap_pmf(
        self, bundle_t: Dict[str, torch.Tensor], interval: int, days: int = 420
    ) -> torch.Tensor:
        embed = self.encoder(bundle_t["feat"])[:, int(interval)]
        K = int(self.cfg.weeks)
        haz, prefix = self._week_log_terms(embed)
        tail_h, tail_s = self._tail()
        day = torch.arange(1, int(days) + 1, device=embed.device)
        week = torch.div(day - 1, 7, rounding_mode="floor")
        idx = week.clamp(max=K - 1).reshape(1, 1, -1).expand(embed.shape[0], 7, -1)
        q = torch.where(
            (week < K)[None, None],
            haz.gather(2, idx) + prefix.gather(2, idx),
            prefix[..., K, None] + (week - K).clamp(min=0)[None, None] * tail_s + tail_h,
        )
        lp = self.elapsed_pi(embed).log_softmax(-1)[..., None] + q
        take = (day - 1).remainder(7)[None, None].expand(embed.shape[0], 1, -1)
        return lp.gather(1, take).squeeze(1).exp()


def expected_parameter_count(n_features: int, config: MatchedNeuralConfig) -> int:
    """Exact parameter count shared across all four matched neural arms."""
    F = int(n_features)
    h = int(config.hidden)
    e = int(config.embed)
    K = int(config.weeks)
    return int(F * h + h + h * e + (K + 57) * e + K + 120)


def build_matched_model(
    arm: str,
    n_features: int = 19,
    seed: int = 0,
    hidden: int = 64,
    embed: int = 16,
    weeks: int = 60,
    l2: float = 1e-4,
    dropout: float = 0.0,
    drift_temp: float = 1.0,
) -> MatchedNeuralSurvivalModel:
    canonical = canonical_arm_name(arm)
    if canonical == FLEXIBLE_ELAPSED:
        return FlexibleElapsedSurvivalModel(
            n_features=n_features,
            seed=seed,
            hidden=hidden,
            embed=embed,
            weeks=weeks,
            l2=l2,
            dropout=dropout,
        )
    return MatchedNeuralSurvivalModel(
        int(n_features),
        MatchedNeuralConfig(
            arm=canonical,
            hidden=hidden,
            embed=embed,
            weeks=weeks,
            dropout=dropout,
            drift_temp=drift_temp,
            l2=l2,
            seed=seed,
        ),
    )


def component_parameter_counts(model: MatchedNeuralSurvivalModel) -> Dict[str, int]:
    """Decompose trainable scalars into shared encoder, shared timing, and arm-specific blocks."""
    groups = {
        "encoder": ("encoder.",),
        "shared_timing": ("week_base", "week_load", "tail_logit"),
        "phase_landing": ("pi_head.", "class_shift", "elapsed_pi.", "elapsed_class_shift"),
        "phase_initial": ("init_logits",),
        "transition": ("drift_head.",),
        "absolute_residue": ("abs_class",),
        "phase_free_adapter": ("capacity_adapter.",),
    }
    out = {group: 0 for group in groups}
    assigned = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        hits = [
            group
            for group, prefixes in groups.items()
            if any(name == p or name.startswith(p) for p in prefixes)
        ]
        if len(hits) != 1:
            raise AssertionError(f"Ambiguous parameter group for {name}: {hits}")
        out[hits[0]] += int(parameter.numel())
        assigned.add(name)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(assigned) != len(trainable):
        raise AssertionError("Not every trainable tensor was assigned to a component")
    out["total"] = int(sum(out.values()))
    return out


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = f"{tuple(value.shape)}|{value.dtype}|"
    return hashlib.sha256(header.encode("utf-8") + value.numpy().tobytes()).hexdigest()


def paired_initialization_summary(
    models: Mapping[str, MatchedNeuralSurvivalModel]
) -> Dict[str, object]:
    """Check byte-for-byte equality of shared encoder and timing initialization across arms."""
    canonical_models = {canonical_arm_name(k): v for k, v in models.items()}
    first_arm = next(iter(canonical_models))
    first_model = canonical_models[first_arm]
    encoder_linear = tuple(
        f"encoder.{idx}.{attr}"
        for idx, mod in enumerate(first_model.encoder)
        if isinstance(mod, nn.Linear)
        for attr in ("weight", "bias")
    )
    common = encoder_linear + (
        "week_base",
        "week_load",
        "tail_logit",
    )
    equal: Dict[str, bool] = {}
    hashes: Dict[str, Dict[str, str]] = {}
    for name in common:
        values = {
            arm: dict(model.named_parameters())[name]
            for arm, model in canonical_models.items()
        }
        hashes[name] = {arm: _tensor_sha256(val) for arm, val in values.items()}
        ref = values[first_arm]
        equal[name] = bool(all(torch.equal(ref, val) for val in values.values()))

    if ABSOLUTE_PHASE in canonical_models and RELATIVE_PHASE in canonical_models:
        for name in ("pi_head.weight", "pi_head.bias", "class_shift", "init_logits"):
            va = dict(canonical_models[ABSOLUTE_PHASE].named_parameters())[name]
            vr = dict(canonical_models[RELATIVE_PHASE].named_parameters())[name]
            hashes[name] = {
                ABSOLUTE_PHASE: _tensor_sha256(va),
                RELATIVE_PHASE: _tensor_sha256(vr),
            }
            equal[name] = bool(torch.equal(va, vr))

    return {
        "common_tensor_sha256": hashes,
        "tensor_byte_identity": equal,
        "all_shared_tensors_byte_identical": bool(all(equal.values())),
    }


def verify_all_parameters_active(
    model: MatchedNeuralSurvivalModel, seed: int = 20260828
) -> Dict[str, object]:
    """Verify via random linear loss projections that all trainable scalars receive nonzero gradients."""
    param = next(model.parameters())
    dtype, device = param.dtype, param.device
    gen = torch.Generator(device=device)
    gen.manual_seed(240819)
    patients, intervals = 7, 5
    feat = torch.randn(
        patients, intervals, model.n_features, dtype=dtype, device=device, generator=gen
    )
    K = int(model.cfg.weeks)
    gap = torch.empty(patients, intervals, dtype=dtype, device=device)
    for p in range(patients):
        gap[p] = torch.tensor(
            [p + 1, p + 8, 7 * K + p + 1, p + 30, 7 * K + p + 80],
            dtype=dtype,
            device=device,
        )
    event = torch.ones_like(gap)
    event[:, 3] = 0.0
    mask = torch.ones_like(gap)
    resid = torch.stack(
        [
            torch.remainder(torch.arange(intervals, device=device) + p, 7)
            for p in range(patients)
        ]
    ).long()
    tensors = {"feat": feat, "gap": gap, "event": event, "mask": mask, "resid": resid}

    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    maximum = {n: torch.zeros_like(p) for n, p in parameters.items()}
    was_training = model.training
    model.eval()
    for proj in range(4):
        model.zero_grad(set_to_none=True)
        rows = model.predictive_rows(tensors)
        w_gen = torch.Generator(device=rows.device)
        w_gen.manual_seed(int(seed) + proj)
        weight = torch.randn(rows.shape, dtype=rows.dtype, device=rows.device, generator=w_gen)
        objective = (rows * tensors["mask"] * weight).sum()
        objective.backward()
        for name, p in parameters.items():
            if p.grad is not None:
                maximum[name] = torch.maximum(maximum[name], p.grad.detach().abs())
    if was_training:
        model.train()

    raw = sum(int(p.numel()) for p in parameters.values())
    active = sum(int(torch.count_nonzero(v).item()) for v in maximum.values())
    finite = all(bool(torch.isfinite(v).all()) for v in maximum.values())
    return {
        "arm": model.arm,
        "trainable_parameters": raw,
        "active_parameters": active,
        "all_active": bool(active == raw and finite),
    }


def to_tensors(
    bundle: SeqBundle, device: str = "cpu", dtype: str = "float32"
) -> Dict[str, torch.Tensor]:
    td = getattr(torch, dtype)
    return {
        "feat": torch.tensor(bundle.feat, dtype=td, device=device),
        "gap": torch.tensor(bundle.gap, dtype=td, device=device),
        "event": torch.tensor(bundle.event, dtype=td, device=device),
        "mask": torch.tensor(bundle.mask, dtype=td, device=device),
        "resid": torch.tensor(bundle.resid, dtype=torch.long, device=device),
    }


def subset_tensors(tensors: Dict[str, torch.Tensor], index) -> Dict[str, torch.Tensor]:
    return {k: v[index] for k, v in tensors.items()}


def trim_tensors(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    live = (
        int(tensors["mask"].sum(dim=0).nonzero().max().item()) + 1
        if float(tensors["mask"].sum()) > 0
        else 1
    )
    if live >= tensors["mask"].shape[1]:
        return tensors
    return {k: (v[:, :live] if v.dim() >= 2 else v) for k, v in tensors.items()}


@dataclass
class TrainConfig:
    epochs: int = 400
    lr: float = 0.02
    batch_patients: int = 0
    patience: int = 40
    val_fraction: float = 0.2
    seed: int = 0
    verbose: bool = False


def train_model(
    model: MatchedNeuralSurvivalModel,
    tensors: Dict[str, torch.Tensor],
    cfg: TrainConfig,
) -> Dict[str, object]:
    """Train with Adam and development-only early stopping."""
    rng = np.random.default_rng(int(cfg.seed))
    n_p = int(tensors["gap"].shape[0])
    order = rng.permutation(n_p)
    n_val = max(1, int(round(cfg.val_fraction * n_p)))
    val_idx = np.sort(order[:n_val])
    tr_idx = np.sort(order[n_val:])
    train_t = subset_tensors(tensors, torch.tensor(tr_idx, device=tensors["gap"].device))
    val_t = trim_tensors(
        subset_tensors(tensors, torch.tensor(val_idx, device=tensors["gap"].device))
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(cfg.lr), weight_decay=float(model.cfg.l2))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(5, cfg.patience // 4)
    )

    best = float("inf")
    best_state = None
    bad = 0
    history: List[Dict[str, float]] = []
    n_train = int(train_t["gap"].shape[0])
    batch = int(cfg.batch_patients) or n_train
    lengths = train_t["mask"].sum(dim=1).cpu().numpy()
    by_length = np.argsort(lengths, kind="mergesort")
    buckets = [by_length[s : s + batch] for s in range(0, n_train, batch)]

    for epoch in range(int(cfg.epochs)):
        model.train()
        running = 0.0
        for position in rng.permutation(len(buckets)):
            take = torch.tensor(
                np.sort(buckets[position]), device=train_t["gap"].device
            )
            chunk = trim_tensors(subset_tensors(train_t, take))
            opt.zero_grad()
            loss = -model.patient_log_likelihood(chunk).sum()
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch}")
            (loss / max(take.numel(), 1)).backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            running += float(loss)
        tr = running / max(n_train, 1)
        model.eval()
        with torch.no_grad():
            val = float(-model.patient_log_likelihood(val_t).mean())
        sched.step(val)
        history.append({"epoch": float(epoch), "train_nll": tr, "val_nll": val})
        if val < best - 1e-6:
            best = val
            bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {
        "best_val_nll": float(best),
        "epochs_run": len(history),
        "history": history,
        "n_parameters": int(model.n_parameters()),
    }


def score_patients(
    model: MatchedNeuralSurvivalModel,
    tensors: Dict[str, torch.Tensor],
    batch: int = 256,
) -> np.ndarray:
    """Compute exact per-patient log-likelihoods on held-out sequences."""
    model.eval()
    n = int(tensors["gap"].shape[0])
    lengths = tensors["mask"].sum(dim=1).cpu().numpy()
    order = np.argsort(lengths, kind="mergesort")
    out = np.zeros(n, dtype=float)
    with torch.no_grad():
        for start in range(0, n, batch):
            take = np.sort(order[start : start + batch])
            idx = torch.tensor(take, device=tensors["gap"].device)
            val = model.patient_log_likelihood(trim_tensors(subset_tensors(tensors, idx)))
            out[take] = val.cpu().numpy().astype(float)
    return out
