# Reproducibility Guide

This document describes the experimental protocols, parameter-matching invariants, statistical resampling procedures, and verification commands used in *Auditing and Enforcing Calendar-Shift Invariance in Clinical Prediction Models*.

## 1. Environment Setup

Install the pinned Python dependencies in a clean virtual environment (Python 3.10+):

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

## 2. Parameter Matching and Shift-Invariance Architecture

### 2.1 Neural Recurrent-Event Survival Models (`src/calendar_shift_audit/models.py`)

All four neural survival arms share identical feature dimensionality (`F = 19`: 8 standardized interval/history covariates from `common8_design` plus 11 strictly prior gap-history summaries), hidden width (`h = 64`), latent representation dimension (`e = 16`), explicit horizon (`K = 60` weeks = `420` days), and 7 geometric tail channels (`> 420` days), totaling exactly **4,356 active trainable parameters**:

- **Shared encoder (`2,320` scalars)**: two-layer `Tanh` MLP mapping `19 -> 64 -> 16`.
- **Shared week/tail timing head (`1,021` scalars)**: baseline week logits (`60`), history-to-week loading (`60 x 16 = 960`), and scalar geometric tail hazard logit (`1`).
- **Arm-specific timing/phase head (`1,015` scalars)**:
  - `Calendar-blind`: `PhaseFreeWeekAdapter` of width `56` (`1,015` scalars) without calendar-phase state.
  - `Flexible-elapsed`: elapsed modulo-7 landing head (`119` scalars), elapsed class shifts (`7` scalars), and `PhaseFreeWeekAdapter` of width `49` (`889` scalars; `1,015` total) without calendar-phase inputs.
  - `Absolute-phase`: latent modulo-7 booking-phase filter (`126 + 7 + 833 = 966` scalars) plus a `7 x 7` absolute opening-weekday table (`49` scalars; `1,015` total).
  - `Relative-phase`: equivariant cyclic modulo-7 latent booking-phase filter (`126 + 7 + 119 = 252` scalars) plus `PhaseFreeWeekAdapter` of width `42` (`763` scalars; `1,015` total).

For every seed `s in {3, ..., 17}` and held-out hospital `h in {DFCI, MSK, UHN, VICC}`, shared encoder and timing weights are initialized byte-identically across paired arms (`paired_initialization_summary`), and `verify_all_parameters_active` verifies that all `4,356` scalars receive nonzero gradients. Under any per-patient integer date shift, `Calendar-blind` and `Flexible-elapsed` have zero mathematical dependence on `resid`, and `Relative-phase` permutes its 7 latent phase channels equivariantly while producing shift-invariant predictive gap PMFs up to floating-point precision.

### 2.2 Gradient-Boosted Discrete-Hazard Survival (`src/calendar_shift_audit/xalign.py`)

In the boosted tree benchmark on NSCLC (`4` institutions) and CRC (`3` institutions):

- Each interval is expanded into person-period daily hazard rows up to `120` explicit days (`HORIZON_DAYS = 120`) alongside candidate-day predictors `(u, u mod 7, log u)` and a development-estimated geometric tail hazard beyond day `120`.
- `X-base` uses the 8 shift-invariant interval history covariates (`COMMON8_FEATURES`).
- `X-align` (`keep_features = 1`) appends **one** shift-invariant phase-alignment feature derived from the patient's observed opening day and strictly prior assessment days: the circular concentration parameter `kappa` is estimated by L-BFGS on the outer development fold only (`estimate_phase_kappa`), yielding a square-root-tempered modulo-7 posterior `p_i(r)` that shifts equivariantly when all dates of patient `i` shift by `delta_i`. Evaluating `p_i` at candidate landing residue `(d_{i,j} + u) mod 7` cancels `delta_i` identically (`all_interval_max_spread == 0.0` across all 56 fitted boosters). Optional feature expansions (`keep_features in {2, 4}`) expose neighboring-day mass, posterior concentration, and circular distance.
- Six rotated negative-control arms (`X-align-rot1` through `X-align-rot6`) apply a fixed nonzero cyclic offset `c in {1, ..., 6}` to the posterior before computing the candidate-day feature, preserving feature dimensionality and marginal scale while breaking alignment with the patient's empirical booking phase.

### 2.3 Continuous Glucose Monitoring Audits (`src/calendar_shift_audit/glucose.py`)

- **GlucoBench Colas**: Primary evaluation uses `test_ood` (`21` held-out participants, `9,168` 60-minute windows strictly disjoint from training/validation). Secondary evaluation reports GlucoBench's native `test` split (`176` participants, `19,192` windows, which shares participant trajectories temporally with the fitting partitions). Released Colas timestamps begin at synthetic origin `2012-01-01`; the `canonical` remedy maps each participant's first observed timestamp to Monday `2000-01-03` while preserving time-of-day (`canonical_relative_dates`), and `random_anchor` randomizes whole-patient training anchors across offsets `0..7` (`patient_anchor_schedule`).
- **CGMacros (`test`, `44` held-out participants, `430` windows)**:
  - `glucose_audit_summary.csv` and `glucose_offset_rmse.csv` pool squared forecast errors across all `430` windows (`5,160` 5-minute target occurrences; shift-0 pooled RMSE `13.92` mg/dL for `native` vs. `14.44` mg/dL for `relative_origin` mapped to `2000-01-03`).
  - `cgmacros_window_audit.csv` reports the companion participant-median-window GluFormer diagnostic (`median_window_rmse_mgdl`, shift-0 score `6.73` mg/dL) across both the 7-day shift grid (`0..6`, mean patient range `0.90` mg/dL) and weekday-preserving multiples of 7 (`0, ±7, ±14` days, mean patient range `0.58` mg/dL) that isolate day-of-month and day-of-year sensitivity from weekday effects.
- **Consensus glycemic categories**: Predictions are classified into `< 70` mg/dL (hypoglycemia), `70–180` mg/dL (target range), and `> 180` mg/dL (hyperglycemia) to measure shift-induced clinical category flips.

## 3. Statistical Resampling

- **Crossed whole-patient and seed bootstrap (`crossed_patient_seed_bootstrap`)**: Used for the 15-seed neural survival campaign (`20,000` draws, seed `20260828`), resampling patients with replacement within each hospital and seeds with replacement across the 15 training seeds, weighting patients equally within hospital and hospitals equally overall.
- **Hospital-stratified patient cluster bootstrap (`ClusterBootstrap`)**: Used for deterministic `X-align` contrasts (`20,000` draws, seed `20260819` for NSCLC and `20260820` for CRC), with Bonferroni family-wise error control (`alpha = 0.05 / 6`, `99.17%` CIs) across the six rotated negative controls within each disease cohort. Coarse (`xalign_history_depth.csv`) and fine (`xalign_fine_history_depth.csv`) history-depth tables share identical point estimates and use separate `20,000`-draw hospital-cluster bootstrap streams.

## 4. End-to-End Verification Commands

1. **Run all unit and manuscript-table tests**:
   ```bash
   pytest -v
   ```
2. **Reproduce the synthetic audit, 200-replicate negative control, all 6 figures, and 200+ quantitative and structural checks**:
   ```bash
   python scripts/reproduce_all.py
   ```
3. **Run standalone numerical verification**:
   ```bash
   python scripts/verify_repository.py
   ```
4. **Rebuild vector PDF and 300-DPI PNG figures from `results/tables/`**:
   ```bash
   python scripts/build_figures.py
   ```
