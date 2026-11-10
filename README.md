# Auditing and Enforcing Calendar-Shift Invariance in Clinical Prediction Models

Companion code, aggregate result tables, and figure-reproduction scripts for the ML4H 2026 Findings Track paper:

> **Auditing and Enforcing Calendar-Shift Invariance in Clinical Prediction Models**

## Overview

De-identified clinical datasets routinely shift all dates of a patient by a secret, patient-specific whole-day offset to protect privacy while preserving elapsed durations between clinical events. Models that consume calendar phase features (such as day-of-week, day-of-month, or day-of-year encodings) without accounting for arbitrary per-patient origin shifts can change their predictions depending on which date offset was assigned before release.

This repository implements:

1. **Calendar-shift auditing (`calendar_shift_audit.models`, `calendar_shift_audit.tv`, `calendar_shift_audit.glucose`)**: Patient-level and pooled evaluation protocols over the 7-day modulo-7 shift orbit (and multi-week offsets for continuous glucose monitoring), including exact log-likelihood ranges, certified Total Variation (TV) distance over finite-day plus geometric-tail predictive distributions, and consensus glycemic category-crossing metrics.
2. **Capacity-matched neural survival representations (`calendar_shift_audit.models`)**: Four recurrent discrete-time survival architectures matched at exactly `4,356` active scalar parameters (`Calendar-blind`, `Flexible-elapsed`, `Absolute-phase`, and `Relative-phase`).
3. **Shift-invariant phase alignment for gradient-boosted trees (`calendar_shift_audit.xalign`)**: Patient-relative modulo-7 booking-phase posterior features (`X-align`) paired with `X-base` and six fixed nonzero phase-rotation controls (`X-align-rot1` through `X-align-rot6`).
4. **Known-nuisance negative control (`calendar_shift_audit.negative_control`)**: Synthetic 200-replicate discrete-hazard benchmark isolating finite-sample estimation variance across phase-stratified vs. shift-invariant estimators at `N in {200, 2000, 20000}`.
5. **Aggregate tables and figure reproduction (`results/`, `scripts/`)**: De-identified institutional and overall summary tables, deterministic figure rendering (`results/figures/`), and automated numerical verification against the manuscript.

## Data Availability

Patient-level clinical data are not redistributed in this repository. Upstream access instructions for AACR Project GENIE Biopharma Collaborative (BPC) v2.0 (NSCLC and CRC), PhysioNet CGMacros v1.0.0, and the Colas et al. / GlucoBench continuous glucose monitoring cohort are documented in [`data/README.md`](data/README.md).

All synthetic invariance audits, the 200-replicate known-nuisance negative control, figure rendering from the released aggregate tables, and numerical verification run directly without external patient data.

## Repository Layout

```text
├── configs/
│   └── protocols.json               # Model dimensions, bootstrap seeds, and evaluation thresholds
├── data/
│   └── README.md                    # Upstream dataset access and local directory conventions
├── results/
│   ├── figures/                     # Deterministic vector (.pdf) and raster (.png) figures
│   └── tables/                      # Aggregate institutional and overall CSV/JSON result tables
├── scripts/
│   ├── build_figures.py             # Render Figures 1-6 from results/tables/
│   ├── reproduce_all.py             # Run synthetic audit, negative control, figures, and checks
│   ├── run_genie_campaign.py        # Leave-one-institution-out runner for local GENIE BPC tables
│   ├── run_glucose_campaign.py      # 7-offset CGM forecast sensitivity runner
│   └── verify_repository.py         # Cross-check tables, live model invariants, and figures
├── src/calendar_shift_audit/
│   ├── bootstrap.py                 # Hospital-cluster and crossed patient-seed bootstrap
│   ├── cli.py                       # Command-line entrypoint (calendar-shift-audit)
│   ├── cohort.py                    # GENIE BPC interval construction and date-shift operators
│   ├── config.py                    # Protocol configuration loader
│   ├── glucose.py                   # CGM timestamp canonicalization and glycemic shift metrics
│   ├── models.py                    # Matched 4,356-parameter neural recurrent survival models
│   ├── negative_control.py          # Known-nuisance calendar-phase negative control
│   ├── tv.py                        # Certified Total Variation distance and 7-shift TTA
│   └── xalign.py                    # Shift-invariant phase-alignment features for XGBoost
└── tests/                           # Unit and manuscript-verification test suite
```

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

## Quick Start

Run the unit and numerical verification test suite:

```bash
pytest -v
```

Run the synthetic 7-shift invariance check, recompute the 200-replicate known-nuisance negative control, regenerate all figures in `results/figures/`, and verify every released table and figure output:

```bash
python scripts/reproduce_all.py
```

The package also provides a command-line interface:

```bash
calendar-shift-audit synthetic-audit
calendar-shift-audit negative-control --output-csv results/tables/negative_control_summary.csv
```

When local copies of the upstream GENIE BPC v2.0 release files are available (see [`data/README.md`](data/README.md)), construct the recurrent assessment cohort and run leave-one-institution-out training and shift auditing:

```bash
calendar-shift-audit build-genie-cohort --release-dir /path/to/genie_nsclc_v2 --output-parquet data/nsclc_frame.parquet
python scripts/run_genie_campaign.py --release-dir /path/to/genie_nsclc_v2 --disease nsclc --output-json results/genie_nsclc_run.json
```

## Manuscript Crosswalk

| Manuscript Element | Summary | Repository Table / Script |
| :--- | :--- | :--- |
| Section 3.1 & Appendix B | NSCLC (`n = 1,825`, `47,556` completed gaps) and CRC (`n = 1,471`, `47,448` completed gaps) cohort counts | `results/tables/cohort_summary.json` |
| Table 1 & Appendix A | Matched `4,356`-parameter neural survival log scores, MAE, and 7-shift ranges | `results/tables/table1_neural_summary.csv`, `table_a1_neural_contrasts.csv`, `table_a2_neural_by_hospital.csv` |
| Section 3.2 & Appendix A | Pooled common-shift vs. patient-specific shift sensitivity and ECDF | `results/tables/neural_pooled_shift_audit.csv`, `neural_patient_score_ecdf.csv`, `figure_5_patient_score_ecdf.pdf` |
| Figure 1 & Section 3.3 | Deep ensemble size (`M = 1, 3, 5, 7, 15`) calendar-shift TV vs. disjoint seed TV | `results/tables/figure1_ensemble_tv_curve.csv`, `ensemble_tv_contrasts.csv`, `figure_1_ensemble_tv.pdf` |
| Section 3.3 & Appendix G | Test-time 7-origin averaging, 15-seed training augmentation, and negative control | `results/tables/tta_score_contrasts.csv`, `neural_augmentation_15seed.csv`, `negative_control_summary.csv`, `figure_6_augmentation_and_negative_control.pdf` |
| Figures 2–3, Section 3.4 & Appendix D | `X-align` vs. `X-base` and rotated controls (`rot1..rot6`) on NSCLC and CRC | `results/tables/xalign_primary_contrasts.csv`, `xalign_history_depth.csv`, `xalign_fine_history_depth.csv`, `figure_2_xalign_hospital_forest.pdf`, `figure_3_xalign_history_depth.pdf` |
| Section 3.5 & Appendix F | GlucoBench Colas and CGMacros 7-offset and weekday-preserving glucose audits | `results/tables/glucose_audit_summary.csv`, `glucose_offset_rmse.csv`, `cgmacros_window_audit.csv`, `figure_4_glucose_audit.pdf` |
| Appendix H (Table H1) | Hospital-level and overall worst-pair predictive distribution TV | `results/tables/table_h1_patient_tv_summary.csv` |

## License

Released under the [MIT License](LICENSE).
