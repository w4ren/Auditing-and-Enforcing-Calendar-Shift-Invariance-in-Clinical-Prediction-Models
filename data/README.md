# Upstream Clinical Datasets

In accordance with upstream data use agreements and the manuscript's data governance statement, **patient-level clinical records are not redistributed in this repository**.

All synthetic invariance tests, the 200-replicate known-nuisance negative control (`results/tables/negative_control_summary.csv`), deterministic figure generation (`scripts/build_figures.py`), and quantitative verification (`scripts/verify_repository.py`) run out of the box using the de-identified aggregate tables in `results/tables/`.

Researchers who obtain the upstream GENIE BPC tables below can run end-to-end cohort construction, neural and `X-align` model training, rotated negative controls, and 7-shift auditing locally via `calendar-shift-audit` and `scripts/run_genie_campaign.py`. For the external continuous glucose monitoring benchmarks, upstream model training follows the published GlucoBench (N-HiTS) and GluFormer repositories, while `calendar_shift_audit.glucose` and `scripts/run_glucose_campaign.py` provide the timestamp canonicalization, synthetic-anchor randomization schedule, and 7-offset forecast displacement and category-crossing audit.

---

## 1. AACR Project GENIE Biopharma Collaborative (BPC) v2.0

- **Cohorts used**: Non-Small Cell Lung Cancer (`NSCLC v2.0-public`, `1,846` released patients -> `1,825` analyzable patients across `DFCI`, `MSK`, `UHN`, `VICC`) and Colorectal Cancer (`CRC v2.0-public`, `1,485` released patients -> `1,471` analyzable patients across `DFCI`, `MSK`, `VICC`).
- **Upstream access**: Available through Synapse under the AACR Project GENIE BPC data use terms (`https://www.synapse.org/genie`, BPC public releases).
- **Expected local release folder layout**:
  ```text
  /path/to/genie_bpc_nsclc_v2/
  ├── patient_level_dataset.csv
  ├── imaging_level_dataset.csv
  └── med_onc_note_level_dataset.csv
  ```
- **Local cohort construction and audit** (generated `.parquet` files inside `data/` are ignored by `.gitignore` to prevent accidental staging of patient-level records):
  ```bash
  calendar-shift-audit build-genie-cohort \
    --release-dir /path/to/genie_bpc_nsclc_v2 \
    --output-parquet data/nsclc_frame.parquet

  python scripts/run_genie_campaign.py \
    --release-dir /path/to/genie_bpc_nsclc_v2 \
    --disease nsclc \
    --family both \
    --seed 3 \
    --output-json results/local_nsclc_seed3.json
  ```

---

## 2. GlucoBench — Colas Continuous Glucose Monitoring Cohort

- **Splits audited**: Primary held-out out-of-distribution split `test_ood` (`21` held-out participants, `9,168` 60-minute forecast windows disjoint from training/validation) and secondary GlucoBench in-distribution split `test` (`176` participants, `19,192` windows with temporal target overlap across fitting partitions).
- **Upstream access**:
  - Primary dataset: Colas et al., *"Detrended Fluctuation Analysis in the prediction of type 2 diabetes mellitus in patients at risk"*, PLOS ONE (2019), `https://doi.org/10.1371/journal.pone.0225817`.
  - Benchmark preprocessing and N-HiTS implementation: Sergazinov et al., *GlucoBench: Curated List of Continuous Glucose Monitoring Datasets with Prediction Benchmarks* (ICLR 2024), `https://github.com/IrinaStatsLab/GlucoBench`.

---

## 3. PhysioNet CGMacros v1.0.0

- **Cohort used**: Held-out `test` split (`44` participants, `430` forecast windows, `5,160` target occurrences).
- **Upstream access**: Gutierrez-Osuna et al., *CGMacros: a scientific dataset for personalized nutrition and diet monitoring*, PhysioNet v1.0.0, `https://doi.org/10.13026/3z8q-x658` (`https://physionet.org/content/cgmacros/1.0.0/`).

### Auditing Saved CGM Forecast Arrays

Given a `.npz` archive of 7-offset model forecasts with arrays `predictions` (`(S, [M,] 7, W, H)` in mg/dL), `targets` (`(W, H)` in mg/dL), `patient_ids` (`(W,)`), and optional `model_names` (`(M,)`), run:

```bash
python scripts/run_glucose_campaign.py \
  --predictions-npz /path/to/cgm_predictions.npz \
  --dataset colas \
  --output-json results/local_cgm_audit.json
```
