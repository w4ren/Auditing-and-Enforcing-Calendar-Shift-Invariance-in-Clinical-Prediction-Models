"""Verify repository numerical tables, live model invariants, and figure outputs against the paper."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from calendar_shift_audit.cli import run_synthetic_audit
from calendar_shift_audit.models import (
    ALL_NEURAL_ARMS,
    MatchedNeuralConfig,
    build_matched_model,
    expected_parameter_count,
    verify_all_parameters_active,
)


def assert_close(label: str, actual: float, expected: float, tol: float = 1e-4) -> None:
    if not math.isclose(float(actual), float(expected), abs_tol=tol, rel_tol=0.0):
        raise AssertionError(
            f"{label}: expected {expected:.8f}, observed {float(actual):.8f} (tol={tol})"
        )


def verify_cohort_counts(tables_dir: Path) -> int:
    cohort = json.loads((tables_dir / "cohort_summary.json").read_text(encoding="utf-8"))
    checks = 0

    nsclc = cohort["nsclc"]
    nsclc_excl = {row["institution"]: row for row in nsclc["exclusions"]}["ALL"]
    nsclc_flow = {r["reason"]: r["excluded"] for r in nsclc_excl["sequential_flow"]}
    assert nsclc_excl["release_patients"] == 1846
    assert nsclc_flow["no_valid_assessment_day"] == 3
    assert nsclc_flow["only_one_distinct_assessment_day"] == 18
    assert nsclc["counts"]["patients"] == 1825
    assert nsclc["counts"]["completed_intervals"] == 47556
    assert nsclc["counts"]["terminal_intervals"] == 1264
    assert nsclc["counts"]["intervals"] == 48820
    checks += 7

    expected_nsclc_sites = {
        "DFCI": (682, 15529, 455),
        "MSK": (864, 24831, 597),
        "UHN": (55, 1994, 34),
        "VICC": (224, 5202, 178),
    }
    for site, (pat, comp, term) in expected_nsclc_sites.items():
        block = nsclc["counts"]["by_site"][site]
        assert block["patients"] == pat
        assert block["completed_intervals"] == comp
        assert block["terminal_intervals"] == term
        checks += 3

    crc = cohort["crc"]
    crc_excl = {row["institution"]: row for row in crc["exclusions"]}["ALL"]
    crc_flow = {r["reason"]: r["excluded"] for r in crc_excl["sequential_flow"]}
    assert crc_excl["release_patients"] == 1485
    assert crc_flow["no_valid_assessment_day"] == 1
    assert crc_flow["only_one_distinct_assessment_day"] == 13
    assert crc["counts"]["patients"] == 1471
    assert crc["counts"]["completed_intervals"] == 47448
    assert crc["counts"]["terminal_intervals"] == 969
    assert crc["counts"]["intervals"] == 48417
    checks += 7

    expected_crc_sites = {
        "DFCI": (557, 16241, 321),
        "MSK": (689, 24382, 497),
        "VICC": (225, 6825, 151),
    }
    for site, (pat, comp, term) in expected_crc_sites.items():
        block = crc["counts"]["by_site"][site]
        assert block["patients"] == pat
        assert block["completed_intervals"] == comp
        assert block["terminal_intervals"] == term
        checks += 3

    return checks


def verify_live_architecture() -> int:
    checks = 0
    cfg = MatchedNeuralConfig(arm="Calendar-blind", hidden=64, embed=16, weeks=60)
    assert expected_parameter_count(19, cfg) == 4356
    checks += 1

    for arm in ALL_NEURAL_ARMS:
        model = build_matched_model(arm, n_features=19, seed=3)
        assert model.n_parameters() == 4356
        grad_info = verify_all_parameters_active(model, seed=3)
        assert grad_info["all_active"] is True
        checks += 2

    synth = run_synthetic_audit()
    assert synth["parameter_counts"] == {arm: 4356 for arm in ALL_NEURAL_ARMS}
    assert synth["neural_max_shift_delta"]["Calendar-blind"] == 0.0
    assert synth["neural_max_shift_delta"]["Flexible-elapsed"] == 0.0
    assert synth["neural_max_shift_delta"]["Relative-phase"] < 1e-5
    assert synth["neural_max_shift_delta"]["Absolute-phase"] > 1e-3
    assert synth["xalign_max_shift_delta"] < 1e-10
    checks += 6
    return checks


def verify_neural_tables(tables_dir: Path) -> int:
    checks = 0
    t1 = pd.read_csv(tables_dir / "table1_neural_summary.csv").set_index("arm")

    expected_t1 = {
        "Calendar-blind": (-4.577625, -4.635595, -4.521930, 35.66375, 0.0, 0.0, 0.0),
        "Flexible-elapsed": (-4.448876, -4.510417, -4.388896, 35.75146, 0.0, 0.0, 0.0),
        "Absolute-phase": (-4.390121, -4.453454, -4.328706, 35.84867, 0.097634, 0.002972, 0.104618),
        "Relative-phase": (-4.376936, -4.440731, -4.315118, 35.74634, 0.0, 0.0, 0.0),
    }
    for arm, (ls, lo, hi, mae, pr, pl, tv) in expected_t1.items():
        row = t1.loc[arm]
        assert int(row["parameters"]) == 4356
        assert_close(f"{arm} log_score", row["log_score_estimate"], ls, 1e-5)
        assert_close(f"{arm} ci_lower", row["log_score_ci_lower"], lo, 1e-5)
        assert_close(f"{arm} ci_upper", row["log_score_ci_upper"], hi, 1e-5)
        assert_close(f"{arm} mae", row["mae_days_estimate"], mae, 1e-4)
        assert_close(f"{arm} patient_range", row["patient_shift_range_estimate"], pr, 1e-5)
        assert_close(f"{arm} pooled_range", row["pooled_shift_range_estimate"], pl, 1e-5)
        assert_close(f"{arm} patient_tv", row["patient_worst_pair_tv_estimate"], tv, 1e-5)
        checks += 8

    a1 = pd.read_csv(tables_dir / "table_a1_neural_contrasts.csv")
    a1_ov = a1[(a1["scope"] == "overall") & (a1["metric"] == "nats_per_gap")].set_index("contrast")
    expected_contrasts = {
        "Absolute-phase minus Calendar-blind": (0.187504, 0.166924, 0.208683),
        "Relative-phase minus Calendar-blind": (0.200688, 0.180553, 0.221484),
        "Flexible-elapsed minus Calendar-blind": (0.128749, 0.111549, 0.146717),
        "Relative-phase minus Absolute-phase": (0.013184, 0.008587, 0.017640),
        "Relative-phase minus Flexible-elapsed": (0.071940, 0.060950, 0.082666),
        "Absolute-phase minus Flexible-elapsed": (0.058755, 0.047541, 0.069921),
    }
    for cname, (est, lo, hi) in expected_contrasts.items():
        row = a1_ov.loc[cname]
        assert_close(cname, row["estimate"], est, 1e-5)
        assert_close(f"{cname} lo", row["ci_lower"], lo, 1e-5)
        assert_close(f"{cname} hi", row["ci_upper"], hi, 1e-5)
        checks += 3

    a2 = pd.read_csv(tables_dir / "table_a2_neural_by_hospital.csv")
    abs_ranges = a2[
        (a2["arm"] == "Absolute-phase")
        & (a2["metric"] == "nats_per_gap")
        & (a2["kind"] == "patient_specific_offset_range")
    ].set_index("site")
    for site, val in [("DFCI", 0.105899), ("MSK", 0.135751), ("UHN", 0.066170), ("VICC", 0.082715)]:
        assert_close(f"Absolute-phase range {site}", abs_ranges.loc[site, "estimate"], val, 1e-5)
        checks += 1

    nhist = pd.read_csv(tables_dir / "neural_history_depth.csv")
    nh_ov = nhist[
        (nhist["scope"] == "overall")
        & (nhist["contrast"] == "Relative-phase minus Absolute-phase")
    ].set_index("group")
    for grp, val in [("0", 0.005607), ("1-2", 0.011957), ("3-5", 0.011947), ("6+", 0.014610)]:
        assert_close(f"neural history {grp}", nh_ov.loc[grp, "estimate"], val, 1e-5)
        checks += 1

    pooled = pd.read_csv(tables_dir / "neural_pooled_shift_audit.csv")
    ind_mean = pooled[
        (pooled["kind"] == "independent_mean_paired_difference")
        & (pooled["arm"] == "Relative-phase")
        & (pooled["comparator"] == "Absolute-phase")
    ].iloc[0]
    assert_close("64-assignment Relative minus Absolute", ind_mean["estimate"], 0.012809, 1e-5)
    checks += 1

    inv = pd.read_csv(tables_dir / "neural_invariance_checks.csv")
    assert len(inv) == 240
    assert (inv["parameters"] == 4356).all()
    assert inv["all_pass"].all()
    checks += 3

    ecdf = pd.read_csv(tables_dir / "neural_patient_score_ecdf.csv")
    assert len(ecdf) == 201
    assert float(ecdf["pooled_fraction_le"].iloc[0]) == 0.0
    assert float(ecdf["pooled_fraction_le"].iloc[-1]) == 1.0
    checks += 3

    quant = pd.read_csv(tables_dir / "neural_patient_score_quantiles.csv")
    q_med = quant[quant["quantile"] == 0.50].set_index("site")
    for site, val in [("DFCI", 0.084352), ("MSK", 0.113439), ("UHN", 0.060182), ("VICC", 0.076113)]:
        assert_close(f"quantile p50 {site}", q_med.loc[site, "score_range_nats"], val, 1e-5)
        checks += 1

    ens = pd.read_csv(tables_dir / "figure1_ensemble_tv_curve.csv")
    ens_ov = ens[(ens["scope"] == "overall") & (ens["metric"] == "calendar_expected_nonzero")].set_index("ensemble_size")
    for m, val in [(1, 0.073383), (3, 0.058545), (5, 0.054738), (7, 0.052977), (15, 0.050538)]:
        assert_close(f"ensemble M={m} calendar TV", ens_ov.loc[m, "mean_tv"], val, 1e-5)
        checks += 1

    disj = ens[(ens["scope"] == "overall") & (ens["metric"] == "between_disjoint_ensembles")].set_index("ensemble_size")
    for m, val in [(1, 0.138435), (3, 0.081375), (5, 0.062359), (7, 0.052082)]:
        assert_close(f"ensemble M={m} disjoint seed TV", disj.loc[m, "mean_tv"], val, 1e-5)
        checks += 1

    ens_c = pd.read_csv(tables_dir / "ensemble_tv_contrasts.csv")
    m7_diff = ens_c[
        (ens_c["scope"] == "overall") & (ens_c["ensemble_size"] == 7) & (ens_c["operation"] == "difference")
    ].iloc[0]
    assert_close("M=7 calendar minus seed TV", m7_diff["estimate"], 0.000895, 1e-5)
    checks += 1

    h1 = pd.read_csv(tables_dir / "table_h1_patient_tv_summary.csv")
    h1_prim = h1[(h1["population"] == "completed") & (h1["arm"] == "Absolute-phase")].set_index("site")
    for site, val in [("ALL", 0.104618), ("DFCI", 0.107285), ("MSK", 0.136951), ("UHN", 0.086767), ("VICC", 0.087469)]:
        assert_close(f"Table H1 TV {site}", h1_prim.loc[site, "mean_tv"], val, 1e-5)
        checks += 1
    assert_close("Table H1 patient median TV", h1_prim.loc["ALL", "median_seed_averaged_patient_tv"], 0.117044, 1e-5)
    assert_close("Table H1 patient p90 TV", h1_prim.loc["ALL", "p90_seed_averaged_patient_tv"], 0.166786, 1e-5)
    checks += 2

    tv_comp_path = tables_dir / "tv_companion_summary.csv"
    if tv_comp_path.exists():
        tv_comp = pd.read_csv(tv_comp_path)
        assert len(tv_comp) > 0
        checks += 1

    tta = pd.read_csv(tables_dir / "tta_score_contrasts.csv")
    tta_ov = tta[tta["scope"] == "overall"].set_index("contrast")
    assert_close("TTA minus Absolute", tta_ov.loc["Absolute-TTA minus Absolute-phase", "mean_difference"], 0.011926, 1e-5)
    assert_close("TTA minus Relative", tta_ov.loc["Absolute-TTA minus Relative-phase", "mean_difference"], -0.001258, 1e-5)
    checks += 2

    aug = pd.read_csv(tables_dir / "neural_augmentation_15seed.csv")
    aug_all = aug[aug["site"] == "ALL"].set_index(["contrast", "metric"])
    assert_close("Augmented minus Absolute score", aug_all.loc[("Augmented-phase minus Absolute-phase", "released_score"), "estimate"], 0.011107, 1e-5)
    assert_close("Augmented patient shift range", aug_all.loc[("Augmented-phase", "patient_shift_range"), "estimate"], 0.023974, 1e-5)
    checks += 2

    imp = pd.read_csv(tables_dir / "neural_forecast_impact.csv")
    ind_mode = imp[
        (imp["model"] == "Individual Absolute models (15 seeds)")
        & (imp["scope"] == "overall")
        & (imp["metric"] == "mode_changed_any_offset")
    ].iloc[0]
    ens_mode = imp[
        (imp["model"] == "15-model Absolute probability ensemble")
        & (imp["scope"] == "overall")
        & (imp["metric"] == "mode_changed_any_offset")
    ].iloc[0]
    assert_close("Individual mode change fraction", ind_mode["estimate"], 0.328366, 1e-5)
    assert_close("Ensemble mode change fraction", ens_mode["estimate"], 0.241989, 1e-5)
    checks += 2

    lh = pd.read_csv(tables_dir / "neural_long_history_sensitivity.csv")
    lh_all = lh[
        (lh["site"] == "ALL")
        & (lh["model_or_contrast"] == "Relative-phase minus Absolute-phase")
        & (lh["metric"] == "released_log_score")
    ].set_index("group")
    for grp, val in [
        ("all_completed_reference", 0.013184),
        ("patients_ge8_completed", 0.012459),
        ("patients_ge12_completed", 0.014484),
        ("landmark_ge6_prior", 0.014610),
    ]:
        assert_close(f"long history {grp}", lh_all.loc[grp, "estimate"], val, 1e-5)
        checks += 1

    lh_hosp = pd.read_csv(tables_dir / "neural_hospital_exclusion_sensitivity.csv")
    no_msk = lh_hosp[(lh_hosp["scope"] == "EXCLUDING_MSK") & (lh_hosp["offset"].astype(str) == "0")].iloc[0]
    assert_close("Excluding MSK Relative minus Absolute", no_msk["estimate"], 0.009518, 1e-5)
    checks += 1

    nc = pd.read_csv(tables_dir / "negative_control_summary.csv").set_index(["training_n", "model", "metric"])
    for n, tv_v, ell_v in [(200, 0.215946, 0.097694), (2000, 0.076096, 0.010702), (20000, 0.024089, 0.001057)]:
        assert_close(f"NC n={n} TV", nc.loc[(n, "absolute_table", "mean_nonzero_shift_tv"), "mean"], tv_v, 1e-5)
        assert_close(f"NC n={n} excess log loss", nc.loc[(n, "absolute_table", "excess_expected_log_loss"), "mean"], ell_v, 1e-5)
        assert_close(f"NC n={n} pooled TV", nc.loc[(n, "pooled_shared", "mean_nonzero_shift_tv"), "mean"], 0.0, 1e-12)
        checks += 3

    return checks


def verify_xalign_and_glucose(tables_dir: Path) -> int:
    checks = 0
    xc = pd.read_csv(tables_dir / "xalign_primary_contrasts.csv")
    prim = xc[
        (xc["stratum"] == "completed_gap_mean_within_patient")
        & (xc["contrast"] == "X-align minus X-base")
    ]

    nsclc_eq = prim[(prim["disease"] == "nsclc") & (prim["aggregation"] == "site_equal")].iloc[0]
    crc_eq = prim[(prim["disease"] == "crc") & (prim["aggregation"] == "site_equal")].iloc[0]
    assert_close("NSCLC X-align gain", nsclc_eq["estimate"], 0.090634, 1e-5)
    assert_close("CRC X-align gain", crc_eq["estimate"], 0.114304, 1e-5)
    checks += 2

    for disease, site, val in [
        ("nsclc", "DFCI", 0.075855),
        ("nsclc", "MSK", 0.107861),
        ("nsclc", "UHN", 0.105517),
        ("nsclc", "VICC", 0.073304),
        ("crc", "DFCI", 0.096919),
        ("crc", "MSK", 0.161465),
        ("crc", "VICC", 0.084529),
    ]:
        row = prim[(prim["disease"] == disease) & (prim["site"] == site)].iloc[0]
        assert_close(f"{disease} {site} X-align gain", row["estimate"], val, 1e-5)
        checks += 1

    rots = xc[
        (xc["stratum"] == "completed_gap_mean_within_patient")
        & (xc["aggregation"] == "site_equal")
        & (xc["contrast"].str.startswith("X-align minus X-align-rot"))
    ]
    assert len(rots) == 12
    assert (rots["ci_lower"] > 0.07).all()
    checks += 2

    xh = pd.read_csv(tables_dir / "xalign_history_depth.csv")
    xh_eq = xh[xh["aggregation"] == "site_equal"].set_index(["disease", "stratum"])
    for dis, strat, val in [
        ("nsclc", "history_0", 0.002875),
        ("nsclc", "history_1-2", 0.000504),
        ("nsclc", "history_3-5", 0.046877),
        ("nsclc", "history_6+", 0.126310),
        ("crc", "history_0", -0.003988),
        ("crc", "history_1-2", 0.018419),
        ("crc", "history_3-5", 0.067413),
        ("crc", "history_6+", 0.146911),
    ]:
        assert_close(f"xalign coarse {dis} {strat}", xh_eq.loc[(dis, strat), "estimate"], val, 1e-5)
        checks += 1

    xfine = pd.read_csv(tables_dir / "xalign_fine_history_depth.csv").set_index(["disease", "history_group"])
    assert_close("xalign fine NSCLC 0", xfine.loc[("NSCLC", "0"), "estimate"], 0.002875, 1e-5)
    assert_close("xalign fine NSCLC 12+", xfine.loc[("NSCLC", "12+"), "estimate"], 0.159304, 1e-5)
    assert_close("xalign fine CRC 0", xfine.loc[("CRC", "0"), "estimate"], -0.003988, 1e-5)
    assert_close("xalign fine CRC 12+", xfine.loc[("CRC", "12+"), "estimate"], 0.153193, 1e-5)
    checks += 4

    xa7 = pd.read_csv(tables_dir / "xalign_seven_shift_audit.csv")
    assert len(xa7) == 56
    assert (xa7["all_interval_max_spread"] == 0.0).all()
    assert xa7["passed"].all()
    checks += 3

    xkap = pd.read_csv(tables_dir / "xalign_kappa_and_sensitivity.csv")
    assert (xkap["three_seed_max_score_diff"] == 0.0).all()
    kap_hosp = xkap[xkap["scope"] == "hospital"].set_index(["disease", "site"])
    assert_close("NSCLC DFCI kappa", kap_hosp.loc[("nsclc", "DFCI"), "fitted_kappa"], 0.868682, 1e-5)
    assert_close("CRC MSK kappa", kap_hosp.loc[("crc", "MSK"), "fitted_kappa"], 1.061354, 1e-5)
    checks += 3

    gl = pd.read_csv(tables_dir / "glucose_audit_summary.csv").set_index(["dataset", "split", "model"])
    colas_orig = gl.loc[("colas", "test_ood", "original")]
    assert_close("Colas original displacement", colas_orig["typical_nonzero_displacement_mgdl"], 44.070615, 1e-4)
    assert_close("Colas original cat change", colas_orig["typical_nonzero_category_change"], 0.563366, 1e-5)
    assert_close("Colas original shift0 RMSE", colas_orig["shift0_pooled_rmse_mgdl"], 10.615724, 1e-4)
    assert_close("Colas original 7-offset mean RMSE", colas_orig["seven_offset_mean_rmse_mgdl"], 44.968150, 1e-4)
    assert_close("Colas original worst RMSE", colas_orig["per_seed_worst_offset_rmse_mgdl"], 73.112582, 1e-4)
    checks += 5

    colas_ra = gl.loc[("colas", "test_ood", "random_anchor")]
    assert_close("Colas random_anchor displacement", colas_ra["typical_nonzero_displacement_mgdl"], 2.759313, 1e-4)
    assert_close("Colas random_anchor cat change", colas_ra["typical_nonzero_category_change"], 0.004206, 1e-5)
    assert_close("Colas random_anchor shift0 RMSE", colas_ra["shift0_pooled_rmse_mgdl"], 12.645039, 1e-4)
    assert_close("Colas random_anchor worst RMSE", colas_ra["per_seed_worst_offset_rmse_mgdl"], 13.339471, 1e-4)
    checks += 4

    colas_can = gl.loc[("colas", "test_ood", "canonical")]
    assert_close("Colas canonical displacement", colas_can["typical_nonzero_displacement_mgdl"], 0.0, 1e-12)
    assert_close("Colas canonical shift0 RMSE", colas_can["shift0_pooled_rmse_mgdl"], 10.615724, 1e-4)
    checks += 2

    cgm_nat = gl.loc[("cgmacros", "test", "native")]
    assert_close("CGMacros native displacement", cgm_nat["typical_nonzero_displacement_mgdl"], 0.694191, 1e-5)
    assert_close("CGMacros native patient cat change", cgm_nat["patient_any_category_change"], 0.196970, 1e-5)
    assert_close("CGMacros native shift0 RMSE", cgm_nat["shift0_pooled_rmse_mgdl"], 13.919332, 1e-4)
    checks += 3

    cgm_rel = gl.loc[("cgmacros", "test", "relative_origin")]
    assert_close("CGMacros relative_origin displacement", cgm_rel["typical_nonzero_displacement_mgdl"], 0.0, 1e-12)
    assert_close("CGMacros relative_origin shift0 RMSE", cgm_rel["shift0_pooled_rmse_mgdl"], 14.443541, 1e-4)
    checks += 2

    gl_off = pd.read_csv(tables_dir / "glucose_offset_rmse.csv")
    colas_off6 = gl_off[
        (gl_off["dataset"] == "Colas")
        & (gl_off["split"] == "test_ood")
        & (gl_off["model"] == "original")
        & (gl_off["offset"] == 6)
    ].iloc[0]
    assert_close("Colas offset 6 pooled RMSE", colas_off6["estimate"], 72.069269, 1e-4)
    cgm_off5 = gl_off[
        (gl_off["dataset"] == "CGMacros")
        & (gl_off["split"] == "test")
        & (gl_off["model"] == "native")
        & (gl_off["offset"] == 5)
    ].iloc[0]
    assert_close("CGMacros offset 5 pooled RMSE", cgm_off5["estimate"], 14.099652, 1e-4)
    checks += 2

    cgm_win = pd.read_csv(tables_dir / "cgmacros_window_audit.csv")
    r7 = cgm_win[
        (cgm_win["analysis"] == "seven_day_shift_grid")
        & (cgm_win["variant"] == "native")
        & (cgm_win["statistic"] == "mean_patient_range")
    ].iloc[0]
    rwk = cgm_win[
        (cgm_win["analysis"] == "weekday_preserving_multiples_of_7")
        & (cgm_win["variant"] == "native")
        & (cgm_win["statistic"] == "mean_patient_range")
    ].iloc[0]
    assert_close("CGMacros 7-day window RMSE range", r7["estimate"], 0.901924, 1e-5)
    assert_close("CGMacros weekday-preserving RMSE range", rwk["estimate"], 0.578820, 1e-5)
    checks += 2

    return checks


def verify_figures(figures_dir: Path) -> int:
    expected_stems = [
        "figure_1_ensemble_tv",
        "figure_2_xalign_hospital_forest",
        "figure_3_xalign_history_depth",
        "figure_4_glucose_audit",
        "figure_5_patient_score_ecdf",
        "figure_6_augmentation_and_negative_control",
    ]
    checks = 0
    for stem in expected_stems:
        pdf_path = figures_dir / f"{stem}.pdf"
        png_path = figures_dir / f"{stem}.png"
        pdf_bytes = pdf_path.read_bytes()
        png_bytes = png_path.read_bytes()
        assert pdf_bytes.startswith(b"%PDF-")
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        for token in (b"Matplotlib", b"/Creator", b"/Producer", b"/CreationDate", b"Software"):
            assert token not in pdf_bytes
            assert token not in png_bytes
        checks += 4
    return checks


def run_verification(repo_root: Path) -> int:
    tables_dir = repo_root / "results" / "tables"
    figures_dir = repo_root / "results" / "figures"
    total = 0
    total += verify_cohort_counts(tables_dir)
    total += verify_live_architecture()
    total += verify_neural_tables(tables_dir)
    total += verify_xalign_and_glucose(tables_dir)
    total += verify_figures(figures_dir)
    return total


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify repository tables, models, and figures.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    args = parser.parse_args()

    total = run_verification(args.repo_root)
    print(f"Verified {total} quantitative and structural checks successfully.")


if __name__ == "__main__":
    main()
