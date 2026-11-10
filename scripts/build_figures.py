"""Render publication figures from the aggregate result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "navy": "#1f4e79",
    "teal": "#0072b2",
    "orange": "#d55e00",
    "green": "#009e73",
    "purple": "#cc79a7",
    "amber": "#e69f00",
    "slate": "#4d5656",
    "lightgray": "#d9dee3",
}

PDF_META = {
    "Creator": None,
    "Producer": None,
    "CreationDate": None,
    "Title": None,
    "Author": None,
}
PNG_META = {
    "Software": None,
    "Author": None,
    "Description": None,
    "Title": None,
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "figure.dpi": 150,
        }
    )


def save_dual(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", metadata=PDF_META)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", metadata=PNG_META)
    plt.close(fig)


def plot_figure_1_ensemble_tv(tables_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(tables_dir / "figure1_ensemble_tv_curve.csv")
    overall = df[df["scope"] == "overall"].copy()

    cal = overall[overall["metric"] == "calendar_expected_nonzero"].sort_values("ensemble_size")
    seed_tv = overall[
        (overall["metric"] == "between_disjoint_ensembles") & (overall["estimable"] == True)
    ].sort_values("ensemble_size")

    fig, ax = plt.subplots(figsize=(6.4, 4.1))

    x_seed = seed_tv["ensemble_size"].to_numpy(dtype=float)
    y_seed = 100.0 * seed_tv["mean_tv"].to_numpy(dtype=float)
    lo_seed = 100.0 * seed_tv["ci_lower"].to_numpy(dtype=float)
    hi_seed = 100.0 * seed_tv["ci_upper"].to_numpy(dtype=float)
    ax.plot(
        x_seed,
        y_seed,
        marker="s",
        linewidth=2.0,
        color=PALETTE["slate"],
        linestyle="--",
        label="Disjoint-ensemble seed TV (Absolute-phase)",
    )
    ax.fill_between(x_seed, lo_seed, hi_seed, color=PALETTE["slate"], alpha=0.16)

    x_cal = cal["ensemble_size"].to_numpy(dtype=float)
    y_cal = 100.0 * cal["mean_tv"].to_numpy(dtype=float)
    lo_cal = 100.0 * cal["ci_lower"].to_numpy(dtype=float)
    hi_cal = 100.0 * cal["ci_upper"].to_numpy(dtype=float)
    ax.plot(
        x_cal,
        y_cal,
        marker="o",
        linewidth=2.2,
        color=PALETTE["orange"],
        label="Expected nonzero calendar-shift TV (Absolute-phase)",
    )
    ax.fill_between(x_cal, lo_cal, hi_cal, color=PALETTE["orange"], alpha=0.18)

    ax.axhline(
        0.0,
        color=PALETTE["green"],
        linewidth=2.0,
        linestyle="-",
        label="Relative-phase / Flexible-elapsed (exact 0.00%)",
    )

    ax.set_xticks([1, 3, 5, 7, 15])
    ax.set_xlabel("Deep ensemble size M (probability-space mixture)")
    ax.set_ylabel("Total variation distance (%)")
    ax.set_title("Ensemble Averaging Leaves a Persistent Calendar-Shift Floor")
    ax.set_ylim(-0.6, 15.2)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    fig.tight_layout()
    save_dual(fig, out_dir, "figure_1_ensemble_tv")


def plot_figure_2_xalign_forest(tables_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(tables_dir / "xalign_primary_contrasts.csv")
    sub = df[
        (df["stratum"] == "completed_gap_mean_within_patient")
        & (df["contrast"] == "X-align minus X-base")
        & (df["aggregation"].isin(["within_site_patient_equal", "site_equal"]))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharex=True)
    panels = [
        ("nsclc", "NSCLC (4 Institutions, n = 1,825)", ["DFCI", "MSK", "UHN", "VICC", "site_equal"]),
        ("crc", "CRC (3 Institutions, n = 1,471)", ["DFCI", "MSK", "VICC", "site_equal"]),
    ]

    for ax, (disease, title, order) in zip(axes, panels):
        dsub = sub[sub["disease"] == disease].set_index("site").loc[order].reset_index()
        y_pos = np.arange(len(dsub))[::-1]
        labels = [
            "Equal-hospital mean" if s == "site_equal" else f"{s} (n={int(p)})"
            for s, p in zip(dsub["site"], dsub["patients"])
        ]
        est = dsub["estimate"].to_numpy(dtype=float)
        lo = dsub["ci_lower"].to_numpy(dtype=float)
        hi = dsub["ci_upper"].to_numpy(dtype=float)

        ax.axvline(0.0, color="#666666", linestyle=":", linewidth=1.2)
        for idx, y in enumerate(y_pos):
            is_summary = dsub.loc[idx, "site"] == "site_equal"
            color = PALETTE["navy"] if is_summary else PALETTE["teal"]
            marker = "D" if is_summary else "o"
            lw = 2.3 if is_summary else 1.8
            ax.errorbar(
                est[idx],
                y,
                xerr=[[est[idx] - lo[idx]], [hi[idx] - est[idx]]],
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=lw,
                capsize=3.5,
                markersize=6.5 if is_summary else 5.5,
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Completed-gap log-score gain (nats)")
        ax.set_title(title)
        ax.set_xlim(-0.01, 0.19)

    fig.tight_layout()
    save_dual(fig, out_dir, "figure_2_xalign_hospital_forest")


def plot_figure_3_xalign_history_depth(tables_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(tables_dir / "xalign_fine_history_depth.csv")
    order = [str(i) for i in range(12)] + ["12+"]

    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    specs = [
        ("NSCLC", PALETTE["teal"], "o", "NSCLC (equal-hospital mean, 95% CI)"),
        ("CRC", PALETTE["orange"], "s", "CRC (equal-hospital mean, 95% CI)"),
    ]
    x = np.arange(len(order), dtype=float)

    ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.2)
    for disease, color, marker, label in specs:
        dsub = df[df["disease"] == disease].set_index("history_group").loc[order].reset_index()
        est = dsub["estimate"].to_numpy(dtype=float)
        lo = dsub["ci_low"].to_numpy(dtype=float)
        hi = dsub["ci_high"].to_numpy(dtype=float)
        ax.plot(x, est, marker=marker, linewidth=2.0, color=color, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.16)

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("Number of prior completed imaging intervals")
    ax.set_ylabel("X-align minus X-base log-score gain (nats)")
    ax.set_title("Phase-Alignment Gain Emerges as Longitudinal Cadence Accumulates")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)
    fig.tight_layout()
    save_dual(fig, out_dir, "figure_3_xalign_history_depth")


def plot_figure_4_glucose_audit(tables_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(tables_dir / "glucose_offset_rmse.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))

    colas = df[(df["dataset"] == "Colas") & (df["split"] == "test_ood")].copy()
    ax = axes[0]
    colas_specs = [
        ("original", PALETTE["orange"], "o", "-", "Original calendar encoding"),
        ("random_anchor", PALETTE["purple"], "^", "--", "Random-anchor training"),
        ("canonical", PALETTE["green"], "s", "-", "Canonical patient-relative"),
    ]
    for model, color, marker, ls, label in colas_specs:
        msub = colas[colas["model"] == model].sort_values("offset")
        x = msub["offset"].to_numpy(dtype=float)
        y = msub["estimate"].to_numpy(dtype=float)
        lo = msub["ci_low"].to_numpy(dtype=float)
        hi = msub["ci_high"].to_numpy(dtype=float)
        ax.plot(x, y, marker=marker, linestyle=ls, linewidth=2.0, color=color, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.14)
    ax.set_xticks(range(7))
    ax.set_xlabel("Uniform date shift (days)")
    ax.set_ylabel("Pooled 60-minute forecast RMSE (mg/dL)")
    ax.set_title("A. GlucoBench Colas Held-Out OOD (n = 21)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    cgm = df[(df["dataset"] == "CGMacros") & (df["split"] == "test")].copy()
    ax2 = axes[1]
    cgm_specs = [
        ("native", PALETTE["orange"], "o", "-", "Native calendar encoding"),
        ("relative_origin", PALETTE["green"], "s", "-", "Relative-origin encoding"),
        ("persistence", PALETTE["slate"], "d", ":", "Last-observation persistence"),
    ]
    for model, color, marker, ls, label in cgm_specs:
        msub = cgm[cgm["model"] == model].sort_values("offset")
        x = msub["offset"].to_numpy(dtype=float)
        y = msub["estimate"].to_numpy(dtype=float)
        lo = msub["ci_low"].to_numpy(dtype=float)
        hi = msub["ci_high"].to_numpy(dtype=float)
        ax2.plot(x, y, marker=marker, linestyle=ls, linewidth=2.0, color=color, label=label)
        ax2.fill_between(x, lo, hi, color=color, alpha=0.12)
    ax2.set_xticks(range(7))
    ax2.set_xlabel("Uniform date shift (days)")
    ax2.set_ylabel("Pooled 60-minute forecast RMSE (mg/dL)")
    ax2.set_title("B. CGMacros Held-Out Test (n = 44)")
    ax2.set_ylim(10.5, 18.2)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92)

    fig.tight_layout()
    save_dual(fig, out_dir, "figure_4_glucose_audit")


def plot_figure_5_patient_score_ecdf(tables_dir: Path, out_dir: Path) -> None:
    ecdf = pd.read_csv(tables_dir / "neural_patient_score_ecdf.csv")
    quant = pd.read_csv(tables_dir / "neural_patient_score_quantiles.csv")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))

    ax = axes[0]
    ax.plot(
        ecdf["score_range_nats"],
        ecdf["pooled_fraction_le"],
        linewidth=2.1,
        color=PALETTE["teal"],
        label="Pooled patients (n = 1,825)",
    )
    ax.plot(
        ecdf["score_range_nats"],
        ecdf["hospital_equal_fraction_le"],
        linewidth=2.0,
        linestyle="--",
        color=PALETTE["orange"],
        label="Equal-hospital weighted",
    )
    ax.axvline(
        0.0976337659,
        color=PALETTE["slate"],
        linestyle=":",
        linewidth=1.4,
        label="Equal-hospital mean (0.098 nats)",
    )
    ax.set_xlim(0.0, 0.35)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Patient 7-shift log-score range (nats)")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_title("A. Patient-Level Shift Sensitivity ECDF")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    ax2 = axes[1]
    sites = ["DFCI", "MSK", "UHN", "VICC"]
    y_pos = np.arange(len(sites))[::-1]
    for idx, site in enumerate(sites):
        s_df = quant[quant["site"] == site].set_index("quantile")
        q05 = float(s_df.loc[0.05, "score_range_nats"])
        q25 = float(s_df.loc[0.25, "score_range_nats"])
        q50 = float(s_df.loc[0.50, "score_range_nats"])
        q75 = float(s_df.loc[0.75, "score_range_nats"])
        q95 = float(s_df.loc[0.95, "score_range_nats"])
        y = y_pos[idx]
        ax2.plot([q05, q95], [y, y], color=PALETTE["slate"], linewidth=1.4)
        ax2.plot([q25, q75], [y, y], color=PALETTE["teal"], linewidth=5.5, solid_capstyle="butt")
        ax2.plot(q50, y, marker="o", color="white", markeredgecolor=PALETTE["navy"], markersize=6.0, zorder=4)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sites)
    ax2.set_xlabel("Patient 7-shift log-score range (nats)")
    ax2.set_title("B. Hospital Quantiles (5/25/50/75/95th)")
    ax2.set_xlim(0.0, 0.32)

    fig.tight_layout()
    save_dual(fig, out_dir, "figure_5_patient_score_ecdf")


def plot_figure_6_augmentation_and_negative_control(tables_dir: Path, out_dir: Path) -> None:
    aug = pd.read_csv(tables_dir / "neural_augmentation_15seed.csv")
    nc = pd.read_csv(tables_dir / "negative_control_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9))

    ax = axes[0]
    all_aug = aug[(aug["site"] == "ALL") & (aug["metric"] == "patient_shift_range")].set_index("contrast")
    arms = ["Absolute-phase", "Augmented-phase", "Relative-phase"]
    labels = ["Absolute-phase\n(standard)", "Augmented-phase\n(random shifts)", "Relative-phase\n(equivariant)"]
    colors = [PALETTE["orange"], PALETTE["amber"], PALETTE["green"]]
    vals = [float(all_aug.loc[a, "estimate"]) for a in arms]
    los = [float(all_aug.loc[a, "ci_lower"]) for a in arms]
    his = [float(all_aug.loc[a, "ci_upper"]) for a in arms]
    x_pos = np.arange(len(arms))

    bars = ax.bar(x_pos, vals, width=0.52, color=colors, edgecolor="#333333", linewidth=0.8)
    for idx in range(len(arms)):
        ax.errorbar(
            x_pos[idx],
            vals[idx],
            yerr=[[max(0.0, vals[idx] - los[idx])], [max(0.0, his[idx] - vals[idx])]],
            fmt="none",
            ecolor="#222222",
            elinewidth=1.5,
            capsize=4,
        )
        ax.text(
            x_pos[idx],
            vals[idx] + 0.0045,
            f"{vals[idx]:.3f}" if vals[idx] > 1e-6 else "0.000",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean patient 7-shift score range (nats)")
    ax.set_ylim(0.0, 0.118)
    ax.set_title("A. GENIE NSCLC Shift Augmentation (15 Seeds)")
    _ = bars

    ax2 = axes[1]
    abs_nc = nc[(nc["model"] == "absolute_table") & (nc["metric"] == "excess_expected_log_loss")].sort_values("training_n")
    pool_nc = nc[(nc["model"] == "pooled_shared") & (nc["metric"] == "excess_expected_log_loss")].sort_values("training_n")
    sh_nc = nc[(nc["model"] == "shift_average") & (nc["metric"] == "excess_expected_log_loss")].sort_values("training_n")

    for sub_df, color, marker, ls, label in [
        (abs_nc, PALETTE["orange"], "o", "-", "Absolute phase-stratified table"),
        (pool_nc, PALETTE["green"], "s", "-", "Shift-invariant pooled estimator"),
        (sh_nc, PALETTE["teal"], "^", "--", "7-shift mixture average"),
    ]:
        xn = sub_df["training_n"].to_numpy(dtype=float)
        yn = sub_df["mean"].to_numpy(dtype=float)
        lo_n = sub_df["fit_percentile_2_5"].to_numpy(dtype=float)
        hi_n = sub_df["fit_percentile_97_5"].to_numpy(dtype=float)
        ax2.plot(xn, yn, marker=marker, linestyle=ls, linewidth=2.0, color=color, label=label)
        ax2.fill_between(xn, lo_n, hi_n, color=color, alpha=0.14)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks([200, 2000, 20000])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel("Training sample size N (known-nuisance control)")
    ax2.set_ylabel("Excess expected log loss (nats, log scale)")
    ax2.set_title("B. Synthetic Known-Nuisance Negative Control")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92)

    fig.tight_layout()
    save_dual(fig, out_dir, "figure_6_augmentation_and_negative_control")


def build_all_figures(tables_dir: Path, out_dir: Path) -> list[Path]:
    configure_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_figure_1_ensemble_tv(tables_dir, out_dir)
    plot_figure_2_xalign_forest(tables_dir, out_dir)
    plot_figure_3_xalign_history_depth(tables_dir, out_dir)
    plot_figure_4_glucose_audit(tables_dir, out_dir)
    plot_figure_5_patient_score_ecdf(tables_dir, out_dir)
    plot_figure_6_augmentation_and_negative_control(tables_dir, out_dir)
    return sorted(out_dir.glob("figure_*.*"))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render publication figures into results/figures.")
    parser.add_argument("--tables-dir", type=Path, default=repo_root / "results" / "tables")
    parser.add_argument("--figures-dir", type=Path, default=repo_root / "results" / "figures")
    args = parser.parse_args()

    written = build_all_figures(args.tables_dir, args.figures_dir)
    print(f"Rendered {len(written)} figure files into {args.figures_dir}")


if __name__ == "__main__":
    main()
