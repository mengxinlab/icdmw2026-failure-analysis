#!/usr/bin/env python3
"""Re-render the manuscript figures with Times New Roman + consecutive numbers.

Run after ``run_all.py``. This final presentation renderer uses exported
source CSVs and aggregate tables, without recomputing statistics. It formats
the manuscript co-error heatmap and five-curve risk-coverage plot with compact
labels, readable column-width text and grayscale-distinguishable line styles.

It depends only on numpy/pandas/matplotlib (not scipy/sklearn), so it runs even
when those compiled libraries are unavailable. Both PNG and vector PDF are
written, and copied to ``../manuscript_icdm/figures/`` only if that directory
exists. The manuscript includes the PDF versions. Local output figure
derivatives are overwritten, but raw inputs and aggregate CSVs are untouched.

File numbering retains historical pipeline names: manuscript Fig. 2 is
``fig1_error_cooccurrence_heatmap.pdf`` and manuscript Fig. 3 is
``fig4_risk_coverage_curve.pdf``. The workflow diagram is authored in LaTeX.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG_SRC = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
MANU_FIG = ROOT.parent / "manuscript_icdm" / "figures"

MODELS = ["STU-Net", "EfficientNet-B0", "ResNet-18", "DenseNet-121", "ResNet-50", "Swin-UNETR", "ViT-Base"]

# Consistent (color, marker, linestyle) per referral strategy, shared across the
# two risk-coverage figures so a strategy looks identical in both.
STRATEGY_STYLE = {
    "ensemble_margin": ("#4c78a8", "o", "-"),
    "STU-Net_margin": ("#f58518", "s", "--"),
    "EfficientNet-B0_margin": ("#54a24b", "^", "-."),
    "vote_entropy": ("#e15759", "D", ":"),
    "random_expected": ("#7f7f7f", "x", (0, (1, 1))),
    "hard_case_score": ("#b279a2", "v", (0, (3, 1, 1, 1))),
}


def set_times_new_roman() -> None:
    found = False
    for name in [
        "Times New Roman.ttf",
        "Times New Roman Bold.ttf",
        "Times New Roman Italic.ttf",
        "Times New Roman Bold Italic.ttf",
    ]:
        p = Path("/System/Library/Fonts/Supplemental") / name
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            found = True
    serif = (["Times New Roman"] if found else []) + ["Times", "DejaVu Serif", "serif"]
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            # Embed real TrueType (Type 42) glyphs, not Type 3 bitmaps, so the
            # figure PDFs pass IEEE PDF checks and stay searchable.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    print("Times New Roman registered:", found)


def _save(fig, name: str) -> None:
    out_png = FIG_SRC / name
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    if MANU_FIG.is_dir():
        shutil.copyfile(out_png, MANU_FIG / name)
        shutil.copyfile(out_png.with_suffix(".pdf"), (MANU_FIG / name).with_suffix(".pdf"))
    print("wrote", name, "(png+pdf)")


def fig_cooccurrence() -> None:
    jac = pd.read_csv(FIG_SRC / "fig1_error_cooccurrence_heatmap_source.csv").set_index("model")
    jac = jac.loc[MODELS, MODELS]
    cooc = pd.read_csv(TABLE_DIR / "error_cooccurrence_matrix.csv").set_index("model").loc[MODELS, MODELS]
    diag_error_counts = np.array([int(cooc.loc[m, m]) for m in MODELS])
    mat = jac.to_numpy(dtype=float)
    np.fill_diagonal(mat, np.nan)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#f2f2f2")
    # Sized for a single IEEE column, so text is readable at final placement.
    fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=200)
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=np.nanmax(mat))
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_yticks(np.arange(len(MODELS)))
    labels = ["STU", "Eff-B0", "Res18", "Dense", "Res50", "Swin", "ViT"]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            if i == j:
                label, color = f"{diag_error_counts[i]}", "black"
            else:
                label = f"{mat[i, j]:.2f}"
                color = "white" if mat[i, j] >= 0.38 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Jaccard", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save(fig, "fig1_error_cooccurrence_heatmap.png")


def fig_subgroups() -> None:
    plot_df = pd.read_csv(FIG_SRC / "fig2_consensus_failure_by_subgroup_source.csv")
    overall = float((plot_df["consensus_error_rate"] / plot_df["enrichment_ratio"]).dropna().iloc[0])
    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=160)
    ax.barh(plot_df["label"][::-1], plot_df["consensus_error_rate"][::-1], color="#4c78a8")
    ax.axvline(overall, color="black", linestyle="--", linewidth=1.0, label="overall")
    ax.set_xlabel("Consensus-error rate")
    ax.set_title("Subgroups enriched for majority DL failure")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig2_consensus_failure_by_subgroup.png")


def fig_high_conf() -> None:
    plot = pd.read_csv(FIG_SRC / "fig3_high_confidence_error_rates_source.csv")
    x = np.arange(len(plot))
    fp = plot["fp_conf_ge_0.8_n"].to_numpy()
    fn = plot["fn_conf_ge_0.8_n"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.bar(x, fp, width=0.62, label="false positive", color="#e15759")
    ax.bar(x, fn, bottom=fp, width=0.62, label="false negative", color="#4e79a7")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["model"], rotation=45, ha="right")
    ax.set_ylabel("High-probability error count")
    ax.set_title("High-probability errors by FP/FN type")
    ax.set_ylim(0, max((fp + fn).max() * 1.18, 1))
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig3_high_confidence_error_rates.png")


def _risk_coverage(source_csv: str, out_name: str, strategies: list[str], title: str) -> None:
    table = pd.read_csv(FIG_SRC / source_csv)
    fig, ax = plt.subplots(figsize=(3.5, 2.45), dpi=200)
    for strategy in strategies:
        color, marker, ls = STRATEGY_STYLE[strategy]
        sub = table[table["referral_strategy"].eq(strategy)].sort_values("coverage")
        ax.plot(sub["coverage"], sub["auto_error_rate"], color=color, marker=marker,
                linestyle=ls, linewidth=1.2, markersize=3,
                label={"ensemble_margin":"Ensemble margin", "STU-Net_margin":"STU-Net margin",
                       "vote_entropy":"Vote entropy", "random_expected":"Random expectation",
                       "hard_case_score":"Label-informed heuristic"}.get(strategy,strategy))
    ax.set_xlabel("Auto-handled coverage", fontsize=9)
    ax.set_ylabel("Auto-handled error", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    _save(fig, out_name)


def fig_risk_coverage() -> None:
    _risk_coverage(
        "fig4_risk_coverage_curve_source.csv",
        "fig4_risk_coverage_curve.png",
        ["ensemble_margin", "STU-Net_margin", "vote_entropy", "random_expected", "hard_case_score"],
        "Risk-coverage under selective referral",
    )


def fig_architecture() -> None:
    plot = pd.read_csv(FIG_SRC / "fig5_architecture_specific_failure_patterns_source.csv")
    densities = [d for d in ["Solid", "Part-solid", "Ground-glass", "Mixed/other", "Missing", "Not determined"] if d in set(plot["density"])]
    groups = list(plot["architecture_group"].drop_duplicates())
    x = np.arange(len(densities))
    width = 0.8 / max(len(groups), 1)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    for i, group in enumerate(groups):
        vals = []
        for density in densities:
            match = plot[plot["architecture_group"].eq(group) & plot["density"].eq(density)]
            vals.append(float(match["error_rate"].iloc[0]) if len(match) else np.nan)
        ax.bar(x + (i - (len(groups) - 1) / 2) * width, vals, width=width, label=group, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(densities, rotation=35, ha="right")
    ax.set_ylabel("Mean model error rate")
    ax.set_title("Architecture-specific failure patterns by density")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig5_architecture_specific_failure_patterns.png")


def fig_external() -> None:
    plot = pd.read_csv(FIG_SRC / "fig6_external_shift_if_available_source.csv")
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(9, 5.0), dpi=160)
    ax.bar(x - 0.18, plot["internal_auc"], width=0.36, label="LUNA25 internal", color="#4c78a8")
    ax.bar(x + 0.18, plot["external_auc"], width=0.36, label="LNDb external", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["model"], rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Internal-to-external performance shift")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig6_external_shift_if_available.png")


def fig_lndb_risk_coverage() -> None:
    _risk_coverage(
        "fig7_lndb_risk_coverage_curve_source.csv",
        "fig7_lndb_risk_coverage_curve.png",
        ["ensemble_margin", "STU-Net_margin", "vote_entropy", "random_expected"],
        "LNDb risk-coverage under label-free referral",
    )


def main() -> None:
    set_times_new_roman()
    fig_cooccurrence()
    fig_subgroups()
    fig_high_conf()
    fig_risk_coverage()
    fig_architecture()
    fig_external()
    fig_lndb_risk_coverage()
    print("done")


if __name__ == "__main__":
    main()
