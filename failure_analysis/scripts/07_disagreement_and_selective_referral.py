from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from referral import persist_rankings, random_expected_row, referral_scores
from utils import MODEL_SLUGS, binary_metrics, cluster_bootstrap_rate_ci, configure_matplotlib, load_behavior_table, save_figure_source, write_table

configure_matplotlib()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_auto_subset(sub: pd.DataFrame, *, seed: int) -> dict[str, float]:
    metrics = binary_metrics(sub["y_true"], sub["mean_p"])
    pred = (sub["mean_p"] >= 0.5).astype(int)
    confidence = np.maximum(sub["mean_p"], 1 - sub["mean_p"])
    error = pred.ne(sub["y_true"])
    fp = pred.eq(1) & sub["y_true"].eq(0)
    fn = pred.eq(0) & sub["y_true"].eq(1)
    error_ci = cluster_bootstrap_rate_ci(error.astype(float), sub["patient_id"], seed=seed)
    return {
        "auto_n": len(sub),
        "coverage": len(sub),
        "roc_auc": metrics["roc_auc"],
        "accuracy": metrics["accuracy"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "false_positive_rate": float(fp.sum() / max((sub["y_true"] == 0).sum(), 1)),
        "false_negative_rate": float(fn.sum() / max((sub["y_true"] == 1).sum(), 1)),
        "high_confidence_error_rate": float((error & (confidence >= 0.8)).mean()) if len(sub) else np.nan,
        "auto_error_rate": float(error.mean()) if len(sub) else np.nan,
        "auto_error_cluster_ci_low": error_ci[0],
        "auto_error_cluster_ci_high": error_ci[1],
    }


def main() -> None:
    df = load_behavior_table().reset_index(drop=True)
    strategies = referral_scores(df, "LUNA25")
    orders = persist_rankings(df, strategies, "LUNA25")
    rows = []
    for strategy, score in strategies.items():
        order = orders[strategy]
        for pct in config.REFERRAL_PCTS:
            k = int(math.ceil(len(df) * pct / 100.0))
            referred_idx = set(order[:k])
            auto = df.loc[[i for i in df.index if i not in referred_idx]].copy()
            metrics = evaluate_auto_subset(auto, seed=config.RANDOM_SEED + len(rows) + k)
            rows.append(
                {
                    "referral_strategy": strategy,
                    "label_free": strategy != "hard_case_score",
                    "statistic": "selected_subset",
                    "referral_pct": pct,
                    "referred_n": k,
                    "auto_n": len(auto),
                    "coverage": len(auto) / len(df),
                    **{k2: v for k2, v in metrics.items() if k2 not in ["coverage", "auto_n"]},
                }
            )
    rows.extend(random_expected_row(df, pct) for pct in config.REFERRAL_PCTS)
    table = pd.DataFrame(rows)
    write_table(table, config.TABLE_DIR / "table5_disagreement_selective_referral.csv")

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    bins = np.linspace(0, max(df["std_p"].max(), 1e-6), 20)
    ax.hist(df.loc[~df["ensemble_error"], "std_p"], bins=bins, alpha=0.65, label="ensemble correct", color="#59a14f")
    ax.hist(df.loc[df["ensemble_error"], "std_p"], bins=bins, alpha=0.65, label="ensemble wrong", color="#e15759")
    ax.set_xlabel("Across-model probability SD")
    ax.set_ylabel("Case count")
    ax.set_title("Model disagreement distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_dist = config.FIGURE_DIR / "extra_model_disagreement_distribution.png"
    fig.savefig(out_dist, bbox_inches="tight")
    fig.savefig(out_dist.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    save_figure_source(df[["case_id", "std_p", "max_p_gap", "vote_entropy", "hard_case_score", "ensemble_error"]], out_dist.name)

    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=160)
    plot_strategies = [
        "ensemble_margin",
        "STU-Net_margin",
        "EfficientNet-B0_margin",
        "vote_entropy",
        "random_expected",
        "hard_case_score",
    ]
    for strategy in plot_strategies:
        sub = table[table["referral_strategy"].eq(strategy)].sort_values("coverage")
        ax.plot(sub["coverage"], sub["auto_error_rate"], marker="o", linewidth=1.5, label=strategy)
    ax.set_xlabel("Auto-handled coverage")
    ax.set_ylabel("Auto-handled error rate")
    ax.set_title("Risk-coverage under selective referral")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_curve = config.FIGURE_DIR / "fig4_risk_coverage_curve.png"
    fig.savefig(out_curve, bbox_inches="tight")
    fig.savefig(out_curve.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    save_figure_source(table, out_curve.name)
    print("Wrote disagreement/referral table and figures.")


if __name__ == "__main__":
    main()
