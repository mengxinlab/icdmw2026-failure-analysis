from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils import MODEL_SLUGS, configure_matplotlib, load_behavior_table, model_names_with_ensemble, probability_column_for_model, save_figure_source, write_table

configure_matplotlib()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def model_error_frame(df: pd.DataFrame, model: str) -> pd.DataFrame:
    p_col = probability_column_for_model(model)
    sub = df.copy()
    sub["model"] = model
    sub["p_malignant"] = sub[p_col]
    sub["pred"] = (sub["p_malignant"] >= 0.5).astype(int)
    sub["confidence"] = np.maximum(sub["p_malignant"], 1 - sub["p_malignant"])
    sub["error"] = sub["pred"].ne(sub["y_true"])
    sub["false_positive"] = sub["pred"].eq(1) & sub["y_true"].eq(0)
    sub["false_negative"] = sub["pred"].eq(0) & sub["y_true"].eq(1)
    return sub


def main() -> None:
    df = load_behavior_table()
    summary_rows = []
    subgroup_rows = []
    representative = []
    for model in model_names_with_ensemble():
        sub = model_error_frame(df, model).dropna(subset=["p_malignant"])
        row = {
            "model": model,
            "architecture_group": config.MODEL_GROUPS.get(model, "unspecified"),
            "n": len(sub),
            "error_n": int(sub["error"].sum()),
            "error_rate": float(sub["error"].mean()),
        }
        for threshold in [0.8, 0.9]:
            hce = sub["error"] & (sub["confidence"] >= threshold)
            row[f"hce_{threshold}_n"] = int(hce.sum())
            row[f"hce_{threshold}_rate"] = float(hce.mean())
            row[f"fp_conf_ge_{threshold}_n"] = int((sub["false_positive"] & (sub["confidence"] >= threshold)).sum())
            row[f"fn_conf_ge_{threshold}_n"] = int((sub["false_negative"] & (sub["confidence"] >= threshold)).sum())
        summary_rows.append(row)

        for family in ["size_bin", "density"]:
            for level, g in sub.groupby(family, dropna=False):
                if len(g) < config.MIN_SUBGROUP_N:
                    continue
                subgroup_rows.append(
                    {
                        "model": model,
                        "subgroup_family": family,
                        "subgroup": str(level),
                        "n": len(g),
                        "malignant_n": int(g["y_true"].sum()),
                        "hce_0.8_n": int((g["error"] & (g["confidence"] >= 0.8)).sum()),
                        "hce_0.8_rate": float((g["error"] & (g["confidence"] >= 0.8)).mean()),
                        "hce_0.9_n": int((g["error"] & (g["confidence"] >= 0.9)).sum()),
                        "hce_0.9_rate": float((g["error"] & (g["confidence"] >= 0.9)).mean()),
                    }
                )

        reps = sub[sub["error"] & (sub["confidence"] >= 0.8)].sort_values("confidence", ascending=False).head(20)
        representative.extend(
            reps[
                [
                    "model",
                    "case_id",
                    "patient_id",
                    "y_true",
                    "pred",
                    "p_malignant",
                    "confidence",
                    "false_positive",
                    "false_negative",
                    "size_mm",
                    "size_bin",
                    "density",
                    "margin",
                    "lobe_or_location",
                    "age",
                    "sex",
                    "smoking_status",
                ]
            ].to_dict("records")
        )

    table = pd.DataFrame(summary_rows).sort_values("hce_0.8_rate", ascending=False)
    write_table(table, config.TABLE_DIR / "table4_high_confidence_errors.csv")
    write_table(pd.DataFrame(subgroup_rows), config.TABLE_DIR / "high_confidence_errors_by_subgroup.csv")
    write_table(pd.DataFrame(representative), config.TABLE_DIR / "representative_high_confidence_errors.csv")

    plot = table.copy()
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    fp = plot["fp_conf_ge_0.8_n"].to_numpy()
    fn = plot["fn_conf_ge_0.8_n"].to_numpy()
    ax.bar(x, fp, width=0.62, label="false positive", color="#e15759")
    ax.bar(x, fn, bottom=fp, width=0.62, label="false negative", color="#4e79a7")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["model"], rotation=45, ha="right")
    ax.set_ylabel("High-probability error count")
    ax.set_title("High-probability errors by FP/FN type")
    ax.set_ylim(0, max((fp + fn).max() * 1.18, 1))
    ax.legend(frameon=False)
    fig.tight_layout()
    out_png = config.FIGURE_DIR / "fig3_high_confidence_error_rates.png"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    save_figure_source(plot, out_png.name)
    print("Wrote high-probability error tables and figure.")


if __name__ == "__main__":
    main()
