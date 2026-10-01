from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils import (
    MODEL_SLUGS,
    binary_metrics,
    configure_matplotlib,
    load_behavior_table,
    model_names_with_ensemble,
    probability_column_for_model,
    save_figure_source,
    write_table,
)

configure_matplotlib()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    df = load_behavior_table()
    rows = []
    subgroup_families = ["size_bin", "density", "age_group", "sex", "smoking_status", "margin", "lobe_or_location"]
    for model in model_names_with_ensemble():
        p_col = probability_column_for_model(model)
        for family in subgroup_families:
            for level, sub in df.groupby(family, dropna=False):
                sub = sub.dropna(subset=[p_col])
                if len(sub) < config.MIN_SUBGROUP_N:
                    continue
                metrics = binary_metrics(sub["y_true"], sub[p_col])
                rows.append(
                    {
                        "model": model,
                        "architecture_group": config.MODEL_GROUPS.get(model, "unspecified"),
                        "subgroup_family": family,
                        "subgroup": str(level),
                        "n": len(sub),
                        "positive_label_n": int(sub["y_true"].sum()),
                        "prevalence": float(sub["y_true"].mean()),
                        "roc_auc": metrics["roc_auc"],
                        "accuracy": metrics["accuracy"],
                        "sensitivity": metrics["sensitivity"],
                        "specificity": metrics["specificity"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                    }
                )
    table = pd.DataFrame(rows).sort_values(["subgroup_family", "model", "subgroup"])
    write_table(table, config.TABLE_DIR / "table6_subgroup_performance.csv")

    arch_rows = []
    for model in config.MODEL_FILES:
        slug = MODEL_SLUGS[model]
        err_col = f"{slug}_error"
        for density, sub in df.groupby("density", dropna=False):
            if len(sub) < config.MIN_SUBGROUP_N:
                continue
            arch_rows.append(
                {
                    "model": model,
                    "architecture_group": config.MODEL_GROUPS.get(model, "unspecified"),
                    "density": str(density),
                    "n": len(sub),
                    "error_rate": float(sub[err_col].fillna(False).mean()),
                }
            )
    arch = pd.DataFrame(arch_rows)
    write_table(arch, config.TABLE_DIR / "architecture_specific_failure_patterns.csv")
    plot = arch.groupby(["architecture_group", "density"], as_index=False)["error_rate"].mean()
    densities = [d for d in ["Solid", "Part-solid", "Ground-glass", "Mixed/other", "Missing", "Not determined"] if d in set(plot["density"])]
    groups = list(plot["architecture_group"].drop_duplicates())
    x = np.arange(len(densities))
    width = 0.8 / max(len(groups), 1)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
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
    out_png = config.FIGURE_DIR / "fig5_architecture_specific_failure_patterns.png"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    save_figure_source(plot, out_png.name)
    print(f"Wrote subgroup performance table ({len(table)} rows) and architecture pattern figure.")


if __name__ == "__main__":
    main()
