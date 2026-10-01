from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils import (
    binary_metrics,
    cluster_bootstrap_auc_ci,
    load_behavior_table,
    mcnemar_exact_pvalue,
    model_names_with_ensemble,
    probability_column_for_model,
    stratified_bootstrap_ci,
    write_table,
)


def main() -> None:
    df = load_behavior_table()
    rows = []
    for i, model in enumerate(model_names_with_ensemble()):
        p_col = probability_column_for_model(model)
        sub = df[["y_true", p_col, "patient_id"]].dropna()
        metrics = binary_metrics(sub["y_true"], sub[p_col])
        ci = stratified_bootstrap_ci(sub["y_true"], sub[p_col], seed=config.RANDOM_SEED + i)
        cluster_ci = cluster_bootstrap_auc_ci(sub["y_true"], sub[p_col], sub["patient_id"], seed=config.RANDOM_SEED + 100 + i)
        rows.append(
            {
                "model": model,
                "architecture_group": config.MODEL_GROUPS.get(model, "unspecified"),
                "n": len(sub),
                "positive_label_n": int(sub["y_true"].sum()),
                "roc_auc": metrics["roc_auc"],
                "roc_auc_ci_low": ci[0],
                "roc_auc_ci_high": ci[1],
                "patient_cluster_auc_ci_low": cluster_ci[0],
                "patient_cluster_auc_ci_high": cluster_ci[1],
                "pr_auc": metrics["pr_auc"],
                "accuracy": metrics["accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
            }
        )
    perf = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    write_table(perf, config.TABLE_DIR / "table1_model_performance.csv")

    pair_rows = []
    for a, b in itertools.combinations(model_names_with_ensemble(), 2):
        col_a = probability_column_for_model(a)
        col_b = probability_column_for_model(b)
        sub = df[["y_true", col_a, col_b]].dropna()
        pair_rows.append(
            {
                "model_a": a,
                "model_b": b,
                "n": len(sub),
                "mcnemar_exact_p": mcnemar_exact_pvalue(sub["y_true"], sub[col_a], sub[col_b]),
            }
        )
    write_table(pd.DataFrame(pair_rows), config.TABLE_DIR / "mcnemar_pairwise_model_errors.csv")
    print(f"Wrote model performance table with {len(perf)} rows.")


if __name__ == "__main__":
    main()
