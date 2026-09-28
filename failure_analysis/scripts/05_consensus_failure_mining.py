from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils import cluster_bootstrap_rate_ci, configure_matplotlib, fisher_or_chi2_pvalue, load_behavior_table, save_figure_source, write_table, write_text

configure_matplotlib()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def subgroup_rows(df: pd.DataFrame, family: str, values: pd.Series) -> list[dict[str, object]]:
    overall = float(df["consensus_error_majority"].mean())
    out = []
    values = values.fillna("Missing").astype(str)
    for level in sorted(values.unique()):
        mask = values.eq(level)
        n = int(mask.sum())
        if n < config.MIN_SUBGROUP_N:
            continue
        sub = df[mask]
        rest = df[~mask]
        err = int(sub["consensus_error_majority"].sum())
        non_err = n - err
        rest_err = int(rest["consensus_error_majority"].sum())
        rest_non_err = len(rest) - rest_err
        ci = cluster_bootstrap_rate_ci(
            sub["consensus_error_majority"].astype(float),
            sub["patient_id"],
            seed=config.RANDOM_SEED + n,
        )
        rate = err / n if n else np.nan
        hce_rate = float(sub["high_confidence_error_any_0_8"].mean())
        pvalue = fisher_or_chi2_pvalue(err, non_err, rest_err, rest_non_err)
        out.append(
            {
                "subgroup_family": family,
                "subgroup": level,
                "n": n,
                "malignant_n": int(sub["y_true"].sum()),
                "prevalence": float(sub["y_true"].mean()),
                "consensus_error_rate": rate,
                "consensus_error_ci_low": ci[0],
                "consensus_error_ci_high": ci[1],
                "high_confidence_error_rate": hce_rate,
                "mean_disagreement_score": float(sub["std_p"].mean()),
                "enrichment_ratio": rate / overall if overall else np.nan,
                "p_value": pvalue,
                "exploratory": int(sub["y_true"].sum()) < config.EXPLORATORY_MALIGNANT_N,
            }
        )
    return out


def pretty_condition(feature: str, operator: str, threshold: float) -> str:
    labels = [
        "sex",
        "density",
        "margin",
        "lobe_or_location",
        "smoking_status",
        "size_bin",
        "age_group",
    ]
    if abs(threshold - 0.5) < 1e-9:
        for prefix in labels:
            stem = f"{prefix}_"
            if feature.startswith(stem):
                level = feature[len(stem):]
                return f"{prefix} is {level}" if operator == ">" else f"{prefix} is not {level}"
    return f"{feature} {operator} {threshold:.1f}"


def leaf_paths(tree, feature_names: list[str]) -> dict[int, list[str]]:
    paths: dict[int, list[str]] = {}

    def walk(node: int, conditions: list[str]) -> None:
        left = tree.children_left[node]
        right = tree.children_right[node]
        if left == right:
            paths[node] = conditions
            return
        feature = feature_names[tree.feature[node]]
        threshold = float(tree.threshold[node])
        walk(left, conditions + [pretty_condition(feature, "<=", threshold)])
        walk(right, conditions + [pretty_condition(feature, ">", threshold)])

    walk(0, [])
    return paths


def fit_tree(df: pd.DataFrame) -> str:
    y = df["consensus_error_majority"].astype(int)
    feature_cols = [
        "age",
        "size_mm",
        "perpendicular_size_mm",
        "current_smoker",
        "spiculated_or_irregular",
        "upper_lobe",
        "sex",
        "density",
        "margin",
        "lobe_or_location",
        "smoking_status",
        "size_bin",
        "age_group",
    ]
    X = df[feature_cols].copy()
    for col in ["current_smoker", "spiculated_or_irregular", "upper_lobe"]:
        X[col] = X[col].astype(float)
    X = pd.get_dummies(X, columns=["sex", "density", "margin", "lobe_or_location", "smoking_status", "size_bin", "age_group"], dummy_na=False)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    min_leaf = max(10, int(len(df) * 0.03))
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=min_leaf, class_weight="balanced", random_state=config.RANDOM_SEED)
    clf.fit(X, y)
    importances = pd.DataFrame({"feature": X.columns, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
    write_table(importances, config.TABLE_DIR / "consensus_failure_tree_feature_importance.csv")
    leaves = clf.apply(X)
    paths = leaf_paths(clf.tree_, list(X.columns))
    overall = float(y.mean())
    rule_rows = []
    for leaf_id, conditions in paths.items():
        mask = leaves == leaf_id
        support = int(mask.sum())
        if support == 0:
            continue
        sub = df.loc[mask]
        rate = float(sub["consensus_error_majority"].mean())
        ci = cluster_bootstrap_rate_ci(
            sub["consensus_error_majority"].astype(float),
            sub["patient_id"],
            seed=config.RANDOM_SEED + support + int(leaf_id),
        )
        predicted = int(clf.tree_.value[leaf_id][0].argmax())
        rule_rows.append(
            {
                "rule_id": f"R{len(rule_rows) + 1}",
                "rule": " AND ".join(conditions) if conditions else "All cases",
                "predicted_majority_failure_class": predicted,
                "support_n": support,
                "malignant_n": int(sub["y_true"].sum()),
                "consensus_failure_n": int(sub["consensus_error_majority"].sum()),
                "consensus_failure_rate": rate,
                "consensus_failure_ci_low": ci[0],
                "consensus_failure_ci_high": ci[1],
                "enrichment_ratio": rate / overall if overall else np.nan,
            }
        )
    rule_table = pd.DataFrame(rule_rows).sort_values(
        ["consensus_failure_rate", "support_n"], ascending=[False, False]
    )
    write_table(rule_table, config.TABLE_DIR / "consensus_failure_tree_rules_table.csv")
    return export_text(clf, feature_names=list(X.columns), decimals=3)


def main() -> None:
    df = load_behavior_table()
    rows = []
    subgroup_specs = {
        "size_bin": df["size_bin"],
        "density": df["density"],
        "margin": df["margin"],
        "lobe_or_location": df["lobe_or_location"],
        "age_group": df["age_group"],
        "sex": df["sex"],
        "smoking_status": df["smoking_status"],
        "size_bin + density": df["size_bin"].astype(str) + " | " + df["density"].astype(str),
        "size_bin + smoking": df["size_bin"].astype(str) + " | " + df["smoking_status"].astype(str),
        "density + margin": df["density"].astype(str) + " | " + df["margin"].astype(str),
    }
    for family, values in subgroup_specs.items():
        rows.extend(subgroup_rows(df, family, values))
    table = pd.DataFrame(rows).sort_values(["consensus_error_rate", "n"], ascending=[False, False])
    table["p_value_fdr_bh"] = np.nan
    valid = table["p_value"].notna()
    if valid.any():
        pvals = table.loc[valid, "p_value"].to_numpy(dtype=float)
        order = np.argsort(pvals)
        ranked_p = pvals[order]
        m = len(ranked_p)
        adjusted = ranked_p * m / np.arange(1, m + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.minimum(1.0, adjusted)
        restored = np.empty_like(adjusted)
        restored[order] = adjusted
        table.loc[valid, "p_value_fdr_bh"] = restored
    write_table(table, config.TABLE_DIR / "table3_consensus_failure_subgroups.csv")

    rules = fit_tree(df)
    write_text(config.PAPER_SUMMARY_DIR / "consensus_failure_tree_rules.txt", rules)

    plot_df = table[table["n"] >= 20].head(12).copy()
    plot_df["label"] = plot_df["subgroup_family"] + ": " + plot_df["subgroup"]
    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=160)
    ax.barh(plot_df["label"][::-1], plot_df["consensus_error_rate"][::-1], color="#4c78a8")
    ax.axvline(df["consensus_error_majority"].mean(), color="black", linestyle="--", linewidth=1.0, label="overall")
    ax.set_xlabel("Consensus-error rate")
    ax.set_title("Subgroups enriched for majority DL failure")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_png = config.FIGURE_DIR / "fig2_consensus_failure_by_subgroup.png"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    save_figure_source(plot_df, out_png.name)
    print(f"Wrote consensus failure subgroup table ({len(table)} rows) and decision-tree rules.")


if __name__ == "__main__":
    main()
