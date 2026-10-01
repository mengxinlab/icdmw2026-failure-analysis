from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from referral import persist_rankings, random_expected_row, referral_order, referral_scores
from utils import binary_metrics, configure_matplotlib, save_figure_source, vote_entropy, write_table

configure_matplotlib()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_REFERRAL_BOOTSTRAPS = 500


def load_lndb_model(model: str) -> pd.DataFrame:
    path = config.LNDB_DL_DIR / config.LNDB_MODEL_FILES[model]
    df = pd.read_csv(path)
    required = ["FindingID", "label", "pred_prob"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    if df["FindingID"].isna().any() or df["FindingID"].astype(str).str.strip().eq("").any():
        raise ValueError(f"Missing or blank finding ID in {path}")
    out = pd.DataFrame(
        {
            "finding_id": df["FindingID"].astype(str),
            "y_true": pd.to_numeric(df["label"], errors="coerce"),
            model: pd.to_numeric(df["pred_prob"], errors="coerce"),
        }
    )
    if out["finding_id"].eq("").any() or not out["y_true"].isin([0, 1]).all() or not out[model].between(0, 1).all():
        raise ValueError(f"Invalid ID, binary label, or probability in {path}")
    # Repeated evaluation-sheet rows may collapse only when values agree exactly.
    conflicts = out.groupby("finding_id")[["y_true", model]].nunique(dropna=False).gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError(f"Conflicting duplicate findings in {path}: {conflicts[conflicts].index.tolist()[:5]}")
    return out.drop_duplicates("finding_id", keep="first")


def missing_external() -> bool:
    return not all((config.LNDB_DL_DIR / fn).exists() for fn in config.LNDB_MODEL_FILES.values())


def write_placeholder(note: str) -> None:
    table = pd.DataFrame([{"analysis": "LNDb external validation", "status": "skipped", "note": note}])
    write_table(table, config.TABLE_DIR / "table7_external_validation_if_available.csv")
    write_table(table, config.TABLE_DIR / "table8_lndb_selective_referral.csv")
    fig, ax = plt.subplots(figsize=(7, 3), dpi=160)
    ax.text(0.5, 0.5, note, ha="center", va="center", wrap=True)
    ax.axis("off")
    out = config.FIGURE_DIR / "fig6_external_shift_if_available.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def add_lndb_behavior_features(base: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    out = base.copy()
    p_mat = out[model_cols].astype(float)
    out["model_count"] = p_mat.notna().sum(axis=1)
    out["mean_p"] = p_mat.mean(axis=1)
    out["std_p"] = p_mat.std(axis=1, ddof=0)
    out["max_p_gap"] = p_mat.max(axis=1) - p_mat.min(axis=1)
    vote_mat = p_mat.ge(0.5)
    out["vote_malignant_count"] = vote_mat.sum(axis=1).astype(int)
    out["vote_entropy"] = [vote_entropy(v, n) for v, n in zip(out["vote_malignant_count"], out["model_count"])]
    out["ensemble_pred"] = out["mean_p"].ge(0.5).astype(int)
    out["ensemble_error"] = out["ensemble_pred"].ne(out["y_true"].astype(int))
    out["ensemble_margin"] = (out["mean_p"] - 0.5).abs()
    for model in model_cols:
        out[f"{model}_margin"] = (out[model] - 0.5).abs()
    return out


def evaluate_auto_subset(sub: pd.DataFrame) -> dict[str, float]:
    metrics = binary_metrics(sub["y_true"], sub["mean_p"])
    pred = sub["mean_p"].ge(0.5).astype(int)
    error = pred.ne(sub["y_true"].astype(int))
    fp = pred.eq(1) & sub["y_true"].eq(0)
    fn = pred.eq(0) & sub["y_true"].eq(1)
    return {
        "auto_n": len(sub),
        "roc_auc": metrics["roc_auc"],
        "accuracy": metrics["accuracy"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "false_positive_rate": float(fp.sum() / max((sub["y_true"] == 0).sum(), 1)),
        "false_negative_rate": float(fn.sum() / max((sub["y_true"] == 1).sum(), 1)),
        "auto_error_rate": float(error.mean()) if len(sub) else np.nan,
    }


def evaluate_auto_rates(sub: pd.DataFrame) -> dict[str, float]:
    pred = sub["mean_p"].ge(0.5).astype(int)
    y = sub["y_true"].astype(int)
    error = pred.ne(y)
    tp = int(pred.eq(1).mul(y.eq(1)).sum())
    tn = int(pred.eq(0).mul(y.eq(0)).sum())
    pos = int(y.eq(1).sum())
    neg = int(y.eq(0).sum())
    return {
        "auto_error_rate": float(error.mean()) if len(sub) else np.nan,
        "sensitivity": tp / pos if pos else np.nan,
        "specificity": tn / neg if neg else np.nan,
    }


def bootstrap_referral_intervals(
    base: pd.DataFrame,
    score: pd.Series,
    pct: int,
    *,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pos_idx = base.index[base["y_true"].eq(1)].to_numpy()
    neg_idx = base.index[base["y_true"].eq(0)].to_numpy()
    values = {"auto_error_rate": [], "sensitivity": [], "specificity": []}
    for _ in range(N_REFERRAL_BOOTSTRAPS):
        boot_idx = np.concatenate(
            [
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        sample = base.loc[boot_idx].reset_index(drop=True)
        sample_score = pd.Series(score.loc[boot_idx].to_numpy(), index=sample.index)
        k = int(math.ceil(len(sample) * pct / 100.0))
        referred_idx = set(referral_order(sample, sample_score, "LNDb")[:k])
        auto = sample.loc[[i for i in sample.index if i not in referred_idx]]
        rates = evaluate_auto_rates(auto)
        for key, value in rates.items():
            values[key].append(value)

    out = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        out[f"{key}_boot_ci_low"] = float(np.nanquantile(arr, 0.025))
        out[f"{key}_boot_ci_high"] = float(np.nanquantile(arr, 0.975))
    return out


def write_lndb_referral_outputs(base: pd.DataFrame, model_cols: list[str]) -> None:
    strategies = referral_scores(base, "LNDb")
    orders = persist_rankings(base, strategies, "LNDb")
    rows = []
    for strategy, score in strategies.items():
        order = orders[strategy]
        for pct in config.REFERRAL_PCTS:
            k = int(math.ceil(len(base) * pct / 100.0))
            referred_idx = set(order[:k])
            auto = base.loc[[i for i in base.index if i not in referred_idx]].copy()
            ci = bootstrap_referral_intervals(
                base,
                score,
                pct,
                seed=config.RANDOM_SEED + 9000 + len(rows),
            )
            rows.append(
                {
                    "referral_strategy": strategy,
                    "label_free": True,
                    "statistic": "selected_subset",
                    "referral_pct": pct,
                    "referred_n": k,
                    "auto_n": len(auto),
                    "coverage": len(auto) / len(base),
                    **evaluate_auto_subset(auto),
                    **ci,
                }
            )
    rows.extend(random_expected_row(base, pct) for pct in config.REFERRAL_PCTS)
    table = pd.DataFrame(rows)
    write_table(table, config.TABLE_DIR / "table8_lndb_selective_referral.csv")

    plot_strategies = [
        "ensemble_margin",
        "STU-Net_margin",
        "EfficientNet-B0_margin",
        "vote_entropy",
        "random_expected",
    ]
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=160)
    for strategy in plot_strategies:
        sub = table[table["referral_strategy"].eq(strategy)].sort_values("coverage")
        ax.plot(sub["coverage"], sub["auto_error_rate"], marker="o", linewidth=1.5, label=strategy)
    ax.set_xlabel("Auto-handled coverage")
    ax.set_ylabel("Auto-handled error rate")
    ax.set_title("LNDb risk-coverage under label-free referral")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = config.FIGURE_DIR / "fig7_lndb_risk_coverage_curve.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    save_figure_source(table, out.name)


def main() -> None:
    if missing_external():
        write_placeholder("External LNDb DL prediction files were not available.")
        print("Skipped external validation because LNDb prediction files are missing.")
        return

    base = None
    audit = []
    for model in config.LNDB_MODEL_FILES:
        model_df = load_lndb_model(model)
        if base is None:
            base = model_df
        else:
            if set(base["finding_id"]) != set(model_df["finding_id"]):
                raise ValueError(f"{model}: external ID sets differ; refusing silent inner-join loss")
            aligned = model_df.set_index("finding_id").loc[base["finding_id"]]
            if not np.array_equal(base["y_true"].to_numpy(), aligned["y_true"].to_numpy()):
                raise ValueError(f"{model}: conflicting external labels")
            base = base.merge(model_df.drop(columns="y_true"), on="finding_id", validate="one_to_one", how="left")
        audit.append({"model": model, "unique_findings": len(model_df), "id_set_matches": True,
                      "labels_match": True, "silent_drop_n": 0})
    assert base is not None
    meta = pd.read_csv(config.LNDB_METADATA_PATH, dtype={"FindingID": str})
    if meta.groupby("FindingID")[["LNDbID", "label"]].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Conflicting LNDb metadata duplicates")
    if meta.groupby("FindingID")["LNDbID"].nunique().gt(1).any():
        raise ValueError("FindingID is not globally unique across LNDbID")
    if not meta["FindingID"].str.split("_").str[0].astype(int).eq(meta["LNDbID"]).all():
        raise ValueError("Composite FindingID does not include LNDbID")
    meta_unique = meta.drop_duplicates("FindingID").set_index("FindingID")
    if set(meta_unique.index) != set(base["finding_id"]) or not np.array_equal(meta_unique.loc[base["finding_id"], "label"], base["y_true"]):
        raise ValueError("External metadata IDs/labels do not match prediction tables")
    write_table(pd.DataFrame(audit), config.LOG_DIR / "lndb_alignment_checks.csv", latex=False)
    model_cols = list(config.LNDB_MODEL_FILES)
    base["Ensemble mean"] = base[model_cols].mean(axis=1)
    base = add_lndb_behavior_features(base, model_cols)

    internal = pd.read_csv(config.TABLE_DIR / "table1_model_performance.csv") if (config.TABLE_DIR / "table1_model_performance.csv").exists() else pd.DataFrame()
    internal_auc = dict(zip(internal.get("model", []), internal.get("roc_auc", [])))
    rows = []
    for model in model_cols + ["Ensemble mean"]:
        metrics = binary_metrics(base["y_true"], base[model])
        rows.append(
            {
                "model": model,
                "architecture_group": config.MODEL_GROUPS.get(model, "unspecified"),
                "external_n": len(base),
                "external_positive_label_n": int(base["y_true"].sum()),
                "internal_auc": internal_auc.get(model, np.nan),
                "external_auc": metrics["roc_auc"],
                "auc_delta_external_minus_internal": metrics["roc_auc"] - internal_auc.get(model, np.nan),
                "external_pr_auc": metrics["pr_auc"],
                "external_accuracy": metrics["accuracy"],
                "external_sensitivity": metrics["sensitivity"],
                "external_specificity": metrics["specificity"],
                "external_brier": metrics["brier"],
            }
        )
    table = pd.DataFrame(rows)
    paired = table[table["model"].isin(model_cols)].dropna(subset=["internal_auc", "external_auc"])
    rho, pvalue = (np.nan, np.nan)
    if len(paired) >= 3:
        rho, pvalue = stats.spearmanr(paired["internal_auc"], paired["external_auc"])
    table["internal_external_rank_spearman"] = rho
    table["internal_external_rank_spearman_p"] = pvalue
    write_table(table, config.TABLE_DIR / "table7_external_validation_if_available.csv")
    write_table(base, config.INTERMEDIATE_DIR / "lndb_behavior_table.csv", latex=False)
    write_lndb_referral_outputs(base, model_cols)

    plot = table.dropna(subset=["internal_auc", "external_auc"]).copy()
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
    out = config.FIGURE_DIR / "fig6_external_shift_if_available.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    save_figure_source(plot, out.name)
    print(f"Wrote external validation table for {len(base)} LNDb rows.")


if __name__ == "__main__":
    main()
