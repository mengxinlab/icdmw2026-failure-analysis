"""Fixed-grid threshold audit with unchanged 0.5-based referral rankings."""
from __future__ import annotations
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from referral import referral_order, referral_scores, saved_order
from utils import MODEL_SLUGS, load_behavior_table, write_table

THRESHOLDS = (0.3, 0.5, 0.7)  # All reported, never selected using outcomes.
STRATEGIES = ("STU-Net_margin", "mean_individual_margin", "ensemble_margin", "vote_entropy")
N_TIE_SEEDS = 100

def main() -> None:
    internal = load_behavior_table().reset_index(drop=True)
    external = pd.read_csv(config.INTERMEDIATE_DIR / "lndb_behavior_table.csv").reset_index(drop=True)
    rows, tie_rows = [], []
    for cohort, df in [("LUNA25", internal), ("LNDb", external)]:
        cols = [f"{MODEL_SLUGS[m]}_p" for m in config.MODEL_FILES] if cohort == "LUNA25" else list(config.MODEL_FILES)
        y = df["y_true"].to_numpy(dtype=int)
        p = df[cols].to_numpy(dtype=float)
        k = math.ceil(len(df) * 0.2)
        for threshold in THRESHOLDS:
            errors = (p >= threshold) != y[:, None]
            majority = errors.sum(axis=1) >= 4
            unanimous = errors.all(axis=1)
            ensemble_error = df["mean_p"].ge(threshold).to_numpy() != y
            base_fn = int(((df["mean_p"].to_numpy() < threshold) & (y == 1)).sum())
            big = df["size_mm"].gt(15).to_numpy() if cohort == "LUNA25" else np.zeros(len(df), dtype=bool)
            for strategy in (*STRATEGIES, "random_expected"):
                common = {"cohort": cohort, "threshold": threshold, "strategy": strategy,
                          "ranking_reference_threshold": 0.5, "referral_pct": 20,
                          "majority_failure_n": int(majority.sum()), "unanimous_failure_n": int(unanimous.sum()),
                          "majority_failure_rate": float(majority.mean()), "base_error_rate": float(ensemble_error.mean()),
                          "base_false_negative_n": base_fn,
                          "large_nodule_majority_rate": float(majority[big].mean()) if big.any() else np.nan,
                          "large_nodule_enrichment": float(majority[big].mean()/majority.mean()) if big.any() and majority.mean() else np.nan}
                if strategy == "random_expected":
                    rows.append({**common, "auto_error_rate": float(ensemble_error.mean()),
                                 "expected_auto_false_negative_n": base_fn * (len(df)-k)/len(df),
                                 "fn_capture": k/len(df) if base_fn else np.nan})
                    continue
                referred = set(saved_order(df, cohort, strategy)[:k])
                keep = np.array([idx not in referred for idx in df.index])
                auto_pos = int(((y == 1) & keep).sum())
                auto_fn = int(((df["mean_p"].to_numpy() < threshold) & (y == 1) & keep).sum())
                rows.append({**common, "auto_error_rate": float(ensemble_error[keep].mean()),
                             "auto_positive_n": auto_pos, "auto_false_negative_n": auto_fn,
                             "residual_false_negative_rate": auto_fn/auto_pos if auto_pos else np.nan,
                             "fn_capture": (base_fn-auto_fn)/base_fn if base_fn else np.nan})
        score = referral_scores(df, cohort)["vote_entropy"]
        for offset in range(N_TIE_SEEDS):
            seed = config.RANDOM_SEED + offset
            referred = set(referral_order(df, score, cohort, seed=seed)[:k])
            auto = df.loc[[idx for idx in df.index if idx not in referred]]
            auto_fn = int((auto["mean_p"].lt(0.5) & auto["y_true"].eq(1)).sum())
            base_fn = int((df["mean_p"].lt(0.5) & df["y_true"].eq(1)).sum())
            tie_rows.append({"cohort": cohort, "seed": seed, "referral_pct": 20,
                             "auto_error_rate": float(auto["mean_p"].ge(0.5).ne(auto["y_true"]).mean()),
                             "auto_positive_n": int(auto["y_true"].sum()), "auto_false_negative_n": auto_fn,
                             "residual_false_negative_rate": auto_fn/auto["y_true"].sum(),
                             "fn_capture": (base_fn-auto_fn)/base_fn})
    write_table(pd.DataFrame(rows), config.TABLE_DIR / "threshold_sensitivity.csv")
    ties = pd.DataFrame(tie_rows)
    write_table(ties, config.TABLE_DIR / "vote_entropy_tie_sensitivity.csv")
    summary = []
    for cohort, group in ties.groupby("cohort"):
        for metric in ["auto_error_rate", "residual_false_negative_rate", "fn_capture"]:
            summary.append({"cohort": cohort, "metric": metric, "seeds": N_TIE_SEEDS,
                            "minimum": group[metric].min(), "maximum": group[metric].max(),
                            "q025": group[metric].quantile(0.025), "q975": group[metric].quantile(0.975)})
    write_table(pd.DataFrame(summary), config.TABLE_DIR / "vote_entropy_tie_summary.csv")
    print("Wrote complete threshold grid and 100-seed tie sensitivity.")

if __name__ == "__main__":
    main()
