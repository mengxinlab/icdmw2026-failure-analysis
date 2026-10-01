from __future__ import annotations

import math
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_ROOT / "outputs" / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PACKAGE_ROOT / "outputs" / "mplconfig"))

import numpy as np
import pandas as pd

sys.path.insert(0, str(PACKAGE_ROOT))
import config
from utils import MODEL_SLUGS, load_behavior_table, write_table
from referral import referral_order, referral_scores


N_DRAWS = 1000
REFERRAL_PCT = 20


def referral_auto_error(df: pd.DataFrame, score: pd.Series) -> float:
    k = int(math.ceil(len(df) * REFERRAL_PCT / 100.0))
    referred = set(referral_order(df, score, "LUNA25")[:k])
    auto = df.loc[[idx for idx in df.index if idx not in referred]]
    return float(auto["ensemble_error"].mean())


def metric_values(df: pd.DataFrame, rng: np.random.Generator | None = None) -> dict[str, float]:
    mean_margin_cols = [f"{MODEL_SLUGS[m]}_margin" for m in config.MODEL_FILES]
    values = {
        "majority_consensus_failure_rate": float(df["consensus_error_majority"].mean()),
        "ensemble_error_rate": float(df["ensemble_error"].mean()),
        "random_expected_20pct_auto_error": float(df["ensemble_error"].mean()),
        "STU-Net_margin_20pct_auto_error": referral_auto_error(
            df, 0.5 - df[f"{MODEL_SLUGS['STU-Net']}_margin"]
        ),
        "mean_individual_margin_20pct_auto_error": referral_auto_error(
            df, 0.5 - df[mean_margin_cols].mean(axis=1)
        ),
        "ensemble_margin_20pct_auto_error": referral_auto_error(df, 0.5 - df["ensemble_margin"]),
        "vote_entropy_20pct_auto_error": referral_auto_error(df, df["vote_entropy"]),
    }
    if rng is not None:
        values["random_simulated_20pct_auto_error"] = referral_auto_error(
            df, referral_scores(df, "LUNA25")["random_simulated"]
        )
    return values


def main() -> None:
    df = load_behavior_table().reset_index(drop=True)
    row_level = metric_values(df, np.random.default_rng(config.RANDOM_SEED))
    patient_groups = [group.index.to_numpy() for _, group in df.groupby("patient_id", sort=False)]

    rng = np.random.default_rng(config.RANDOM_SEED)
    sampled_values: dict[str, list[float]] = {metric: [] for metric in row_level}
    for _ in range(N_DRAWS):
        sampled_idx = [int(rng.choice(group_idx)) for group_idx in patient_groups]
        sampled = df.loc[sampled_idx].reset_index(drop=True)
        draw_values = metric_values(sampled, rng)
        for metric, value in draw_values.items():
            sampled_values[metric].append(value)

    rows = []
    for metric, values in sampled_values.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "row_level": row_level[metric],
                "one_annotation_per_patient_mean": float(np.mean(arr)),
                "one_annotation_per_patient_ci_low": float(np.quantile(arr, 0.025)),
                "one_annotation_per_patient_ci_high": float(np.quantile(arr, 0.975)),
                "draws": N_DRAWS,
            }
        )
    write_table(pd.DataFrame(rows), config.TABLE_DIR / "patient_level_sensitivity.csv")
    print("Wrote patient-level sensitivity table.")


if __name__ == "__main__":
    main()
