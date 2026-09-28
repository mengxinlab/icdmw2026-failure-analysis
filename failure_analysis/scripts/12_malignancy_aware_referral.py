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


REFERRAL_PCT = 20


def referral_summary(
    df: pd.DataFrame,
    *,
    cohort: str,
    strategy: str,
    score: pd.Series,
) -> dict[str, float | int | str]:
    k = int(math.ceil(len(df) * REFERRAL_PCT / 100.0))
    order = score.sort_values(ascending=False).index.to_numpy()
    referred_idx = set(order[:k])
    referred = df.loc[[idx for idx in df.index if idx in referred_idx]].copy()
    auto = df.loc[[idx for idx in df.index if idx not in referred_idx]].copy()

    total_malignant = int(df["y_true"].eq(1).sum())
    auto_malignant = int(auto["y_true"].eq(1).sum())
    referred_malignant = int(referred["y_true"].eq(1).sum())
    pred_auto = auto["mean_p"].ge(0.5).astype(int)
    pred_all = df["mean_p"].ge(0.5).astype(int)
    auto_fn = int(pred_auto.eq(0).mul(auto["y_true"].eq(1)).sum())
    auto_tp = int(pred_auto.eq(1).mul(auto["y_true"].eq(1)).sum())
    base_tp = int(pred_all.eq(1).mul(df["y_true"].eq(1)).sum())

    return {
        "cohort": cohort,
        "strategy": strategy,
        "referral_pct": REFERRAL_PCT,
        "referred_n": k,
        "referred_malignant_n": referred_malignant,
        "total_malignant_n": total_malignant,
        "malignancy_capture": referred_malignant / total_malignant if total_malignant else np.nan,
        "auto_malignant_n": auto_malignant,
        "auto_false_negative_n": auto_fn,
        "residual_false_negative_rate": auto_fn / auto_malignant if auto_malignant else np.nan,
        "detection_upper_bound": (referred_malignant + auto_tp) / total_malignant if total_malignant else np.nan,
        "base_threshold_sensitivity": base_tp / total_malignant if total_malignant else np.nan,
    }


def luna_rows() -> list[dict[str, float | int | str]]:
    df = load_behavior_table().reset_index(drop=True)
    margin_cols = [f"{MODEL_SLUGS[m]}_margin" for m in config.MODEL_FILES]
    strategies = {
        "vote_entropy": df["vote_entropy"],
        "mean_individual_margin": 0.5 - df[margin_cols].mean(axis=1),
        "ensemble_margin": 0.5 - df["ensemble_margin"],
        "STU-Net_margin": 0.5 - df[f"{MODEL_SLUGS['STU-Net']}_margin"],
    }
    return [referral_summary(df, cohort="LUNA25", strategy=name, score=score) for name, score in strategies.items()]


def lndb_rows() -> list[dict[str, float | int | str]]:
    path = config.INTERMEDIATE_DIR / "lndb_behavior_table.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path).reset_index(drop=True)
    model_cols = list(config.LNDB_MODEL_FILES)
    strategies = {
        "vote_entropy": df["vote_entropy"],
        "ensemble_margin": 0.5 - df["ensemble_margin"],
        "mean_individual_margin": 0.5 - df[[f"{m}_margin" for m in model_cols]].mean(axis=1),
        "STU-Net_margin": 0.5 - df["STU-Net_margin"],
    }
    return [referral_summary(df, cohort="LNDb", strategy=name, score=score) for name, score in strategies.items()]


def main() -> None:
    table = pd.DataFrame(luna_rows() + lndb_rows())
    write_table(table, config.TABLE_DIR / "table9_malignancy_aware_referral.csv")
    print("Wrote malignancy-aware referral table.")


if __name__ == "__main__":
    main()
