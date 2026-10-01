"""Shared, label-free ordering and persisted referral selections.

Ties use SHA-256(seed, cohort, stable case ID), never labels or row order.
The seed is fixed before reviewing the revised results.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

import config
from utils import MODEL_SLUGS, write_table


def id_column(df: pd.DataFrame) -> str:
    return "case_id" if "case_id" in df else "finding_id"


def hash_key(case_id: str, cohort: str, seed: int, purpose: str = "tie") -> str:
    return hashlib.sha256(f"{seed}|{cohort}|{purpose}|{case_id}".encode()).hexdigest()


def referral_scores(df: pd.DataFrame, cohort: str) -> dict[str, pd.Series]:
    margins = [f"{MODEL_SLUGS[m]}_margin" if cohort == "LUNA25" else f"{m}_margin"
               for m in config.MODEL_FILES]
    stu = f"{MODEL_SLUGS['STU-Net']}_margin" if cohort == "LUNA25" else "STU-Net_margin"
    eff = f"{MODEL_SLUGS['EfficientNet-B0']}_margin" if cohort == "LUNA25" else "EfficientNet-B0_margin"
    out = {
        "ensemble_margin": 0.5 - df["ensemble_margin"],
        "STU-Net_margin": 0.5 - df[stu],
        "EfficientNet-B0_margin": 0.5 - df[eff],
        "mean_individual_margin": 0.5 - df[margins].mean(axis=1),
        "std_p": df["std_p"], "max_p_gap": df["max_p_gap"],
        "vote_entropy": df["vote_entropy"],
    }
    if cohort == "LUNA25":
        out["hard_case_score"] = df["hard_case_score"]
    # One reproducible pseudo-random simulation, separately named from expectation.
    out["random_simulated"] = df[id_column(df)].astype(str).map(
        lambda cid: int(hash_key(cid, cohort, config.RANDOM_SEED, "random")[:13], 16) / 16**13
    )
    return out


def referral_order(df: pd.DataFrame, score: pd.Series, cohort: str,
                   seed: int = config.RANDOM_SEED) -> np.ndarray:
    raw_ids = df[id_column(df)]
    if raw_ids.isna().any() or raw_ids.astype(str).str.strip().eq("").any():
        raise ValueError("Referral ordering requires nonmissing, nonblank stable IDs")
    ids = raw_ids.astype(str)
    keys = ids.map(lambda cid: hash_key(cid, cohort, seed))
    order = pd.DataFrame({"score": score.to_numpy(), "tie": keys.to_numpy(),
                          "id": ids.to_numpy(), "position": np.arange(len(df))})
    if not np.isfinite(order["score"]).all():
        raise ValueError("Referral scores must be finite")
    positions = order.sort_values(["score", "tie", "id", "position"],
                                 ascending=[False, True, True, True], kind="stable")["position"]
    return df.index.to_numpy()[positions.to_numpy()]


def persist_rankings(df: pd.DataFrame, scores: dict[str, pd.Series], cohort: str) -> dict[str, np.ndarray]:
    cid = id_column(df)
    if df[cid].duplicated().any():
        raise ValueError("Primary referral rankings require unique stable IDs")
    orders = {name: referral_order(df, score, cohort) for name, score in scores.items()}
    rows = []
    for name, order in orders.items():
        for rank, index in enumerate(order, 1):
            rows.append({"cohort": cohort, "strategy": name, "rank": rank,
                         "case_id": str(df.loc[index, cid]), "score": float(scores[name].loc[index]),
                         "tie_seed": config.RANDOM_SEED})
    write_table(pd.DataFrame(rows), config.INTERMEDIATE_DIR / f"{cohort.lower()}_referral_rankings.csv", latex=False)
    return orders


def saved_order(df: pd.DataFrame, cohort: str, strategy: str) -> np.ndarray:
    path = config.INTERMEDIATE_DIR / f"{cohort.lower()}_referral_rankings.csv"
    ranks = pd.read_csv(path, dtype={"case_id": str})
    ids = ranks[ranks["strategy"].eq(strategy)].sort_values("rank")["case_id"]
    lookup = pd.Series(df.index.to_numpy(), index=df[id_column(df)].astype(str))
    if lookup.index.duplicated().any() or set(ids) != set(lookup.index) or len(ids) != len(df):
        raise ValueError(f"Stored {cohort}/{strategy} ranking does not match current cases")
    return lookup.loc[ids].to_numpy()


def random_expected_row(df: pd.DataFrame, pct: int) -> dict[str, object]:
    k = math.ceil(len(df) * pct / 100)
    error = df["mean_p"].ge(0.5).ne(df["y_true"].astype(int))
    return {"referral_strategy": "random_expected", "label_free": True,
            "statistic": "analytical_expectation", "referral_pct": pct,
            "referred_n": k, "auto_n": len(df)-k, "coverage": (len(df)-k)/len(df),
            "auto_error_rate": float(error.mean()),
            "expected_auto_error_n": float(error.sum() * (len(df)-k)/len(df))}
