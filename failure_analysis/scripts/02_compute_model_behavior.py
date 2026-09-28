from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils import MODEL_SLUGS, ensure_output_dirs, load_model_predictions, load_luna25_metadata, vote_entropy, write_table


def normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def main() -> None:
    ensure_output_dirs()
    meta = load_luna25_metadata()

    label_frames = []
    for model in config.MODEL_FILES:
        pred = load_model_predictions(model)
        pred["model"] = model
        label_frames.append(pred[["case_id", "y_true", "model"]])
    labels = pd.concat(label_frames, ignore_index=True)
    label_by_case = labels.sort_values("model").drop_duplicates("case_id")[["case_id", "y_true"]]

    keep_meta_cols = [
        "case_id",
        "patient_id",
        "age",
        "sex",
        "size_mm",
        "perpendicular_size_mm",
        "density",
        "margin",
        "lobe_or_location",
        "current_smoker",
        "smoking_status",
        "spiculated_or_irregular",
        "upper_lobe",
        "size_bin",
        "age_group",
    ]
    df = label_by_case.merge(meta[keep_meta_cols], on="case_id", how="left")
    df["y_true"] = df["y_true"].astype(int)

    mapping_rows = []
    for model in config.MODEL_FILES:
        slug = MODEL_SLUGS[model]
        pred = load_model_predictions(model).rename(
            columns={"p_malignant": f"{slug}_p", "logit": f"{slug}_logit", "y_true": f"{slug}_label"}
        )
        cols = ["case_id", f"{slug}_p"]
        if f"{slug}_logit" in pred.columns:
            cols.append(f"{slug}_logit")
        df = df.merge(pred[cols], on="case_id", how="left")
        p = df[f"{slug}_p"]
        df[f"{slug}_pred"] = (p >= 0.5).astype("Int64")
        df[f"{slug}_correct"] = df[f"{slug}_pred"].eq(df["y_true"])
        df[f"{slug}_error"] = ~df[f"{slug}_correct"]
        df.loc[p.isna(), [f"{slug}_pred", f"{slug}_correct", f"{slug}_error"]] = pd.NA
        df[f"{slug}_confidence"] = np.maximum(p, 1.0 - p)
        df[f"{slug}_margin"] = (p - 0.5).abs()
        df[f"{slug}_signed_margin"] = p - 0.5
        df[f"{slug}_high_confidence_error_0_8"] = df[f"{slug}_error"].fillna(False) & (df[f"{slug}_confidence"] >= 0.8)
        df[f"{slug}_high_confidence_error_0_9"] = df[f"{slug}_error"].fillna(False) & (df[f"{slug}_confidence"] >= 0.9)
        mapping_rows.append(
            {
                "model": model,
                "slug": slug,
                "probability_column": f"{slug}_p",
                "error_column": f"{slug}_error",
                "source_file": str(config.LUNA25_DL_DIR / config.MODEL_FILES[model]),
                "architecture_group": config.MODEL_GROUPS.get(model, "unspecified"),
            }
        )

    p_cols = [f"{MODEL_SLUGS[m]}_p" for m in config.MODEL_FILES]
    pred_cols = [f"{MODEL_SLUGS[m]}_pred" for m in config.MODEL_FILES]
    error_cols = [f"{MODEL_SLUGS[m]}_error" for m in config.MODEL_FILES]
    hce08_cols = [f"{MODEL_SLUGS[m]}_high_confidence_error_0_8" for m in config.MODEL_FILES]
    hce09_cols = [f"{MODEL_SLUGS[m]}_high_confidence_error_0_9" for m in config.MODEL_FILES]

    p_mat = df[p_cols].astype(float)
    df["model_count"] = p_mat.notna().sum(axis=1)
    df["error_count"] = df[error_cols].fillna(False).sum(axis=1).astype(int)
    df["correct_count"] = df["model_count"] - df["error_count"]
    df["mean_p"] = p_mat.mean(axis=1)
    df["std_p"] = p_mat.std(axis=1, ddof=0)
    df["max_p_gap"] = p_mat.max(axis=1) - p_mat.min(axis=1)
    df["vote_malignant_count"] = df[pred_cols].fillna(0).sum(axis=1).astype(int)
    df["vote_entropy"] = [vote_entropy(v, n) for v, n in zip(df["vote_malignant_count"], df["model_count"])]
    df["ensemble_pred"] = (df["mean_p"] >= 0.5).astype(int)
    df["ensemble_correct"] = df["ensemble_pred"].eq(df["y_true"])
    df["ensemble_error"] = ~df["ensemble_correct"]
    df["ensemble_confidence"] = np.maximum(df["mean_p"], 1.0 - df["mean_p"])
    df["ensemble_margin"] = (df["mean_p"] - 0.5).abs()
    df["consensus_error_majority"] = df["error_count"] >= (np.floor(df["model_count"] / 2.0) + 1)
    df["consensus_error_strict"] = df["error_count"] >= np.ceil(0.75 * df["model_count"])
    df["unanimous_correct"] = df["correct_count"].eq(df["model_count"])
    df["unanimous_wrong"] = df["error_count"].eq(df["model_count"])
    df["hce_count_0_8"] = df[hce08_cols].fillna(False).sum(axis=1).astype(int)
    df["hce_count_0_9"] = df[hce09_cols].fillna(False).sum(axis=1).astype(int)
    df["high_confidence_error_any_0_8"] = df["hce_count_0_8"] > 0
    df["high_confidence_error_any_0_9"] = df["hce_count_0_9"] > 0
    std_cut = float(df["std_p"].quantile(0.80))
    df["disagreement_high"] = (df["std_p"] >= std_cut) | (df["max_p_gap"] >= 0.5)

    low_margin = 1.0 - (df["ensemble_margin"] / 0.5).clip(0, 1)
    df["hard_case_score"] = (
        0.35 * (df["error_count"] / df["model_count"].replace(0, np.nan)).fillna(0)
        + 0.25 * normalize(df["std_p"])
        + 0.20 * (df["hce_count_0_8"] / df["model_count"].replace(0, np.nan)).fillna(0)
        + 0.20 * low_margin.fillna(0)
    )

    out = config.INTERMEDIATE_DIR / "dl_behavior_table.csv"
    df.to_csv(out, index=False)
    write_table(pd.DataFrame(mapping_rows), config.INTERMEDIATE_DIR / "model_column_mapping.csv", latex=False)
    print(f"Wrote behavior table: {out} ({len(df)} rows, {len(df.columns)} columns).")


if __name__ == "__main__":
    main()
