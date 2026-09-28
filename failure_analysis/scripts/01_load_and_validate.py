from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils import ensure_output_dirs, load_all_luna_predictions_long, load_luna25_metadata, model_prediction_path, write_table, write_text


def file_inventory() -> pd.DataFrame:
    rows = []
    candidates = [
        ("metadata", config.LUNA25_METADATA_PATH),
        ("metadata", config.LUNA25_PUBLIC_METADATA_PATH),
        ("metadata", config.LUNA25_MERGED_METADATA_PATH),
        ("metadata", config.PATIENT_SPLIT_PATH),
        ("external_metadata", config.LNDB_METADATA_PATH),
    ]
    candidates.extend(("luna25_prediction", model_prediction_path(model)) for model in config.MODEL_FILES)
    candidates.extend(("lndb_prediction", config.LNDB_DL_DIR / fn) for fn in config.LNDB_MODEL_FILES.values())
    for kind, path in candidates:
        row = {"kind": kind, "path": str(path), "exists": path.exists()}
        if path.exists() and path.suffix == ".csv":
            head = pd.read_csv(path, nrows=0)
            row["columns"] = "; ".join(head.columns)
            with path.open("rb") as handle:
                row["rows"] = max(sum(1 for _ in handle) - 1, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ensure_output_dirs()
    inventory = file_inventory()
    write_table(inventory, config.INTERMEDIATE_DIR / "candidate_input_files.csv", latex=False)

    checks: list[dict[str, object]] = []

    meta = load_luna25_metadata()
    preds = load_all_luna_predictions_long()
    case_ids = sorted(preds["case_id"].unique())
    test_meta = meta[meta["case_id"].isin(case_ids)].copy()

    checks.append(
        {
            "check": "input_file_presence",
            "status": "PASS" if inventory["exists"].all() else "WARN",
            "detail": f"{int(inventory['exists'].sum())}/{len(inventory)} candidate files found",
        }
    )

    label_counts = preds.groupby("case_id")["y_true"].nunique(dropna=True)
    checks.append(
        {
            "check": "one_ground_truth_label_per_case_in_predictions",
            "status": "PASS" if int((label_counts > 1).sum()) == 0 else "FAIL",
            "detail": f"{int((label_counts > 1).sum())} cases have conflicting prediction-file labels",
        }
    )

    meta_label_counts = test_meta.groupby("case_id")["y_true"].nunique(dropna=True)
    checks.append(
        {
            "check": "one_ground_truth_label_per_case_in_metadata",
            "status": "PASS" if int((meta_label_counts > 1).sum()) == 0 else "FAIL",
            "detail": f"{int((meta_label_counts > 1).sum())} cases have conflicting metadata labels",
        }
    )

    pred_label = preds.drop_duplicates("case_id")[["case_id", "y_true"]].rename(columns={"y_true": "pred_label"})
    merged_label = pred_label.merge(test_meta[["case_id", "y_true"]], on="case_id", how="left")
    mismatched = merged_label[merged_label["y_true"].notna() & merged_label["pred_label"].ne(merged_label["y_true"])]
    checks.append(
        {
            "check": "prediction_metadata_label_consistency",
            "status": "PASS" if len(mismatched) == 0 else "FAIL",
            "detail": f"{len(mismatched)} mismatched case labels between prediction files and metadata",
        }
    )

    prob_bad = preds[~preds["p_malignant"].between(0.0, 1.0, inclusive="both") | preds["p_malignant"].isna()]
    checks.append(
        {
            "check": "probability_range",
            "status": "PASS" if len(prob_bad) == 0 else "FAIL",
            "detail": f"{len(prob_bad)} probability values are missing or outside [0, 1]",
        }
    )

    per_model_counts = preds.groupby("model")["case_id"].nunique().sort_index()
    checks.append(
        {
            "check": "per_model_case_counts",
            "status": "PASS" if per_model_counts.nunique() == 1 else "WARN",
            "detail": "; ".join(f"{k}: {v}" for k, v in per_model_counts.items()),
        }
    )

    missing_meta = len(set(case_ids) - set(test_meta["case_id"]))
    checks.append(
        {
            "check": "metadata_available_for_prediction_cases",
            "status": "PASS" if missing_meta == 0 else "WARN",
            "detail": f"{missing_meta} prediction cases missing from clinical metadata",
        }
    )

    missing_fields = []
    for col in ["age", "sex", "size_mm", "density", "margin", "lobe_or_location", "smoking_status"]:
        n_missing = int(test_meta[col].isna().sum() if col in test_meta else len(test_meta))
        if col in ["density", "margin", "lobe_or_location", "smoking_status"]:
            n_missing = int(test_meta[col].astype(str).eq("Missing").sum())
        missing_fields.append({"field": col, "missing_n": n_missing, "missing_pct": 100.0 * n_missing / len(test_meta)})
    missing_df = pd.DataFrame(missing_fields)
    write_table(missing_df, config.INTERMEDIATE_DIR / "metadata_missingness.csv", latex=False)

    multi_ann_patients = int((test_meta.groupby("patient_id")["case_id"].nunique() > 1).sum())
    checks.append(
        {
            "check": "multiple_annotations_per_patient",
            "status": "INFO",
            "detail": f"{multi_ann_patients} test-set patients have more than one annotation",
        }
    )

    class_counts = pred_label["pred_label"].value_counts().sort_index()
    checks.append(
        {
            "check": "class_distribution",
            "status": "INFO",
            "detail": f"benign={int(class_counts.get(0, 0))}, malignant={int(class_counts.get(1, 0))}, total={len(pred_label)}",
        }
    )

    checks_df = pd.DataFrame(checks)
    write_table(checks_df, config.LOG_DIR / "validation_checks.csv", latex=False)

    report_lines = [
        "# Validation report",
        "",
        f"- Source root: `{config.SOURCE_ROOT}`",
        f"- LUNA25 prediction cases: {len(case_ids)}",
        f"- LUNA25 metadata rows matched to predictions: {len(test_meta)}",
        f"- Models: {', '.join(config.MODEL_FILES)}",
        f"- Probability column: `{config.PROBABILITY_COLUMN}`",
        "",
        "## Column mapping",
    ]
    for logical, source in config.COLUMN_MAPPING.items():
        report_lines.append(f"- `{logical}` -> `{source}`")
    report_lines.extend(["", "## Checks"])
    for row in checks:
        report_lines.append(f"- {row['status']}: {row['check']} - {row['detail']}")
    report_lines.extend(["", "## Metadata missingness"])
    for _, row in missing_df.iterrows():
        report_lines.append(f"- {row['field']}: {int(row['missing_n'])} missing ({row['missing_pct']:.1f}%)")
    write_text(config.LOG_DIR / "validation_report.md", "\n".join(report_lines) + "\n")
    print(f"Validated {len(case_ids)} LUNA25 cases across {len(config.MODEL_FILES)} DL models.")


if __name__ == "__main__":
    main()
