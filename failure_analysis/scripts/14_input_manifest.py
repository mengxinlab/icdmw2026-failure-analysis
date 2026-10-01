"""Hash original inputs and save validation evidence without publishing subject IDs."""
from __future__ import annotations
import hashlib
import json
import platform
import sys
from pathlib import Path
import importlib.metadata
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils import load_behavior_table, write_table, write_text

def main() -> None:
    paths = [config.LUNA25_METADATA_PATH, config.LUNA25_PUBLIC_METADATA_PATH,
             config.LUNA25_MERGED_METADATA_PATH, config.PATIENT_SPLIT_PATH, config.LNDB_METADATA_PATH]
    paths += [config.LUNA25_DL_DIR/fn for fn in config.MODEL_FILES.values()]
    paths += [config.LNDB_DL_DIR/fn for fn in config.LNDB_MODEL_FILES.values()]
    inputs = []
    for path in paths:
        row = {"path_relative_to_source_root": str(path.relative_to(config.SOURCE_ROOT)),
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        if path.suffix == ".csv":
            frame = pd.read_csv(path, low_memory=False)
            row.update(rows=len(frame), columns=list(frame.columns))
            for col in ["AnnotationID", "FindingID"]:
                if col in frame:
                    row["unique_id_count"] = int(frame[col].nunique())
        inputs.append(row)
    split = json.loads(config.PATIENT_SPLIT_PATH.read_text())
    df = load_behavior_table()
    public = pd.read_csv(config.LUNA25_PUBLIC_METADATA_PATH, dtype={"AnnotationID":str})
    missed = df[df["unanimous_wrong"] & df["y_true"].eq(1)][["case_id", "patient_id"]].merge(
        public[["AnnotationID", "NoduleID"]], left_on="case_id", right_on="AnnotationID", validate="one_to_one")
    write_table(missed, config.LOG_DIR/"unanimous_positive_mapping.csv", latex=False)
    data = {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "jinja2"]},
            "referral_tie_seed": config.RANDOM_SEED, "threshold_grid": [0.3,0.5,0.7],
            "split_patient_counts": {name:len(split[name]) for name in ["train","val","test"]},
            "unanimous_positive_summary": {"annotations":len(missed), "nodules":int(missed["NoduleID"].nunique()),
                                            "patients":int(missed["patient_id"].nunique())},
            "inputs": inputs,
            "official_label_verification": json.loads((config.LOG_DIR/"lndb_label_verification.json").read_text()),
            "availability": "Model predictions are study-generated inputs, not downloads supplied by the dataset websites. Exact numerical reproduction requires the hash-matching private inputs; the public code and aggregate outputs alone do not reproduce patient-level results."}
    write_text(config.LOG_DIR/"input_manifest.json", json.dumps(data, indent=2)+"\n")
    print("Wrote input hashes and aggregate validation evidence.")

if __name__ == "__main__":
    main()
