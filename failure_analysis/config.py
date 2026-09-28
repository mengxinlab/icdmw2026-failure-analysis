"""Configuration for the pure-DL LUNA25/LNDb failure analysis pipeline."""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = Path(
    os.environ.get("FAILURE_ANALYSIS_SOURCE_ROOT", str(PROJECT_ROOT.parent / "VLMvsDL"))
).expanduser()

DATA_DIR = SOURCE_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
LUNA25_DL_DIR = DATA_DIR / "predictions" / "luna25_dl" / "files"
LUNA25_DL_CALIBRATED_DIR = LUNA25_DL_DIR / "calibrated"
LNDB_DL_DIR = DATA_DIR / "predictions" / "lndb_dl"

OUTPUT_DIR = PACKAGE_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
LOG_DIR = OUTPUT_DIR / "logs"
PAPER_SUMMARY_DIR = OUTPUT_DIR / "paper_summary"

LUNA25_METADATA_PATH = METADATA_DIR / "luna25_clinical_metadata.csv"
LUNA25_PUBLIC_METADATA_PATH = METADATA_DIR / "luna25_public_training_development_data.csv"
LUNA25_MERGED_METADATA_PATH = METADATA_DIR / "luna25_nlst_merged_v2.csv"
PATIENT_SPLIT_PATH = METADATA_DIR / "patient_split.json"
LNDB_METADATA_PATH = METADATA_DIR / "lndb_10to1_eval.csv"

# Primary analysis uses the raw model output probabilities. Calibrated versions
# are available in the source repository, but high-probability behavior is more
# directly audited on each model's native probability scale here.
USE_CALIBRATED_LUNA25_PROBABILITIES = False
PROBABILITY_COLUMN = "pred_prob"
CALIBRATED_PROBABILITY_COLUMN = "calibrated_pred_prob"
LOGIT_COLUMN = "logit"

CASE_ID_COL = "AnnotationID"
PATIENT_ID_COL = "PatientID"
LABEL_COL = "label"

COLUMN_MAPPING = {
    "case_id": CASE_ID_COL,
    "patient_id": PATIENT_ID_COL,
    "y_true": LABEL_COL,
    "age": "Age_at_StudyDate",
    "sex": "Gender",
    "size_mm": "sct_long_dia",
    "perpendicular_size_mm": "sct_perp_dia",
    "density_code": "sct_pre_att",
    "margin_code": "sct_margins",
    "lobe_code": "sct_epi_loc",
    "smoking_code": "cigsmok",
    "race_code": "race",
    "ct_quality_code": "ctdxqual",
}

MODEL_FILES = {
    "STU-Net": "stunet_base_warmup_test_preds.csv",
    "EfficientNet-B0": "efficientnet_b0_baseline_test_preds.csv",
    "ResNet-18": "resnet18_baseline_test_preds.csv",
    "DenseNet-121": "densenet121_baseline_test_preds.csv",
    "ResNet-50": "resnet50_baseline_test_preds.csv",
    "Swin-UNETR": "swin_unetr_final_gpu_test_preds.csv",
    "ViT-Base": "vit_baseline_test_preds.csv",
}

LNDB_MODEL_FILES = {
    "STU-Net": "stunet_base_warmup_lndb_preds.csv",
    "EfficientNet-B0": "efficientnet_b0_baseline_lndb_preds.csv",
    "ResNet-18": "resnet18_baseline_lndb_preds.csv",
    "DenseNet-121": "densenet121_baseline_lndb_preds.csv",
    "ResNet-50": "resnet50_baseline_lndb_preds.csv",
    "Swin-UNETR": "swin_unetr_final_gpu_lndb_preds.csv",
    "ViT-Base": "vit_baseline_lndb_preds.csv",
}

MODEL_GROUPS = {
    "STU-Net": "medical_pretrained",
    "EfficientNet-B0": "CNN",
    "ResNet-18": "CNN",
    "DenseNet-121": "CNN",
    "ResNet-50": "CNN",
    "Swin-UNETR": "transformer",
    "ViT-Base": "transformer",
    "Ensemble mean": "ensemble",
}

DENSITY_MAP = {
    1.0: "Solid",
    2.0: "Part-solid",
    3.0: "Ground-glass",
    4.0: "Mixed/other",
    6.0: "Fat",
    7.0: "Fluid/water",
    9.0: "Not determined",
}

MARGIN_MAP = {
    1.0: "Smooth",
    2.0: "Lobulated",
    3.0: "Spiculated/irregular",
    9.0: "Not determined",
}

LOBE_MAP = {
    1.0: "Right upper lobe",
    2.0: "Right middle lobe",
    3.0: "Right lower lobe",
    4.0: "Left upper lobe",
    5.0: "Lingula",
    6.0: "Left lower lobe",
    8.0: "Unspecified lobe",
}

SIZE_BINS = [float("-inf"), 6.0, 8.0, 15.0, float("inf")]
SIZE_LABELS = ["<=6 mm", ">6-8 mm", ">8-15 mm", ">15 mm"]

AGE_BINS = [float("-inf"), 60.0, 65.0, 70.0, float("inf")]
AGE_LABELS = ["<60", "60-64", "65-69", ">=70"]

REFERRAL_PCTS = [0, 5, 10, 15, 20, 30, 40]
RANDOM_SEED = 20260530
N_BOOTSTRAP = 1000
MIN_SUBGROUP_N = 10
EXPLORATORY_MALIGNANT_N = 10
