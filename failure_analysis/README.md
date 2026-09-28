# Mining Multi-Model Failure Patterns for Reliability Auditing of 3D Lung Nodule Malignancy Predictors

This folder contains the post-training analysis pipeline for the ARRL workshop
paper at IEEE ICDMW 2026. It audits stored per-case predictions from seven
3D lung-nodule malignancy models; it does not retrain the models.

The pipeline deliberately excludes legacy non-DL analysis outputs. It treats
per-case DL probabilities as model-behavior data and mines shared failures,
high-probability errors, architecture-specific patterns, metadata-interpretable
subgroups, and selective-referral signals.

## Environment

The analysis uses stored CSV probability files and runs on CPU. It requires
Python dependencies including Jinja2 for LaTeX table export. Install them with:

```bash
python3 -m pip install -r failure_analysis/requirements.txt
```

A local version-pinned snapshot is recorded in
`failure_analysis/requirements.lock.txt`.

## Run

From the project root:

```bash
python failure_analysis/run_all.py
```

Outputs are written under `failure_analysis/outputs/`. Raw files in the source
repository configured by `SOURCE_ROOT` in `failure_analysis/config.py` are
read-only inputs and are never modified. By default, `SOURCE_ROOT` points to
a sibling `VLMvsDL` directory; set `FAILURE_ANALYSIS_SOURCE_ROOT` to the
source-data repository on another machine. The code does not distribute CT
images, clinical metadata, or per-case prediction files.

For a standalone public code repository, include the scripts, configuration,
requirements, and documentation only. The local `outputs/` directory contains
case-level derivatives and machine-specific logs and must not be published;
the `.gitignore` excludes it for a new repository. If publishing from this
existing project repository, first verify its tracked-file list because
`.gitignore` does not remove files already tracked by Git. Users must obtain
source metadata and prediction files separately under their dataset terms.

## Primary Inputs

- Metadata: `${SOURCE_ROOT}/data/metadata/luna25_clinical_metadata.csv`
- LUNA25 DL predictions:
  `${SOURCE_ROOT}/data/predictions/luna25_dl/files/*_test_preds.csv`
- LNDb external DL predictions:
  `${SOURCE_ROOT}/data/predictions/lndb_dl/*_lndb_preds.csv`

## Column Mapping

- `case_id` -> `AnnotationID`
- `patient_id` -> `PatientID`
- `y_true` -> benchmark `label` (1 positive, 0 negative); the released
  annotation table alone does not provide lesion-level pathology records
- `p_malignant` -> each model prediction file's `pred_prob`
- `age` -> `Age_at_StudyDate`
- `sex` -> `Gender`
- `size_mm` -> `sct_long_dia`
- `density` -> `sct_pre_att`
- `margin` -> `sct_margins`
- `lobe_or_location` -> `sct_epi_loc`
- `smoking_status` -> `cigsmok`

## Scripts

1. `01_load_and_validate.py`: inventories inputs and validates labels,
   probability ranges, metadata availability, and patient clustering.
2. `02_compute_model_behavior.py`: builds
   `outputs/intermediate/dl_behavior_table.csv`.
3. `03_overall_model_performance.py`: computes model and ensemble performance.
4. `04_error_cooccurrence.py`: mines shared error sets and consensus failures.
5. `05_consensus_failure_mining.py`: extracts shallow decision-tree rules and
   subgroup enrichments.
6. `06_high_confidence_errors.py`: quantifies high-probability false positives
   and false negatives on the native probability scale.
7. `07_disagreement_and_selective_referral.py`: simulates uncertainty-driven
   referral.
8. `08_subgroup_analysis.py`: computes subgroup performance and architecture
   patterns.
9. `09_external_validation_if_available.py`: runs LNDb external validation and
   label-free referral simulation if prediction tables exist.
10. `10_export_paper_tables.py`: writes manuscript-oriented summaries.
11. `11_patient_level_sensitivity.py`: repeats key analyses after sampling
   one annotation per patient.
12. `12_malignancy_aware_referral.py`: exports malignancy-capture and
   residual false-negative referral metrics.
