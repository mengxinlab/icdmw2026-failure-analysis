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

`run_all.py` is the study-specific reproduction entry point and requires both
LUNA25 and LNDb inputs, including the metadata and split files used by the
manifest. It runs LNDb label verification before analysis; external validation
is not optional in this full entry point. Reuse on other prediction tables
requires configuring the input schema and running the applicable audit modules.
The conditional skip inside script 09 alone does not make downstream
class-specific, sensitivity, manifest, or table-generation modules optional.

Outputs are written under `failure_analysis/outputs/`.
Patient-clustered modules require a metadata row and valid patient ID for every
prediction case; missing mappings fail before cohort/split checks or metrics.
Missing optional diameter/density fields are retained and reported, not dropped.

Raw files in the source
repository configured by `SOURCE_ROOT` in `failure_analysis/config.py` are
read-only inputs and are never modified. By default, `SOURCE_ROOT` points to
a sibling `VLMvsDL` directory; set `FAILURE_ANALYSIS_SOURCE_ROOT` to the
source-data repository on another machine. The code does not distribute CT
images, clinical metadata, or per-case prediction files.

For a standalone public code repository, include the scripts, configuration,
requirements, documentation, and reviewed aggregate exports only. The local `outputs/` directory contains
case-level derivatives and machine-specific logs and must not be published;
the `.gitignore` excludes it for a new repository. If publishing from this
existing project repository, first verify its tracked-file list because
`.gitignore` does not remove files already tracked by Git. Users must obtain
source metadata and prediction files separately under their dataset terms.
Study-generated predictions are not available from dataset websites. Exact
numerical reproduction requires the hash-matching private inputs; see the
public release's `ARTIFACTS.md` and `aggregate/input_manifest.json`.

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

Metric naming: the legacy CSV keys `pr_auc` and `external_pr_auc` store
scikit-learn `average_precision_score`, reported as AP in the manuscript
(not trapezoidal PR-curve area). ECE uses 10 equal-width bins of positive-class
probability, with bin-size-weighted absolute differences between mean
probability and positive-label frequency; it does not bin `max(p, 1-p)`.

## Scripts

### Final manuscript figures

After `run_all.py`, run from the project root:

```bash
python failure_analysis/replot_figures_tnr.py
```

This public renderer uses exported source CSVs and aggregate tables without
recomputing statistics. It produces the final Fig. 2 co-error heatmap
(`fig1_error_cooccurrence_heatmap.pdf`) and Fig. 3 risk-coverage curve
(`fig4_risk_coverage_curve.pdf`), with compact model abbreviations and formatted
legends. Fig. 3 shows five curves; EfficientNet-B0 margin remains in Table III
and full CSV outputs. The base pipeline plots are diagnostic, not identical
presentation layouts. PNG/PDF outputs overwrite the local figure derivatives
and copy to `manuscript_icdm/figures` only when that directory exists.
Generated summary prose from script 10 is a review draft, not manuscript source.

### Analysis modules

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
12. `12_malignancy_aware_referral.py`: exports positive-label capture, FN capture,
   and residual false-negative referral metrics using persisted rankings.
13. `13_threshold_and_tie_sensitivity.py`: exports the complete fixed-ranking
   threshold grid (0.3, 0.5, 0.7) and all 100 consecutive tie seeds.
14. `14_input_manifest.py`: hashes inputs and exports aggregate validation
   evidence; individual missed-case mappings remain local.
15. `15_camera_ready_tables.py`: generates manuscript rows from canonical CSVs.

Before these steps, `verify_lndb_labels.py` checks proxy labels against the
hash-pinned official LNDb v4 `allNods.csv` (average radiologist suspicion score
<=2 negative, >=4 positive, intermediate scores excluded). Set
`FAILURE_ANALYSIS_LNDB_OFFICIAL_CSV` to a local official copy for offline use.
The original ROI selection process was not recovered; this verifies labels,
not pathology or historical image extraction.

Referral ties use stable-ID SHA256 ordering with fixed seed 20260530;
`random_expected` is analytical and `random_simulated` is separate. Ranking
files are persisted locally and shared by downstream scripts. The
label-informed hard-case heuristic is not an optimal oracle bound.
