# Failure-pattern audit code (ICDMW 2026)

This repository contains only the post-training analysis code and documentation
for *Mining Multi-Model Failure Patterns for Reliability Auditing of 3D Lung
Nodule Malignancy Predictors*. Start with
[`failure_analysis/README.md`](failure_analysis/README.md).

The input CT images, clinical metadata, stored model predictions, patient-level
behavior tables, and generated outputs are **not** included. Obtain the
underlying datasets and prediction files separately under their own terms and
set `FAILURE_ANALYSIS_SOURCE_ROOT` to the local source-data directory before
running `python failure_analysis/run_all.py`.

The code is provided as a research artifact; it does not constitute a clinical
decision rule or establish the safety of any referral policy.
