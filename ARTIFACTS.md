# Camera-ready artifact scope

Version: `camera-ready-2026-10-02-r1`. This release provides reusable analysis
code and aggregate evidence, not a complete end-to-end reproduction package.

## Inputs and target definitions

LUNA25 source images and annotations are documented at
https://doi.org/10.5281/zenodo.14223624 and
https://doi.org/10.5281/zenodo.14673658. The study's saved patient split has
1484 training, 318 validation, and 318 test patients with disjoint ID sets.
The seven prediction tables share 917 test annotation IDs from exactly the
318 saved test patients. These checks do not establish historical checkpoint
execution or absence of test-set tuning. Four unanimously missed positive
annotations map to two official nodule IDs in two patients.

LNDb uses Dataset v4, https://doi.org/10.5281/zenodo.8348419.
The official `allNods.csv` SHA256 is
`bfdb261199323f86ff2fb96b0ece785955f8c6198b3542c1d076a2a6e576c3b8`.
Its supplied `Malignancy` field is an average radiologist suspicion score,
not lesion-level pathology. Scores <=2 are negative proxy labels; scores >=4
are positive proxy labels; intermediate scores are excluded. No new averaging
of individual reader ratings is performed. `verify_lndb_labels.py` checks
these labels against the hash-pinned official file; this is retrospective
verification, not the recovered original label-generation script.

The upstream crop pool contains 768 ROIs (465 low, 74 high, 229 intermediate).
The eligible extremes contain 539 ROIs. The evaluation subset contains
439 unique ROIs (365 low, 74 high) from 182 CTs; the original process selecting
these cases, including the omission of 100 eligible low-score ROIs, was not
recovered. Image-to-coordinate provenance and original checkpoint execution
cannot be reconstructed by this release.

LNDb prediction files have 814 rows per model. Repeated rows agree on labels
and probabilities and collapse to 439 global composite `LNDbID_FindingID`
identifiers. All seven model ID sets and labels match. The loader rejects
conflicting duplicates or cross-model label differences before ID-only merge;
it does not silently discard conflicts through a joint ID-label key.

## Numerical reproduction and privacy

`aggregate/input_manifest.json` records relative input paths, hashes, schemas,
software versions, split counts, and label-verification totals. It contains
no patient IDs. Model probabilities are study-generated private inputs, not
files supplied by the dataset download pages. Dataset access alone therefore
does not reproduce this paper's results. Exact reproduction requires matching
metadata, split files, and stored predictions under their applicable terms.
Checkpoint training and the original LNDb sampling process are not included.

Public aggregate CSVs and generated table rows allow arithmetic checks of the
reported results. No CT images, raw metadata, case-level predictions, behavior
matrices, per-case referral rankings, or individual missed-case mappings are
published. Subgroup summaries are descriptive; row-level exploratory q values
in the full exports do not establish patient-adjusted significance and are not
used in the manuscript's subgroup table.

## Referral definitions and regeneration

`random_expected` is the analytic random-referral expectation (full-cohort
error), not an observed subset. `random_simulated` is a separate deterministic
simulation. Score ties use label-free SHA256 ordering of seed, cohort, and
stable case ID; the primary seed is 20260530. Rankings are persisted locally
and reused by all referral scripts. Seven-model binary vote entropy has four
distinct values. Sensitivity exports report all 100 consecutive tie seeds,
not a favorable selected seed.

Threshold analyses use the fixed descriptive grid 0.3, 0.5, 0.7 while holding the
0.5 referral ranking fixed. FN capture is referred original FNs divided by
baseline FNs; residual FN rate is conditional on retained positive labels.
The label-informed hard-case heuristic is a retrospective comparator, not an
optimal oracle bound or a clinical policy.

Install `failure_analysis/requirements.lock.txt`, set
`FAILURE_ANALYSIS_SOURCE_ROOT`, then run `python failure_analysis/run_all.py`.
This is the study-specific entry point and requires both LUNA25 and LNDb
inputs. Reuse on other prediction tables requires configuration of the input
schema and applicable audit modules; downstream external-dependent modules
are not made optional by script 09's conditional availability check.
For an offline official LNDb file, set `FAILURE_ANALYSIS_LNDB_OFFICIAL_CSV`.
The input hash is validated. Script 15 generates the camera-ready table rows
from canonical CSVs. See the pipeline README and data dictionary for schemas.

The legacy aggregate fields `pr_auc`/`external_pr_auc` are average precision
(AP), not trapezoidal PR area. ECE bins positive-class probability in 10
equal-width bins and weights calibration gaps by bin size. The Spearman
coefficient across seven models is descriptive; the manuscript does not report
the asymptotic p value stored in historical aggregate outputs. This revision
changes definitions, documentation and interpretation, not numerical results.
