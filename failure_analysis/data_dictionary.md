# Data Dictionary

## Identifiers and Labels

| Field | Source | Meaning |
|---|---|---|
| `case_id` | `AnnotationID` | LUNA25 annotation-level case identifier. |
| `patient_id` | `PatientID` | Patient identifier used for clustered bootstrap. |
| `y_true` | `label` | Binary ground truth, 1 malignant and 0 benign. |

## Model Behavior Fields

For each model slug, the behavior table includes:

| Suffix | Meaning |
|---|---|
| `_p` | Model probability of malignancy. |
| `_logit` | Stored model logit, when available. |
| `_pred` | Binary prediction using probability >= 0.5. |
| `_correct` | Whether prediction equals `y_true`. |
| `_error` | Whether prediction differs from `y_true`. |
| `_confidence` | `max(p, 1-p)`, interpreted as nominal confidence / probability extremeness rather than calibrated clinical certainty. |
| `_margin` | `abs(p - 0.5)`. |
| `_signed_margin` | `p - 0.5`. |
| `_high_confidence_error_0_8` | High-probability error with nominal confidence >= 0.8. |
| `_high_confidence_error_0_9` | High-probability error with nominal confidence >= 0.9. |

Across-model fields include `model_count`, `error_count`, `mean_p`, `std_p`,
`max_p_gap`, `vote_malignant_count`, `vote_entropy`,
`consensus_error_majority`, `consensus_error_strict`, `unanimous_correct`,
`unanimous_wrong`, `disagreement_high`, and `hard_case_score`.

`hard_case_score` includes observed `error_count` and high-probability error
counts, so it is a retrospective audit score. Use `std_p`, `max_p_gap`,
`vote_entropy`, model margin, or ensemble margin for label-free
referral simulations.

## Clinical Metadata

| Derived Field | Source | Notes |
|---|---|---|
| `age` | `Age_at_StudyDate` | Numeric age at study date. |
| `sex` | `Gender` | Original sex field. |
| `size_mm` | `sct_long_dia` | Long-axis diameter in mm. |
| `perpendicular_size_mm` | `sct_perp_dia` | Perpendicular diameter in mm. |
| `size_bin` | `sct_long_dia` | `<=6 mm`, `>6-8 mm`, `>8-15 mm`, `>15 mm`, or `Missing`. |
| `density` | `sct_pre_att` | 1 solid, 2 part-solid, 3 ground-glass, other documented as other/missing. |
| `margin` | `sct_margins` | 1 smooth, 2 lobulated, 3 spiculated/irregular. |
| `lobe_or_location` | `sct_epi_loc` | Lobar location decoded from repository clinical-text generation. |
| `smoking_status` | `cigsmok` | 1 current, 0 former/never. |
| `spiculated_or_irregular` | `sct_margins == 3` | Boolean morphology flag. |
| `upper_lobe` | `sct_epi_loc in {1, 4, 5}` | Right upper, left upper, or lingula. |

## Architecture Groups

| Model | Group |
|---|---|
| STU-Net | medical_pretrained |
| EfficientNet-B0 | CNN |
| ResNet-18 | CNN |
| DenseNet-121 | CNN |
| ResNet-50 | CNN |
| Swin-UNETR | transformer |
| ViT-Base | transformer |

Swin-UNETR can also reasonably be described as medical-pretrained/hybrid; this
pipeline assigns it to the transformer group for mutually exclusive plotting.
