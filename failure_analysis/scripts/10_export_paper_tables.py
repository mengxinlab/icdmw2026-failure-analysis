from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils import fmt_float, required_output_paths, write_text


def read_table(name: str) -> pd.DataFrame:
    path = config.TABLE_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    perf = read_table("table1_model_performance.csv")
    cooc = read_table("table2_error_cooccurrence.csv")
    subgroup = read_table("table3_consensus_failure_subgroups.csv")
    hce = read_table("table4_high_confidence_errors.csv")
    referral = read_table("table5_disagreement_selective_referral.csv")
    external = read_table("table7_external_validation_if_available.csv")
    external_referral = read_table("table8_lndb_selective_referral.csv")

    key_lines = ["# Key findings", ""]
    if len(perf):
        best = perf.sort_values("roc_auc", ascending=False).iloc[0]
        ensemble = perf[perf["model"].eq("Ensemble mean")]
        key_lines.append(
            f"- Best internal LUNA25 ROC-AUC was {fmt_float(best['roc_auc'])} for {best['model']} "
            f"(95% patient-clustered bootstrap CI {fmt_float(best['patient_cluster_auc_ci_low'])}-{fmt_float(best['patient_cluster_auc_ci_high'])})."
        )
        if len(ensemble):
            e = ensemble.iloc[0]
            key_lines.append(
                f"- The ensemble mean reached ROC-AUC {fmt_float(e['roc_auc'])}, accuracy {fmt_float(e['accuracy'])}, "
                f"sensitivity {fmt_float(e['sensitivity'])}, and specificity {fmt_float(e['specificity'])}."
            )
    if len(cooc):
        pair = cooc.sort_values("jaccard_error_similarity", ascending=False).iloc[0]
        key_lines.append(
            f"- The strongest paired error overlap was {pair['model_a']} with {pair['model_b']} "
            f"(Jaccard {fmt_float(pair['jaccard_error_similarity'])}, co-error n={int(pair['co_error_n'])})."
        )
    if len(subgroup):
        top = subgroup[subgroup["n"] >= 20].sort_values("enrichment_ratio", ascending=False).head(1)
        if len(top):
            row = top.iloc[0]
            flag = " exploratory" if bool(row.get("exploratory", False)) else ""
            key_lines.append(
                f"- The most enriched consensus-failure subgroup with n>=20 was {row['subgroup_family']}={row['subgroup']} "
                f"(n={int(row['n'])}, consensus-error rate {fmt_float(row['consensus_error_rate'])}, "
                f"enrichment {fmt_float(row['enrichment_ratio'])}){flag}."
            )
    if len(hce):
        row = hce.sort_values("hce_0.8_rate", ascending=False).iloc[0]
        key_lines.append(
            f"- High-probability errors were nonzero; the largest nominal confidence>=0.8 rate was {fmt_float(row['hce_0.8_rate'])} "
            f"for {row['model']} ({int(row['hce_0.8_n'])}/{int(row['n'])})."
        )
    if len(referral):
        at20 = referral[referral["referral_pct"].eq(20)]
        if len(at20):
            eligible = at20[at20["label_free"].astype(bool) & ~at20["referral_strategy"].str.startswith("random")]
            best = eligible.sort_values("auto_error_rate").iloc[0]
            rnd = at20[at20["referral_strategy"].eq("random_expected")]
            random_text = ""
            if len(rnd):
                random_text = f"; analytical random-referral expected error was {fmt_float(rnd.iloc[0]['auto_error_rate'])}"
            key_lines.append(
                f"- At 20% simulated referral among label-free uncertainty rules, the lowest auto-handled error rate "
                f"was {fmt_float(best['auto_error_rate'])} using {best['referral_strategy']} at coverage "
                f"{fmt_float(best['coverage'])}{random_text}."
            )
            oracle = at20[at20["referral_strategy"].eq("hard_case_score")]
            if len(oracle):
                key_lines.append(
                    f"- The retrospective hard_case_score reached auto-handled error rate {fmt_float(oracle.iloc[0]['auto_error_rate'])} "
                    "at 20% referral, but it uses observed error labels and is an audit score rather than a deployable triage rule."
                )
    if len(external) and "external_auc" in external:
        valid = external.dropna(subset=["external_auc"])
        if len(valid):
            best_ext = valid.sort_values("external_auc", ascending=False).iloc[0]
            key_lines.append(
                f"- On LNDb external validation, the highest external ROC-AUC was {fmt_float(best_ext['external_auc'])} "
                f"for {best_ext['model']}; internal-external ranking Spearman rho was "
                f"{fmt_float(best_ext.get('internal_external_rank_spearman', np.nan))}."
            )
    if len(external_referral) and "referral_pct" in external_referral:
        at20 = external_referral[external_referral["referral_pct"].eq(20)]
        if len(at20):
            best = at20[~at20["referral_strategy"].str.startswith("random")].sort_values("auto_error_rate").iloc[0]
            rnd = at20[at20["referral_strategy"].eq("random_expected")]
            random_text = ""
            if len(rnd):
                random_text = f"; analytical random-referral expected error was {fmt_float(rnd.iloc[0]['auto_error_rate'])}"
            key_lines.append(
                f"- On LNDb at 20% simulated referral, the lowest label-free auto-handled error rate was "
                f"{fmt_float(best['auto_error_rate'])} using {best['referral_strategy']} at coverage "
                f"{fmt_float(best['coverage'])}{random_text}."
            )
    key_lines.append("- Subgroup and rule-mining analyses are descriptive and should be treated as hypothesis-generating.")
    write_text(config.PAPER_SUMMARY_DIR / "key_findings.md", "\n".join(key_lines) + "\n")

    methods = (
        "We performed a retrospective failure-mining audit of stored 3D deep-learning predictions for "
        "LUNA25 lung nodule malignancy classification. Per-annotation malignant probabilities from STU-Net, "
        "EfficientNet-B0, ResNet-18, DenseNet-121, ResNet-50, Swin-UNETR, and ViT-Base were merged with "
        "clinical metadata using AnnotationID, with y_true=1 denoting the benchmark positive label. Model behavior was "
        "summarized by per-model errors, nominal confidence, high-probability errors, error co-occurrence, ensemble mean "
        "probability, vote entropy, probability dispersion, and majority consensus failure. We computed ROC-AUC, "
        "average precision (AP; average_precision_score), accuracy, sensitivity, specificity, F1, balanced accuracy, "
        "Brier score, and ECE using 10 equal-width bins of positive-class probability, weighted by bin size, "
        "not bins of nominal confidence. We computed stratified bootstrap confidence intervals, "
        "patient-cluster AUC intervals, McNemar tests, "
        "patient-clustered subgroup/referral rate intervals, interpretable decision-tree rules, subgroup enrichment "
        "statistics, and selective-referral risk-coverage curves. The hard_case_score was treated as a retrospective audit score because it includes observed error counts. "
        "LNDb was audited against the official v4 average radiologist-suspicion score: scores <=2 were "
        "negative proxy labels, scores >=4 positive proxy labels, and intermediate scores excluded. "
        "These are not pathology-confirmed cancer outcomes. Selected LNDb prediction files were analyzed "
        "with discrimination metrics and label-free referral simulations; original ROI selection and "
        "checkpoint execution were not reconstructed. The full run_all.py entry point requires both cohorts."
    )
    write_text(config.PAPER_SUMMARY_DIR / "methods_paragraph.md",
               "# Generated methods draft\n\nFor inspection only; the final manuscript's definitions govern. "
               "Do not overwrite the final manuscript with this generated summary.\n\n" + methods + "\n")

    result_bits = []
    if len(perf):
        best = perf.sort_values("roc_auc", ascending=False).iloc[0]
        result_bits.append(f"Across seven DL models, {best['model']} had the highest internal ROC-AUC ({fmt_float(best['roc_auc'])}).")
    if len(cooc):
        pair = cooc.sort_values("jaccard_error_similarity", ascending=False).iloc[0]
        result_bits.append(f"Errors showed measurable co-occurrence, with the largest pairwise Jaccard similarity between {pair['model_a']} and {pair['model_b']}.")
    if len(hce):
        total_hce = int(hce["hce_0.8_n"].sum())
        result_bits.append(f"High-probability errors at nominal confidence>=0.8 were observed across model outputs (aggregate model-count total {total_hce}).")
    if len(referral):
        result_bits.append("Selective-referral simulations quantified the trade-off between auto-handled coverage and residual error.")
    write_text(config.PAPER_SUMMARY_DIR / "results_paragraph.md", " ".join(result_bits) + "\n")

    limitations = "\n".join(
        [
            "# Limitations",
            "",
            "- The analysis is retrospective and uses public benchmark-derived tabular artifacts rather than a prospective clinical workflow.",
            "- LUNA25 labels are benchmark curation labels and may not capture all sources of diagnostic uncertainty.",
            "- All mined subgroups are exploratory, with extra uncertainty for small positive-label counts.",
            "- Failure mining is descriptive and does not prove causal mechanisms for model errors.",
            "- The hard_case_score includes observed model errors and should not be interpreted as a deployable referral policy.",
            "- Detection upper bounds alone assume perfect downstream review; referral error metrics do not model expert diagnosis.",
            "- External validation is limited to the available LNDb prediction tables and may not cover all acquisition shifts.",
        ]
    )
    write_text(config.PAPER_SUMMARY_DIR / "limitations.md", limitations + "\n")

    checklist = []
    for path in required_output_paths():
        checklist.append({"path": str(path), "exists": path.exists()})
    pd.DataFrame(checklist).to_csv(config.PAPER_SUMMARY_DIR / "generated_files_checklist.csv", index=False)
    print("Wrote manuscript-oriented summary files.")


if __name__ == "__main__":
    main()
