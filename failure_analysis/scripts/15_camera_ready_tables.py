"""Generate camera-ready LaTeX rows directly from canonical aggregate results."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config
from utils import write_text

OUT = config.OUTPUT_DIR / "camera_ready"
NAMES = {"ensemble_margin":"Ensemble margin", "STU-Net_margin":"STU-Net margin",
         "EfficientNet-B0_margin":"EfficientNet-B0 margin", "mean_individual_margin":"Mean ind.\\ margin",
         "vote_entropy":"Vote entropy", "random_expected":"Random (expected)",
         "max_p_gap":"Max probability gap", "std_p":"Probability SD", "hard_case_score":"Hard-case heuristic"}
def f(x): return f"{x:.3f}"
def rows(name, values): write_text(OUT/name, "\n".join(" & ".join(row)+r" \\" for row in values)+"\n\\bottomrule\n")
def read(name): return pd.read_csv(config.TABLE_DIR/name)
def main():
    p=read("table1_model_performance.csv")
    groups={"ensemble":"Ensemble","medical_pretrained":"Med.-pretrained","CNN":"CNN","transformer":"Transformer"}
    rows("performance_rows.tex", [[r.model,groups.get(r.architecture_group,r.architecture_group), f(r.roc_auc),
         f(r.patient_cluster_auc_ci_low)+"--"+f(r.patient_cluster_auc_ci_high), f(r.pr_auc), f(r.accuracy),
         f(r.sensitivity),f(r.specificity),f(r.brier)] for r in p.itertuples()])
    p=read("table5_disagreement_selective_referral.csv");p=p[p.referral_pct.eq(20)].set_index("referral_strategy")
    order=["STU-Net_margin","mean_individual_margin","ensemble_margin","vote_entropy","EfficientNet-B0_margin",
           "random_expected","max_p_gap","std_p","hard_case_score"]
    values=[]
    for name in order:
        r=p.loc[name];rnd=name=="random_expected"
        values.append([NAMES[name],f(r.auto_error_rate) if rnd else f(r.auto_error_rate)+" ("+f(r.auto_error_cluster_ci_low)+"--"+f(r.auto_error_cluster_ci_high)+")",
                       "--" if rnd else f(r.sensitivity), "--" if rnd else f(r.specificity),"No" if name=="hard_case_score" else "Yes"])
    rows("referral_rows.tex",values)
    p=read("table9_malignancy_aware_referral.csv")
    rows("capture_rows.tex",[[r.cohort,NAMES[r.strategy],f(r.positive_label_capture),
                              f"{r.referred_false_negative_n}/{r.base_false_negative_n}",
                              f(r.fn_capture),f(r.residual_false_negative_rate),f(r.detection_upper_bound)] for r in p.itertuples()])
    p=read("table7_external_validation_if_available.csv").sort_values("external_auc",ascending=False)
    rows("external_auc_rows.tex",[[r.model,f(r.internal_auc),f(r.external_auc),f(r.auc_delta_external_minus_internal).replace("-", "$-$")] for r in p.itertuples()])
    p=read("table8_lndb_selective_referral.csv");p=p[p.referral_pct.eq(20)].set_index("referral_strategy")
    values=[]
    for name in ["mean_individual_margin","ensemble_margin","STU-Net_margin","vote_entropy","EfficientNet-B0_margin","random_expected"]:
        r=p.loc[name];rnd=name=="random_expected"
        values.append([NAMES[name],f(r.auto_error_rate),"--" if rnd else f(r.sensitivity),"--" if rnd else f(r.specificity)])
    rows("external_referral_rows.tex",values)
    p=read("threshold_sensitivity.csv");p=p[p.cohort.eq("LUNA25") & p.strategy.eq("STU-Net_margin")]
    rows("threshold_rows.tex",[[f"{r.threshold:.1f}",str(r.majority_failure_n),str(r.unanimous_failure_n),
                               f"{r.large_nodule_enrichment:.2f}", f(r.auto_error_rate),f(r.residual_false_negative_rate)] for r in p.itertuples()])
    print("Wrote CSV-derived camera-ready table rows.")
if __name__=="__main__": main()
