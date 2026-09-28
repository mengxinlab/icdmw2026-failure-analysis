"""Shared helpers for the ICDM failure-analysis scripts."""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

import config


def ensure_output_dirs() -> None:
    for path in [
        config.OUTPUT_DIR,
        config.TABLE_DIR,
        config.FIGURE_DIR,
        config.INTERMEDIATE_DIR,
        config.LOG_DIR,
        config.PAPER_SUMMARY_DIR,
        config.OUTPUT_DIR / "mplconfig",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def configure_matplotlib() -> None:
    ensure_output_dirs()
    mpl_dir = config.OUTPUT_DIR / "mplconfig"
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_dir))

    # Use Times New Roman for all figures so they match the IEEE manuscript body
    # font. Env vars above must be set before matplotlib is first imported.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager

    tnr_variants = [
        "Times New Roman.ttf",
        "Times New Roman Bold.ttf",
        "Times New Roman Italic.ttf",
        "Times New Roman Bold Italic.ttf",
    ]
    tnr_found = False
    for name in tnr_variants:
        path = Path("/System/Library/Fonts/Supplemental") / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            tnr_found = True

    serif_stack = (["Times New Roman"] if tnr_found else []) + [
        "Times",
        "DejaVu Serif",
        "serif",
    ]
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif_stack,
            "mathtext.fontset": "stix",  # serif math to pair with Times
            "axes.unicode_minus": False,
            # Embed real TrueType (Type 42) glyphs, not Type 3 bitmaps, so the
            # figure PDFs pass IEEE PDF checks and stay searchable.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def slugify_model(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return slug or "model"


MODEL_SLUGS = {name: slugify_model(name) for name in config.MODEL_FILES}
SLUG_TO_MODEL = {slug: name for name, slug in MODEL_SLUGS.items()}


def pct(n: float, d: float) -> float:
    return float(100.0 * n / d) if d else float("nan")


def fmt_float(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_table(df: pd.DataFrame, path: Path, *, latex: bool = True, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    if latex:
        tex_path = path.with_suffix(".tex")
        try:
            df.to_latex(tex_path, index=index, escape=True, float_format=lambda x: f"{x:.3f}")
        except Exception as exc:  # pragma: no cover - best-effort paper export
            tex_path.write_text(f"% Failed to export LaTeX table: {exc}\n", encoding="utf-8")


def label_from_map(value: object, mapping: dict[float, str], missing: str = "Missing") -> str:
    if pd.isna(value):
        return missing
    try:
        return mapping.get(float(value), f"Other ({value})")
    except (TypeError, ValueError):
        return f"Other ({value})"


def load_luna25_metadata() -> pd.DataFrame:
    """Load and standardize LUNA25 clinical metadata without filtering rows."""
    df = pd.read_csv(config.LUNA25_METADATA_PATH, low_memory=False)
    required = [config.CASE_ID_COL, config.PATIENT_ID_COL, config.LABEL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    out = df.copy()
    out["case_id"] = out[config.CASE_ID_COL].astype(str)
    out["patient_id"] = pd.to_numeric(out[config.PATIENT_ID_COL], errors="coerce").astype("Int64")
    out["y_true"] = pd.to_numeric(out[config.LABEL_COL], errors="coerce").astype("Int64")
    out["age"] = pd.to_numeric(out.get("Age_at_StudyDate"), errors="coerce")
    out["sex"] = out.get("Gender", pd.Series(index=out.index, dtype=object)).fillna("Missing").astype(str)
    out["size_mm"] = pd.to_numeric(out.get("sct_long_dia"), errors="coerce")
    out["perpendicular_size_mm"] = pd.to_numeric(out.get("sct_perp_dia"), errors="coerce")
    out["density"] = out.get("sct_pre_att").map(lambda x: label_from_map(x, config.DENSITY_MAP))
    out["margin"] = out.get("sct_margins").map(lambda x: label_from_map(x, config.MARGIN_MAP))
    out["lobe_or_location"] = out.get("sct_epi_loc").map(lambda x: label_from_map(x, config.LOBE_MAP))
    out["current_smoker"] = pd.to_numeric(out.get("cigsmok"), errors="coerce").eq(1)
    out["smoking_status"] = np.where(out["current_smoker"], "Current", "Former/never")
    out.loc[pd.to_numeric(out.get("cigsmok"), errors="coerce").isna(), "smoking_status"] = "Missing"
    out["spiculated_or_irregular"] = pd.to_numeric(out.get("sct_margins"), errors="coerce").eq(3)
    out["upper_lobe"] = pd.to_numeric(out.get("sct_epi_loc"), errors="coerce").isin([1, 4, 5])
    out["size_bin"] = pd.cut(
        out["size_mm"],
        bins=config.SIZE_BINS,
        labels=config.SIZE_LABELS,
        right=True,
    ).astype(object).fillna("Missing")
    out["age_group"] = pd.cut(
        out["age"],
        bins=config.AGE_BINS,
        labels=config.AGE_LABELS,
        right=False,
    ).astype(object).fillna("Missing")
    return out.drop_duplicates("case_id", keep="first")


def model_prediction_path(model_name: str) -> Path:
    filename = config.MODEL_FILES[model_name]
    if config.USE_CALIBRATED_LUNA25_PROBABILITIES:
        stem = Path(filename).stem.replace("_preds", "_calibrated")
        return config.LUNA25_DL_CALIBRATED_DIR / f"{stem}.csv"
    return config.LUNA25_DL_DIR / filename


def load_model_predictions(model_name: str) -> pd.DataFrame:
    path = model_prediction_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file for {model_name}: {path}")
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower().eq("test")].copy()
    prob_col = (
        config.CALIBRATED_PROBABILITY_COLUMN
        if config.USE_CALIBRATED_LUNA25_PROBABILITIES and config.CALIBRATED_PROBABILITY_COLUMN in df.columns
        else config.PROBABILITY_COLUMN
    )
    required = [config.CASE_ID_COL, config.LABEL_COL, prob_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    out = pd.DataFrame(
        {
            "case_id": df[config.CASE_ID_COL].astype(str),
            "y_true": pd.to_numeric(df[config.LABEL_COL], errors="coerce"),
            "p_malignant": pd.to_numeric(df[prob_col], errors="coerce"),
        }
    )
    if config.LOGIT_COLUMN in df.columns:
        out["logit"] = pd.to_numeric(df[config.LOGIT_COLUMN], errors="coerce")
    return out.drop_duplicates("case_id", keep="last").reset_index(drop=True)


def load_all_luna_predictions_long() -> pd.DataFrame:
    rows = []
    for model in config.MODEL_FILES:
        df = load_model_predictions(model)
        df["model"] = model
        df["model_slug"] = MODEL_SLUGS[model]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_behavior_table() -> pd.DataFrame:
    path = config.INTERMEDIATE_DIR / "dl_behavior_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing behavior table. Run 02_compute_model_behavior.py first: {path}")
    return pd.read_csv(path)


def get_probability_columns(df: pd.DataFrame) -> list[str]:
    return [f"{slug}_p" for slug in MODEL_SLUGS.values() if f"{slug}_p" in df.columns]


def get_error_columns(df: pd.DataFrame) -> list[str]:
    return [f"{slug}_error" for slug in MODEL_SLUGS.values() if f"{slug}_error" in df.columns]


def probability_column_for_model(model: str) -> str:
    if model == "Ensemble mean":
        return "mean_p"
    return f"{MODEL_SLUGS[model]}_p"


def model_names_with_ensemble() -> list[str]:
    return list(config.MODEL_FILES.keys()) + ["Ensemble mean"]


def expected_calibration_error(y_true: Iterable[int], y_prob: Iterable[float], n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (p >= lo) & (p <= hi) if lo == 0 else (p > lo) & (p <= hi)
        if not in_bin.any():
            continue
        ece += in_bin.mean() * abs(y[in_bin].mean() - p[in_bin].mean())
    return float(ece)


def binary_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return {k: float("nan") for k in ["roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity", "f1", "balanced_accuracy", "brier", "ece"]}
    pred = (p >= threshold).astype(int)
    if len(np.unique(y)) >= 2:
        roc = float(roc_auc_score(y, p))
        pr = float(average_precision_score(y, p))
    else:
        roc = pr = float("nan")
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=labels).ravel()
    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)) if len(np.unique(y)) >= 2 else float("nan"),
        "ece": expected_calibration_error(y, p),
    }


def stratified_bootstrap_ci(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n_boot: int = config.N_BOOTSTRAP,
    seed: int = config.RANDOM_SEED,
) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(p)
    y, p = y[mask], p[mask]
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ]
        )
        try:
            vals.append(float(metric_fn(y[idx], p[idx])))
        except ValueError:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return tuple(float(x) for x in np.nanpercentile(vals, [2.5, 97.5]))


def cluster_bootstrap_auc_ci(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    clusters: Iterable[object],
    n_boot: int = config.N_BOOTSTRAP,
    seed: int = config.RANDOM_SEED,
) -> tuple[float, float]:
    df = pd.DataFrame({"y": y_true, "p": y_prob, "cluster": clusters}).dropna()
    if df["cluster"].nunique() < 2 or df["y"].nunique() < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    unique_clusters = df["cluster"].dropna().unique()
    idx_by_cluster = {c: df.index[df["cluster"].eq(c)].to_numpy() for c in unique_clusters}
    vals = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in sampled_clusters])
        boot = df.loc[idx]
        if boot["y"].nunique() < 2:
            continue
        vals.append(float(roc_auc_score(boot["y"].astype(int), boot["p"].astype(float))))
    if not vals:
        return (float("nan"), float("nan"))
    return tuple(float(x) for x in np.nanpercentile(vals, [2.5, 97.5]))


def bootstrap_rate_ci(values: Iterable[object], n_boot: int = config.N_BOOTSTRAP, seed: int = config.RANDOM_SEED) -> tuple[float, float]:
    arr = pd.Series(values).dropna().astype(float).to_numpy()
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    rates = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return tuple(float(x) for x in np.nanpercentile(rates, [2.5, 97.5]))


def cluster_bootstrap_rate_ci(
    values: Iterable[object],
    clusters: Iterable[object],
    n_boot: int = config.N_BOOTSTRAP,
    seed: int = config.RANDOM_SEED,
) -> tuple[float, float]:
    df = pd.DataFrame({"value": values, "cluster": clusters}).dropna()
    if df.empty or df["cluster"].nunique() < 2:
        return bootstrap_rate_ci(df["value"], n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed)
    unique_clusters = df["cluster"].dropna().unique()
    idx_by_cluster = {c: df.index[df["cluster"].eq(c)].to_numpy() for c in unique_clusters}
    rates = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in sampled_clusters])
        rates.append(float(df.loc[idx, "value"].astype(float).mean()))
    return tuple(float(x) for x in np.nanpercentile(rates, [2.5, 97.5]))


def fisher_or_chi2_pvalue(a: int, b: int, c: int, d: int) -> float:
    table = np.asarray([[a, b], [c, d]], dtype=float)
    if np.any(table < 0) or table.sum() == 0:
        return float("nan")
    if (table < 5).any():
        return float(stats.fisher_exact(table)[1])
    return float(stats.chi2_contingency(table, correction=False)[1])


def mcnemar_exact_pvalue(y_true: Iterable[int], p_a: Iterable[float], p_b: Iterable[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    a = (np.asarray(p_a, dtype=float) >= 0.5).astype(int)
    b = (np.asarray(p_b, dtype=float) >= 0.5).astype(int)
    a_correct = a == y
    b_correct = b == y
    b01 = int((a_correct & ~b_correct).sum())
    b10 = int((~a_correct & b_correct).sum())
    n = b01 + b10
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(b01, b10), n=n, p=0.5, alternative="two-sided").pvalue)


def vote_entropy(vote_count: float, model_count: float) -> float:
    if not model_count or model_count <= 0:
        return float("nan")
    p = vote_count / model_count
    if p <= 0 or p >= 1:
        return 0.0
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def save_figure_source(df: pd.DataFrame, figure_name: str) -> None:
    source = config.FIGURE_DIR / f"{Path(figure_name).stem}_source.csv"
    df.to_csv(source, index=False)


def required_output_paths() -> list[Path]:
    names = [
        config.INTERMEDIATE_DIR / "dl_behavior_table.csv",
        config.TABLE_DIR / "table1_model_performance.csv",
        config.TABLE_DIR / "table2_error_cooccurrence.csv",
        config.TABLE_DIR / "table3_consensus_failure_subgroups.csv",
        config.TABLE_DIR / "table4_high_confidence_errors.csv",
        config.TABLE_DIR / "table5_disagreement_selective_referral.csv",
        config.TABLE_DIR / "table6_subgroup_performance.csv",
        config.TABLE_DIR / "table7_external_validation_if_available.csv",
        config.TABLE_DIR / "table8_lndb_selective_referral.csv",
        config.TABLE_DIR / "table9_malignancy_aware_referral.csv",
        config.TABLE_DIR / "consensus_failure_tree_rules_table.csv",
        config.TABLE_DIR / "patient_level_sensitivity.csv",
        config.FIGURE_DIR / "fig1_error_cooccurrence_heatmap.png",
        config.FIGURE_DIR / "fig2_consensus_failure_by_subgroup.png",
        config.FIGURE_DIR / "extra_model_disagreement_distribution.png",
        config.FIGURE_DIR / "fig3_high_confidence_error_rates.png",
        config.FIGURE_DIR / "fig4_risk_coverage_curve.png",
        config.FIGURE_DIR / "fig5_architecture_specific_failure_patterns.png",
        config.FIGURE_DIR / "fig6_external_shift_if_available.png",
        config.FIGURE_DIR / "fig7_lndb_risk_coverage_curve.png",
        config.PAPER_SUMMARY_DIR / "key_findings.md",
        config.PAPER_SUMMARY_DIR / "methods_paragraph.md",
        config.PAPER_SUMMARY_DIR / "results_paragraph.md",
        config.PAPER_SUMMARY_DIR / "limitations.md",
    ]
    return names
