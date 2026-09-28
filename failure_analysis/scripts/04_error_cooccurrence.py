from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils import MODEL_SLUGS, configure_matplotlib, load_behavior_table, save_figure_source, write_table

configure_matplotlib()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    df = load_behavior_table()
    models = list(config.MODEL_FILES)
    error_cols = [f"{MODEL_SLUGS[m]}_error" for m in models]
    errors = df[error_cols].fillna(False).astype(int)
    errors.columns = models

    cooc = errors.T.dot(errors)
    jaccard = pd.DataFrame(index=models, columns=models, dtype=float)
    conditional = pd.DataFrame(index=models, columns=models, dtype=float)
    rows = []
    for a in models:
        set_a = errors[a].astype(bool)
        for b in models:
            set_b = errors[b].astype(bool)
            inter = int((set_a & set_b).sum())
            union = int((set_a | set_b).sum())
            jacc = inter / union if union else np.nan
            cond = inter / int(set_a.sum()) if int(set_a.sum()) else np.nan
            jaccard.loc[a, b] = jacc
            conditional.loc[a, b] = cond
            if a < b:
                rows.append(
                    {
                        "model_a": a,
                        "model_b": b,
                        "model_a_error_n": int(set_a.sum()),
                        "model_b_error_n": int(set_b.sum()),
                        "co_error_n": inter,
                        "error_union_n": union,
                        "jaccard_error_similarity": jacc,
                        "p_b_wrong_given_a_wrong": cond,
                    }
                )

    table2 = pd.DataFrame(rows).sort_values("jaccard_error_similarity", ascending=False)
    write_table(table2, config.TABLE_DIR / "table2_error_cooccurrence.csv")
    write_table(cooc.reset_index().rename(columns={"index": "model"}), config.TABLE_DIR / "error_cooccurrence_matrix.csv")
    write_table(jaccard.reset_index().rename(columns={"index": "model"}), config.TABLE_DIR / "error_jaccard_matrix.csv")
    write_table(conditional.reset_index().rename(columns={"index": "model"}), config.TABLE_DIR / "conditional_error_probability_matrix.csv")

    top_cols = [
        "case_id",
        "patient_id",
        "y_true",
        "size_mm",
        "size_bin",
        "density",
        "margin",
        "lobe_or_location",
        "age",
        "sex",
        "smoking_status",
        "error_count",
        "model_count",
        "hce_count_0_8",
        "mean_p",
        "std_p",
        "max_p_gap",
        "hard_case_score",
    ]
    p_cols = [f"{MODEL_SLUGS[m]}_p" for m in models]
    top_cases = df[top_cols + p_cols].sort_values(
        ["error_count", "hce_count_0_8", "std_p"], ascending=False
    ).head(100)
    write_table(top_cases, config.TABLE_DIR / "top_consensus_failure_cases.csv")

    fig, ax = plt.subplots(figsize=(8.0, 6.5), dpi=160)
    mat = jaccard.to_numpy(dtype=float)
    diag_error_counts = np.diag(cooc.to_numpy(dtype=float)).astype(int)
    np.fill_diagonal(mat, np.nan)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#f2f2f2")
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=np.nanmax(mat))
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticklabels(models)
    ax.set_title("Pairwise error-set Jaccard overlap")
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                label = f"n={diag_error_counts[i]}"
                color = "black"
            else:
                label = f"{mat[i, j]:.2f}"
                color = "white" if mat[i, j] >= 0.38 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Jaccard")
    fig.tight_layout()
    out_png = config.FIGURE_DIR / "fig1_error_cooccurrence_heatmap.png"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    save_figure_source(jaccard.reset_index().rename(columns={"index": "model"}), out_png.name)
    print("Wrote error co-occurrence matrices, top cases, and heatmap.")


if __name__ == "__main__":
    main()
