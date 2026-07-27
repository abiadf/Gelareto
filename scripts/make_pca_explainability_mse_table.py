#!/usr/bin/env python3
"""Relate two-component PCA explainability to latent forecasting MSE."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


PCA_PATH = Path("images/latent_spaces/latent_pca_sources.csv")
RESULTS_PATH = Path("latex_tables/pca_kpca_control_summary.csv")
OUTPUT_CSV = Path("latex_tables/pca_explainability_mse_by_dataset.csv")
OUTPUT_TEX = Path("latex_tables/final_table_pca_explainability_mse.txt")

DATASET_ORDER = [
    "bouncing_disks",
    "bouncing_rings",
    "orbiting_disks",
    "orbiting_rings",
    "moving_mnist",
    "lorenz96",
    "electric_devices",
    "glioblastoma",
    "hela",
]

DATASET_LABELS = {
    "bouncing_disks": "Bouncing disks",
    "bouncing_rings": "Bouncing rings",
    "orbiting_disks": "Orbiting disks",
    "orbiting_rings": "Orbiting rings",
    "moving_mnist": "Moving MNIST",
    "lorenz96": "Lorenz-96",
    "electric_devices": "ElectricDevices",
    "glioblastoma": "Glioblastoma",
    "hela": "HeLa",
}

ENCODERS = ["AE", "GeoAE", "TopoAE"]


def main() -> None:
    pca = pd.read_csv(PCA_PATH)
    results = pd.read_csv(RESULTS_PATH)[["dataset", "encoder", "z_mean"]]
    merged = pca.merge(results, on=["dataset", "encoder"], validate="one_to_one")
    merged["pca_pct"] = 100.0 * merged["pca_explained_variance"]
    merged["pca_rank"] = merged.groupby("dataset")["pca_pct"].rank()
    merged["mse_rank"] = merged.groupby("dataset")["z_mean"].rank()

    rows = []
    for dataset in DATASET_ORDER:
        group = merged[merged["dataset"] == dataset].set_index("encoder")
        rho = float(spearmanr(group.loc[ENCODERS, "pca_pct"], group.loc[ENCODERS, "z_mean"]).statistic)
        row = {"dataset": dataset, "spearman_rho": rho}
        for encoder in ENCODERS:
            row[f"{encoder.lower()}_pca_pct"] = float(group.loc[encoder, "pca_pct"])
            row[f"{encoder.lower()}_mse"] = float(group.loc[encoder, "z_mean"])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_CSV, index=False)

    pooled = spearmanr(merged["pca_rank"], merged["mse_rank"])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Two-component PCA explainability and $z$-forecasting MSE. "
            r"Each PCA value is the percentage of latent variance explained by PC1 and PC2. "
            rf"The final column reports Spearman's $\rho$ across the three encoders within each dataset. "
            rf"Pooling within-dataset encoder ranks gives $\rho={float(pooled.statistic):.2f}$; "
            r"this is a descriptive association rather than a causal or independent-sample test.}"
        ),
        r"\label{tab:pca-explainability-mse}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{AE} & \multicolumn{2}{c}{GeoAE} & \multicolumn{2}{c}{TopoAE} & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Dataset & PC1+PC2 (\%) & MSE & PC1+PC2 (\%) & MSE & PC1+PC2 (\%) & MSE & Spearman $\rho$ \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"{DATASET_LABELS[row.dataset]} & "
            f"{row.ae_pca_pct:.1f} & {row.ae_mse:.4f} & "
            f"{row.geoae_pca_pct:.1f} & {row.geoae_mse:.4f} & "
            f"{row.topoae_pca_pct:.1f} & {row.topoae_mse:.4f} & "
            f"{row.spearman_rho:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    OUTPUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Pooled within-dataset rank Spearman rho={pooled.statistic:.4f}, naive p={pooled.pvalue:.6g}")
    print(f"Wrote {OUTPUT_TEX}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
