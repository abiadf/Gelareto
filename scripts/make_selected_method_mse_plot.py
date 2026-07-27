#!/usr/bin/env python3
"""Plot the largest observed topology advantage over matched controls."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = Path("latex_tables/pca_kpca_control_summary.csv")
OUTPUT_PDF = Path("images/selected_method_mse_glioblastoma_geoae.pdf")
OUTPUT_PNG = Path("images/selected_method_mse_glioblastoma_geoae.png")


def main() -> None:
    df = pd.read_csv(INPUT)
    row = df[(df["dataset"] == "glioblastoma") & (df["encoder"] == "GeoAE")].iloc[0]

    methods = ["$z$ only", "PCA", "KPCA", "Laplacian", "Diffusion", "RFF", "Topology"]
    means = np.array(
        [
            row["z_mean"],
            row["fuse_pca_mean"],
            row["fuse_kpca_mean"],
            row["best_laplacian_mean"],
            row["best_diffusion_mean"],
            row["best_rff_mean"],
            row["best_topo_mean"],
        ],
        dtype=float,
    )
    stds = np.array(
        [
            row["z_std"],
            row["fuse_pca_std"],
            row["fuse_kpca_std"],
            row["best_laplacian_std"],
            row["best_diffusion_std"],
            row["best_rff_std"],
            row["best_topo_std"],
        ],
        dtype=float,
    )

    colors = ["#64748b"] + ["#d97706"] * 5 + ["#059669"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Forecasting MSE")
    ax.set_title("Glioblastoma with GeoAE")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.99,
        0.96,
        "Selected as the largest observed topology margin",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout()

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
