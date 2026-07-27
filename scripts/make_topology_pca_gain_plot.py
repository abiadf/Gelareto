from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_PATH = Path("latex_tables/pca_kpca_control_summary.csv")
OUTPUT_PDF = Path("images/topology_pca_gain.pdf")
OUTPUT_PNG = Path("images/topology_pca_gain.png")


DATASET_ORDER = [
    "Bouncing disks",
    "Bouncing rings",
    "Orbiting disks",
    "Orbiting rings",
    "Moving MNIST",
    "HeLa",
    "Glioblastoma",
    "Lorenz-96",
    "ElectricDevices",
]
ENCODER_ORDER = ["AE", "GeoAE", "TopoAE"]

DATASET_LABELS = {
    "bouncing_disks": "Bouncing disks",
    "bouncing_rings": "Bouncing rings",
    "orbiting_disks": "Orbiting disks",
    "orbiting_rings": "Orbiting rings",
    "moving_mnist": "Moving MNIST",
    "lorenz96": "Lorenz-96",
    "electric_devices": "ElectricDevices",
    "hela": "HeLa",
    "glioblastoma": "Glioblastoma",
}


def read_pca_control_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows parsed from {path}")
    df["dataset"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])
    df["z_only"] = df["z_mean"]
    df["best_pca"] = df[["z_pca_mean", "fuse_pca_mean"]].min(axis=1)
    df["best_kpca"] = df[["z_kpca_mean", "fuse_kpca_mean"]].min(axis=1)
    df["best_spectral"] = pd.to_numeric(df["best_spectral_mean"], errors="coerce")
    df["best_laplacian"] = pd.to_numeric(df["best_laplacian_mean"], errors="coerce")
    df["best_diffusion"] = pd.to_numeric(df["best_diffusion_mean"], errors="coerce")
    df["best_rff"] = pd.to_numeric(df["best_rff_mean"], errors="coerce")
    df["best_topology"] = df[["fuse_h1_mean", "best_topo_mean"]].min(axis=1)
    for col in ["z_only", "best_pca", "best_kpca", "best_laplacian", "best_diffusion", "best_rff", "best_topology"]:
        df[f"{col}_gain"] = 100.0 * (df["z_only"] - df[col]) / df["z_only"]
    return df


def make_plot(df: pd.DataFrame) -> None:
    df = df.copy()
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    df["encoder"] = pd.Categorical(df["encoder"], ENCODER_ORDER, ordered=True)
    df = df.sort_values(["dataset", "encoder"])

    n_datasets = len(DATASET_ORDER)
    n_encoders = len(ENCODER_ORDER)
    section_gap = 0.45
    encoder_gap = 0.92
    bar_width = 0.22

    centers = []
    dataset_centers = []
    x = 0.0
    for dataset in DATASET_ORDER:
        start = x
        for _encoder in ENCODER_ORDER:
            centers.append(x)
            x += encoder_gap
        dataset_centers.append((start + x - encoder_gap) / 2.0)
        x += section_gap

    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    colors = {
        "best_laplacian_gain": "#d97706",
        "best_diffusion_gain": "#f59e0b",
        "best_rff_gain": "#7c3aed",
        "best_topology_gain": "#059669",
    }
    labels = {
        "best_laplacian_gain": "Laplacian",
        "best_diffusion_gain": "Diffusion",
        "best_rff_gain": "RFF",
        "best_topology_gain": "Best Topology",
    }
    offsets = {
        "best_laplacian_gain": -1.5 * bar_width,
        "best_diffusion_gain": -0.5 * bar_width,
        "best_rff_gain": 0.5 * bar_width,
        "best_topology_gain": 1.5 * bar_width,
    }

    row_lookup = {(row.dataset, row.encoder): row for row in df.itertuples(index=False)}
    for metric in ["best_laplacian_gain", "best_diffusion_gain", "best_rff_gain", "best_topology_gain"]:
        values = []
        for dataset in DATASET_ORDER:
            for encoder in ENCODER_ORDER:
                row = row_lookup.get((dataset, encoder))
                values.append(np.nan if row is None else getattr(row, metric))
        ax.bar(
            np.asarray(centers) + offsets[metric],
            values,
            width=bar_width,
            color=colors[metric],
            label=labels[metric],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Relative MSE reduction vs. $z$ (%)", fontsize=14)
    ax.set_xticks(centers)
    ax.set_xticklabels(ENCODER_ORDER * n_datasets, rotation=90)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=12)

    ymin, ymax = ax.get_ylim()
    y_top = ymax + 0.16 * (ymax - ymin)
    ax.set_ylim(ymin, y_top)
    label_y = ymax + 0.05 * (ymax - ymin)
    for center, dataset in zip(dataset_centers, DATASET_ORDER):
        ax.text(center, label_y, dataset, ha="center", va="bottom", fontsize=11)

    for boundary_idx in range(1, n_datasets):
        boundary = (centers[boundary_idx * n_encoders - 1] + centers[boundary_idx * n_encoders]) / 2.0
        ax.axvline(boundary, color="#d1d5db", linewidth=0.7, zorder=0)

    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=13)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=0.18, top=0.82, left=0.07, right=0.995)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG, dpi=300)


def main() -> None:
    df = read_pca_control_table(INPUT_PATH)
    make_plot(df)
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
