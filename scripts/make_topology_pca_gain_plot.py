from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_PATH = Path("latex_tables/final_table_pca_control.txt")
OUTPUT_PDF = Path("images/topology_pca_gain.pdf")
OUTPUT_PNG = Path("images/topology_pca_gain.png")


DATASET_ORDER = [
    "Bouncing disks",
    "Bouncing rings",
    "Orbiting disks",
    "Orbiting rings",
    "Moving MNIST",
    "Lorenz-96",
    "ElectricDevices",
    "HeLa",
    "Glioblastoma",
]
ENCODER_ORDER = ["AE", "GeoAE", "TopoAE"]


def _strip_latex(text: str) -> str:
    text = text.replace(r"\mathbf{", "")
    text = text.replace("$", "")
    text = text.replace("}", "")
    text = text.replace(r"\\", "")
    return text.strip()


def _mean(cell: str) -> float:
    cleaned = _strip_latex(cell)
    match = re.search(r"[-+]?\d*\.?\d+", cleaned)
    if match is None:
        raise ValueError(f"Could not parse numeric mean from {cell!r}")
    return float(match.group(0))


def read_pca_control_table(path: Path) -> pd.DataFrame:
    rows = []
    current_dataset = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if "&" not in line or r"\pm" not in line:
            continue
        parts = [part.strip() for part in line.split("&")]
        if len(parts) != 7:
            continue
        dataset = _strip_latex(parts[0])
        if dataset:
            current_dataset = dataset
        if current_dataset is None:
            continue
        encoder = _strip_latex(parts[1])
        rows.append(
            {
                "dataset": current_dataset,
                "encoder": encoder,
                "z": _mean(parts[2]),
                "z_h1": _mean(parts[3]),
                "z_pca": _mean(parts[4]),
                "fuse_h1": _mean(parts[5]),
                "fuse_pca": _mean(parts[6]),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No rows parsed from {path}")
    df["best_topology"] = df[["z_h1", "fuse_h1"]].min(axis=1)
    df["best_pca"] = df[["z_pca", "fuse_pca"]].min(axis=1)
    df["z_only"] = df["z"]
    for col in ["z_only", "best_pca", "best_topology"]:
        df[f"{col}_gain"] = 100.0 * (df["z"] - df[col]) / df["z"]
    return df


def make_plot(df: pd.DataFrame) -> None:
    df = df.copy()
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    df["encoder"] = pd.Categorical(df["encoder"], ENCODER_ORDER, ordered=True)
    df = df.sort_values(["dataset", "encoder"])

    n_datasets = len(DATASET_ORDER)
    n_encoders = len(ENCODER_ORDER)
    section_gap = 0.45
    encoder_gap = 0.74
    bar_width = 0.30

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

    fig, ax = plt.subplots(figsize=(13.6, 5.4))
    colors = {
        "z_only_gain": "#9ca3af",
        "best_pca_gain": "#4f46e5",
        "best_topology_gain": "#059669",
    }
    labels = {
        "z_only_gain": r"$z$",
        "best_pca_gain": "Best PCA",
        "best_topology_gain": "Best Topology",
    }
    offsets = {
        "z_only_gain": -bar_width,
        "best_pca_gain": 0.0,
        "best_topology_gain": bar_width,
    }

    row_lookup = {(row.dataset, row.encoder): row for row in df.itertuples(index=False)}
    for metric in ["z_only_gain", "best_pca_gain", "best_topology_gain"]:
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
    top_pad = 0.12 * (ymax - ymin)
    ax.set_ylim(ymin, ymax + top_pad)
    label_y = ymax + 0.04 * (ymax - ymin)
    for center, dataset in zip(dataset_centers, DATASET_ORDER):
        ax.text(center, label_y, dataset, ha="center", va="bottom", fontsize=11)

    for boundary_idx in range(1, n_datasets):
        boundary = (centers[boundary_idx * n_encoders - 1] + centers[boundary_idx * n_encoders]) / 2.0
        ax.axvline(boundary, color="#d1d5db", linewidth=0.7, zorder=0)

    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=13)
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
