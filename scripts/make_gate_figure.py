#!/usr/bin/env python3
"""Visualize group-wise topology gating and its learned dataset-level values."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


INPUT = Path("latex_tables/gate.txt")
OUTPUT_DIR = Path("images")
DATASETS = [
    "bouncing_rings",
    "bouncing_disks",
    "orbiting_rings",
    "orbiting_disks",
    "moving_mnist",
    "lorenz96",
    "electric_devices",
    "glioblastoma",
    "hela",
]
SCENARIOS = ["latent_tda", "geo_latent_tda", "topo_latent_tda"]
SCENARIO_LABELS = {"latent_tda": "AE", "geo_latent_tda": "GeoAE", "topo_latent_tda": "TopoAE"}


def display_dataset(name):
    return {
        "moving_mnist": "Moving MNIST",
        "lorenz96": "Lorenz-96",
        "electric_devices": "ElectricDevices",
        "glioblastoma": "Glioblastoma",
        "hela": "HeLa",
    }.get(name, name.replace("_", " ").title())


def load_gate_means(path):
    values = {}
    for line in path.read_text().splitlines()[1:]:
        fields = line.split()
        if not fields:
            break
        if len(fields) < 8 or fields[1] not in SCENARIOS:
            continue
        values[(fields[0], fields[1])] = (float(fields[3]), float(fields[6]))
    missing = [
        (dataset, scenario)
        for dataset in DATASETS
        for scenario in SCENARIOS
        if (dataset, scenario) not in values
    ]
    if missing:
        raise ValueError(f"Missing gate summaries: {missing}")
    return values


def add_box(ax, xy, width, height, text, color):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=10)


def arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#444444"})


def draw_mechanism(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_box(ax, (0.02, 0.62), 0.11, 0.22, r"$z_t$", "#d9eaf7")
    add_box(ax, (0.02, 0.35), 0.11, 0.18, r"$B_t^{H_0}$", "#fce0cf")
    add_box(ax, (0.02, 0.08), 0.11, 0.18, r"$B_t^{H_1}$", "#e5dcf5")
    add_box(ax, (0.23, 0.31), 0.18, 0.30, "Gate MLP\n+ sigmoid", "#eeeeee")
    add_box(ax, (0.49, 0.43), 0.13, 0.18, r"$g_0, g_1$", "#fff2b2")
    add_box(
        ax,
        (0.49, 0.12),
        0.13,
        0.18,
        r"$g_0B_t^{H_0}$" "\n" r"$g_1B_t^{H_1}$",
        "#f7e6d5",
    )
    add_box(ax, (0.70, 0.31), 0.13, 0.25, "Concatenate\nwith $z_t$", "#dff0dc")
    add_box(ax, (0.89, 0.31), 0.09, 0.25, "LSTM\npredictor", "#d9eaf7")
    for y in (0.73, 0.44, 0.17):
        arrow(ax, (0.13, y), (0.23, 0.46))
    arrow(ax, (0.41, 0.46), (0.49, 0.52))
    arrow(ax, (0.555, 0.43), (0.555, 0.30))
    arrow(ax, (0.62, 0.21), (0.70, 0.40))
    arrow(ax, (0.13, 0.73), (0.70, 0.50))
    arrow(ax, (0.83, 0.435), (0.89, 0.435))
    ax.text(0.5, 0.96, "Group-wise topology gate", ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.01, r"$g_k\approx1$: retain group; $g_k\approx0$: suppress group", ha="center", fontsize=9)


def main():
    values = load_gate_means(INPUT)
    matrix = np.array(
        [
            [value for scenario in SCENARIOS for value in values[(dataset, scenario)]]
            for dataset in DATASETS
        ]
    )

    fig = plt.figure(figsize=(10.8, 8.6))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.2], hspace=0.22)
    draw_mechanism(fig.add_subplot(grid[0]))

    ax = fig.add_subplot(grid[1])
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(DATASETS)), [display_dataset(dataset) for dataset in DATASETS])
    labels = [f"{SCENARIO_LABELS[scenario]}\n$H_0$" for scenario in SCENARIOS]
    labels_h1 = [f"{SCENARIO_LABELS[scenario]}\n$H_1$" for scenario in SCENARIOS]
    interleaved = [label for pair in zip(labels, labels_h1) for label in pair]
    ax.set_xticks(range(6), interleaved)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if value < 0.48 else "black"
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    for boundary in (1.5, 3.5):
        ax.axvline(boundary, color="white", linewidth=2.2)
    ax.set_title("Mean held-out gate values (five seeds)", fontsize=12, pad=10)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    colorbar.set_label("Gate value")
    fig.suptitle("Adaptive selection of latent $H_0$ and $H_1$ topology", fontsize=14, y=0.995)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"topology_gate.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
