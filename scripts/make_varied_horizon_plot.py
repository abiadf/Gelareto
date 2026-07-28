#!/usr/bin/env python3
"""Plot mean forecasting MSE against prediction horizon for all datasets."""

from pathlib import Path
import re

import matplotlib.pyplot as plt


INPUT = Path("latex_tables/varied_horizon.txt")
OUTPUT_DIR = Path("images")


def load_results(path: Path):
    results = {}
    horizon = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        match = re.fullmatch(r"h=(\d+)", line)
        if match:
            horizon = int(match.group(1))
            continue
        if not line or horizon is None:
            continue
        fields = line.split()
        if len(fields) < 6:
            continue
        dataset = fields[0]
        mean_mse = float(fields[4])
        results.setdefault(dataset, []).append((horizon, mean_mse))
    return results


def display_name(name: str) -> str:
    return {
        "bouncing_rings": "Bouncing Rings",
        "bouncing_disks": "Bouncing Disks",
        "orbiting_rings": "Orbiting Rings",
        "orbiting_disks": "Orbiting Disks",
        "moving_mnist": "Moving MNIST",
        "lorenz96": "Lorenz-96",
        "electric_devices": "ElectricDevices",
        "glioblastoma": "Glioblastoma",
        "hela": "HeLa",
    }.get(name, name.replace("_", " ").title())


def main():
    results = load_results(INPUT)
    if len(results) != 9:
        raise ValueError(f"Expected 9 datasets, found {len(results)}")

    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for (dataset, values), marker in zip(results.items(), markers):
        values = sorted(values)
        horizons, mse = zip(*values)
        ax.plot(
            horizons,
            mse,
            marker=marker,
            markersize=3.6,
            linewidth=1.35,
            label=display_name(dataset),
        )

    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("MSE (log scale)")
    ax.set_xticks(sorted({h for values in results.values() for h, _ in values}))
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.55, alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.25)
    ax.legend(
        fontsize=7.2,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        frameon=False,
        columnspacing=1.2,
        handlelength=1.8,
    )
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            OUTPUT_DIR / f"varied_horizon_mse.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
