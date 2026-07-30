#!/usr/bin/env python3
"""Plot mean forecasting MSE against GeoAE regularization strength."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


INPUT = Path("latex_tables/geoae_lambda.txt")
EXTRA_RESULTS = Path("results/geoae_lambda_extra")
OUTPUT_DIR = Path("images")


def load_results(path: Path):
    results = {}
    regularization = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        match = re.fullmatch(r"lambda=([0-9.eE+-]+)", line)
        if match:
            regularization = float(match.group(1))
            continue
        if not line or regularization is None:
            continue
        fields = line.split()
        if len(fields) < 6:
            continue
        results.setdefault(fields[0], []).append(
            (regularization, float(fields[4]))
        )
    extra_frames = []
    for result_path in EXTRA_RESULTS.glob("*/results.csv"):
        frame = pd.read_csv(result_path)
        required = {"dataset", "geo_ae_lambda", "seed", "mode", "test_mse"}
        if required.issubset(frame.columns):
            extra_frames.append(frame[frame["mode"] == "z"])
    if extra_frames:
        extra = pd.concat(extra_frames, ignore_index=True)
        extra = (
            extra.groupby(["dataset", "geo_ae_lambda"], as_index=False)
            .agg(test_mse=("test_mse", "mean"))
        )
        for row in extra.itertuples(index=False):
            values = {
                regularization: mse
                for regularization, mse in results.get(row.dataset, [])
            }
            values[float(row.geo_ae_lambda)] = float(row.test_mse)
            results[row.dataset] = sorted(values.items())
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
        lambdas, mse = zip(*values)
        ax.plot(
            lambdas,
            mse,
            marker=marker,
            markersize=3.6,
            linewidth=1.35,
            label=display_name(dataset),
        )

    tested_lambdas = sorted(
        {value for values in results.values() for value, _ in values}
    )
    ax.set_xscale("symlog", linthresh=0.001, linscale=0.8)
    ax.set_xlim(left=0)
    ax.set_xticks(tested_lambdas)
    ax.set_xticklabels(
        [f"{value:g}" for value in tested_lambdas],
        rotation=45,
        ha="right",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda_{\mathrm{GeoAE}}$")
    ax.set_ylabel("Prediction MSE")
    ax.set_title(r"Prediction MSE versus $\lambda_{\mathrm{GeoAE}}$, by dataset")
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.55, alpha=0.45)
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.35, alpha=0.25)
    for value in (0.001, 0.01, 0.1, 1.0):
        ax.axvline(value, color="#b0b0b0", linestyle="--", linewidth=0.55, alpha=0.45, zorder=0)
    ax.legend(
        fontsize=7.2,
        ncol=1,
        loc="upper right",
        frameon=True,
        framealpha=0.88,
        handlelength=1.8,
    )
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            OUTPUT_DIR / f"geoae_lambda_mse.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
