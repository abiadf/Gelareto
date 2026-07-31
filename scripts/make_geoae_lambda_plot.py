#!/usr/bin/env python3
"""Plot mean forecasting MSE from the complete GeoAE lambda rerun."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_DIR = Path("results/geoae_lambda_rerun")
OUTPUT_DIR = Path("images")
SEED_OUTPUT = OUTPUT_DIR / "geoae_lambda_mse_seed_results.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "geoae_lambda_mse_summary.csv"
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
LAMBDAS = [0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1]
SEEDS = list(range(5))


def load_results(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for result_path in sorted(path.glob("**/results.csv")):
        frame = pd.read_csv(result_path)
        required = {"dataset", "geo_ae_lambda", "seed", "mode", "test_mse"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{result_path} is missing required columns")
        frames.append(frame[frame["mode"] == "z"])
    if not frames:
        raise FileNotFoundError(f"No result CSV files found below {path}")

    seed_results = pd.concat(frames, ignore_index=True)
    key = ["dataset", "geo_ae_lambda", "seed"]
    expected = pd.MultiIndex.from_product(
        [DATASETS, LAMBDAS, SEEDS], names=key
    )
    actual = pd.MultiIndex.from_frame(seed_results[key])
    if actual.has_duplicates:
        raise ValueError("Duplicate dataset-lambda-seed results found")
    missing = expected.difference(actual)
    unexpected = actual.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(
            f"Incomplete sweep: {len(missing)} missing and "
            f"{len(unexpected)} unexpected combinations"
        )
    if not np.isfinite(seed_results["test_mse"]).all():
        raise ValueError("Non-finite test MSE found")

    seed_results = seed_results.sort_values(key).reset_index(drop=True)
    summary = (
        seed_results.groupby(["dataset", "geo_ae_lambda"], as_index=False)
        .agg(
            test_mse_mean=("test_mse", "mean"),
            test_mse_std=("test_mse", "std"),
            n_seeds=("test_mse", "size"),
        )
    )
    return seed_results, summary


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
    seed_results, summary = load_results(INPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_results.to_csv(SEED_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for dataset, marker in zip(DATASETS, markers):
        values = summary[summary["dataset"] == dataset].sort_values(
            "geo_ae_lambda"
        )
        ax.plot(
            values["geo_ae_lambda"],
            values["test_mse_mean"],
            marker=marker,
            markersize=3.6,
            linewidth=1.35,
            label=display_name(dataset),
        )

    tested_lambdas = LAMBDAS
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

    for suffix in ("pdf", "png"):
        fig.savefig(
            OUTPUT_DIR / f"geoae_lambda_mse.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
