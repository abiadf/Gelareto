#!/usr/bin/env python3
"""Merge the GeoAE PCA lambda sweep with forecasting MSE and plot both."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPECTRUM_ROOT = Path("results/geoae_pca_sweep")
EXTRA_MSE_ROOT = Path("results/geoae_lambda_extra")
MSE_PATH = Path("latex_tables/geoae_lambda.txt")
OUTPUT_DIR = Path("images/geoae_pca_sweep")
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
LAMBDA_GRID_VALUES = [0.001, 0.01, 0.1, 1.0]


def add_lambda_grid(ax: plt.Axes, linewidth: float = 0.6) -> None:
    """Horizontal grid plus vertical grid only at logarithmic decades."""
    ax.grid(axis="y", alpha=0.25, linewidth=linewidth)
    for value in LAMBDA_GRID_VALUES:
        ax.axvline(value, color="#b0b0b0", alpha=0.25, linewidth=linewidth, zorder=0)


def load_mse(path: Path) -> pd.DataFrame:
    rows = []
    current_lambda = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        match = re.fullmatch(r"lambda=([0-9.eE+-]+)", line)
        if match:
            current_lambda = float(match.group(1))
            continue
        fields = line.split()
        if current_lambda is not None and len(fields) >= 6:
            rows.append(
                {
                    "dataset": fields[0],
                    "geo_ae_lambda": current_lambda,
                    "mse_mean": float(fields[4]),
                }
            )
    data = pd.DataFrame(rows)
    extra_frames = []
    for result_path in EXTRA_MSE_ROOT.glob("*/results.csv"):
        frame = pd.read_csv(result_path)
        required = {"dataset", "geo_ae_lambda", "seed", "mode", "test_mse"}
        if required.issubset(frame.columns):
            frame = frame[frame["mode"] == "z"]
            extra_frames.append(
                frame.groupby(["dataset", "geo_ae_lambda"], as_index=False)
                .agg(mse_mean=("test_mse", "mean"))
            )
    if extra_frames:
        data = pd.concat([data, *extra_frames], ignore_index=True)
        data = data.drop_duplicates(["dataset", "geo_ae_lambda"], keep="last")
    elif (OUTPUT_DIR / "geoae_lambda_pca_summary.csv").exists():
        consolidated = pd.read_csv(OUTPUT_DIR / "geoae_lambda_pca_summary.csv")
        if {"dataset", "geo_ae_lambda", "mse_mean"}.issubset(consolidated.columns):
            extra = consolidated[["dataset", "geo_ae_lambda", "mse_mean"]]
            data = pd.concat([data, extra], ignore_index=True)
            data = data.drop_duplicates(["dataset", "geo_ae_lambda"], keep="last")
    return data


def load_spectra(root: Path) -> pd.DataFrame:
    frames = []
    for path in root.glob("*/results.csv"):
        frame = pd.read_csv(path)
        if {"geo_ae_lambda", "effective_rank", "k90"}.issubset(frame.columns):
            frames.append(frame)
    if not frames:
        consolidated = OUTPUT_DIR / "geoae_lambda_pca_per_seed.csv"
        if consolidated.exists():
            return pd.read_csv(consolidated)
        raise FileNotFoundError(
            f"No spectrum results found under {root}. "
            "Run ./scripts/run_geoae_pca_sweep.sh first."
        )
    data = pd.concat(frames, ignore_index=True)
    keys = ["dataset", "geo_ae_lambda", "seed"]
    return data.sort_values(keys).drop_duplicates(keys, keep="last")


def main() -> None:
    spectra = load_spectra(SPECTRUM_ROOT)
    per_dataset = (
        spectra.groupby(["dataset", "geo_ae_lambda"], as_index=False)
        .agg(
            effective_rank=("effective_rank", "mean"),
            effective_rank_std=("effective_rank", "std"),
            k90=("k90", "mean"),
            k90_std=("k90", "std"),
        )
        .merge(load_mse(MSE_PATH), on=["dataset", "geo_ae_lambda"], validate="one_to_one")
    )
    # Dataset-normalized ranks prevent large-MSE datasets from dominating the visual trend.
    per_dataset["mse_rank"] = per_dataset.groupby("dataset")["mse_mean"].rank(pct=True)
    per_dataset["effective_rank_rank"] = per_dataset.groupby("dataset")["effective_rank"].rank(pct=True)
    per_dataset["relative_mse"] = per_dataset["mse_mean"] / per_dataset.groupby("dataset")[
        "mse_mean"
    ].transform("min")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spectra.to_csv(OUTPUT_DIR / "geoae_lambda_pca_per_seed.csv", index=False)
    per_dataset.to_csv(OUTPUT_DIR / "geoae_lambda_pca_summary.csv", index=False)

    aggregated = per_dataset.groupby("geo_ae_lambda", as_index=False).agg(
        effective_rank=("effective_rank", "mean"),
        k90=("k90", "mean"),
        mse_rank=("mse_rank", "mean"),
        relative_mse=("relative_mse", "mean"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axes[0].plot(
        aggregated["geo_ae_lambda"],
        aggregated["effective_rank"],
        "o-",
        color="#C62828",
    )
    scatter = axes[1].scatter(
        per_dataset["effective_rank"],
        per_dataset["mse_mean"],
        c=per_dataset["geo_ae_lambda"],
        cmap="viridis",
        s=35,
        alpha=0.8,
    )
    axes[0].set_xscale("symlog", linthresh=0.001)
    axes[0].set_xlim(left=0)
    axes[0].set_xlabel(r"$\lambda_{\mathrm{GeoAE}}$")
    add_lambda_grid(axes[0])
    axes[0].set_ylabel("Mean PCA effective rank")
    axes[1].set_xlabel("PCA effective rank")
    axes[1].set_ylabel("z-only forecasting MSE")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[1], label=r"$\lambda_{\mathrm{GeoAE}}$")
    axes[2].plot(
        aggregated["geo_ae_lambda"],
        aggregated["relative_mse"],
        "o-",
        color="#2E7D32",
    )
    axes[2].axhline(1.0, color="#666666", linestyle="--", linewidth=0.9)
    axes[2].set_xscale("symlog", linthresh=0.001)
    axes[2].set_xlim(left=0)
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$\lambda_{\mathrm{GeoAE}}$")
    axes[2].set_ylabel("Mean relative MSE (MSE / dataset minimum)")
    add_lambda_grid(axes[2])
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"geoae_lambda_pca_mse.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.5), sharex=False, sharey=False)
    grid_scatter = None
    for ax, dataset in zip(axes.flat, DATASET_ORDER):
        group = per_dataset[per_dataset["dataset"] == dataset].sort_values("geo_ae_lambda")
        grid_scatter = ax.scatter(
            group["effective_rank"],
            group["mse_mean"],
            c=group["geo_ae_lambda"],
            cmap="viridis",
            vmin=per_dataset["geo_ae_lambda"].min(),
            vmax=per_dataset["geo_ae_lambda"].max(),
            s=48,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.45,
        )
        ax.set_title(DATASET_LABELS[dataset], fontsize=10)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, linewidth=0.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("PCA effective rank")
    for ax in axes[:, 0]:
        ax.set_ylabel("z-only forecasting MSE")
    if grid_scatter is not None:
        color_ax = fig.add_axes([0.925, 0.16, 0.018, 0.70])
        fig.colorbar(grid_scatter, cax=color_ax, label=r"$\lambda_{\mathrm{GeoAE}}$")
    fig.suptitle("GeoAE effective rank versus forecasting error", fontsize=13)
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.08, top=0.93, wspace=0.28, hspace=0.32)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"effective_rank_vs_mse_by_dataset.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.5), sharex=True, sharey=False)
    lambda_ticks = [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    for ax, dataset in zip(axes.flat, DATASET_ORDER):
        group = per_dataset[per_dataset["dataset"] == dataset].sort_values("geo_ae_lambda")
        ax.errorbar(
            group["geo_ae_lambda"],
            group["effective_rank"],
            yerr=group["effective_rank_std"],
            marker="o",
            markersize=4.5,
            linewidth=1.5,
            capsize=2.5,
            color="#C62828",
        )
        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_xticks(lambda_ticks)
        ax.set_xticklabels([f"{value:g}" for value in lambda_ticks], rotation=45, ha="right")
        ax.set_title(DATASET_LABELS[dataset], fontsize=10)
        add_lambda_grid(ax, linewidth=0.5)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\lambda_{\mathrm{GeoAE}}$")
    for ax in axes[:, 0]:
        ax.set_ylabel("PCA effective rank")
    fig.suptitle(r"PCA effective rank versus $\lambda_{\mathrm{GeoAE}}$, by dataset", fontsize=13)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.93, wspace=0.27, hspace=0.38)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"lambda_vs_effective_rank_by_dataset.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Wrote sweep tables and plot to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
