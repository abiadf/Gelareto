#!/usr/bin/env python3
"""Analyse the full PCA spectrum of cached AE, GeoAE, and TopoAE latents."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
ENCODER_ORDER = ["AE", "GeoAE", "TopoAE"]
COLORS = {"AE": "#4C78A8", "GeoAE": "#F58518", "TopoAE": "#54A24B"}


def _load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _components_for(cumulative: np.ndarray, threshold: float) -> int:
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def _spectrum(path: Path) -> np.ndarray:
    z = _load(path)["z"].detach().cpu().numpy().astype(np.float64)
    flat = z.reshape(-1, z.shape[-1])
    # This matches the existing PCA diagnostic: each latent coordinate has unit
    # variance, so the spectrum measures correlation/redundancy concentration.
    flat = StandardScaler().fit_transform(flat)
    return PCA().fit(flat).explained_variance_ratio_


def compute_spectra(sources: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    spectrum_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for row in sources.itertuples(index=False):
        variance = _spectrum(Path(row.payload))
        cumulative = np.cumsum(variance)
        positive = variance[variance > 0]
        entropy = -float(np.sum(positive * np.log(positive)))
        effective_rank = float(np.exp(entropy))
        participation_ratio = float(1.0 / np.sum(variance**2))
        common = {
            "dataset": row.dataset,
            "encoder": row.encoder,
            "seed": int(row.seed),
            "latent_dim": int(row.latent_dim),
            "payload": row.payload,
        }
        summary_rows.append(
            {
                **common,
                "pc1_pc2_pct": 100.0 * float(variance[:2].sum()),
                "k50": _components_for(cumulative, 0.50),
                "k90": _components_for(cumulative, 0.90),
                "k95": _components_for(cumulative, 0.95),
                "effective_rank": effective_rank,
                "participation_ratio": participation_ratio,
            }
        )
        for index, (share, total) in enumerate(zip(variance, cumulative), start=1):
            spectrum_rows.append(
                {
                    **common,
                    "component": index,
                    "explained_variance_ratio": float(share),
                    "cumulative_explained_variance": float(total),
                }
            )
    return pd.DataFrame(spectrum_rows), pd.DataFrame(summary_rows)


def _plot_dataset_grid(
    spectra: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.2), sharex=False, sharey=True)
    for ax, dataset in zip(axes.flat, DATASET_ORDER):
        group = spectra[spectra["dataset"] == dataset]
        for encoder in ENCODER_ORDER:
            line = group[group["encoder"] == encoder]
            ax.plot(
                line["component"],
                100.0 * line["cumulative_explained_variance"],
                color=COLORS[encoder],
                linewidth=1.6,
                label=encoder,
            )
        ax.set_title(DATASET_LABELS[dataset], fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.axhline(90, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 102)
    for ax in axes[-1, :]:
        ax.set_xlabel("Principal component")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative variance (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("images/latent_spaces/latent_pca_sources.csv"),
        help="Exact cached payloads used by the existing PCA analysis.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("images/latent_pca_spectrum"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = pd.read_csv(args.sources)
    spectra, summary = compute_spectra(sources)
    spectra.to_csv(args.output_dir / "pca_spectra.csv", index=False)
    summary.to_csv(args.output_dir / "pca_spectrum_summary.csv", index=False)
    _plot_dataset_grid(spectra, args.output_dir / "pca_cumulative_by_dataset")
    print(summary.drop(columns=["payload"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Wrote PCA spectra, summaries, and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
