#!/usr/bin/env python3
"""Plot cached AE, GeoAE, and TopoAE latent trajectories with PCA."""

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


DATASETS = [
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

ENCODERS = [
    ("AE", lambda dataset: dataset),
    ("GeoAE", lambda dataset: f"{dataset}_geoae_lam0.1"),
    ("TopoAE", lambda dataset: f"{dataset}_topoae_lam0.1_signature"),
]


def _load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _payload_path(namespace: str, seed: int, split: str, window: int, bins: int) -> Path:
    directory = Path("models") / namespace / "latent_tda_features"
    candidates = list(directory.glob(f"seed{seed}_{split}_T*_B*_H*_W*_latent*_win{window}_bins{bins}.pt"))
    if not candidates:
        raise FileNotFoundError(f"No cached latent payload found in {directory}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _pca_latents(z: torch.Tensor) -> tuple[np.ndarray, float]:
    z_np = z.detach().cpu().numpy().astype(np.float32)
    time_count, clip_count, latent_dim = z_np.shape
    flat = z_np.reshape(time_count * clip_count, latent_dim)
    flat = StandardScaler().fit_transform(flat)
    pca = PCA(n_components=2)
    xy = pca.fit_transform(flat).reshape(time_count, clip_count, 2)
    return xy, float(pca.explained_variance_ratio_.sum())


def _umap_latents(z: torch.Tensor, seed: int) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP figures require umap-learn.") from exc
    z_np = z.detach().cpu().numpy().astype(np.float32)
    time_count, clip_count, latent_dim = z_np.shape
    flat = StandardScaler().fit_transform(z_np.reshape(time_count * clip_count, latent_dim))
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed, n_jobs=1)
    return reducer.fit_transform(flat).reshape(time_count, clip_count, 2)


def _plot_embedding(
    dataset: str,
    loaded: list[tuple[str, torch.Tensor, np.ndarray, float]],
    embedding: str,
    output_dir: Path,
    seed: int,
) -> None:
    fig, axes = plt.subplots(1, len(loaded), figsize=(12.2, 3.7), constrained_layout=True)
    scatter = None
    for ax, (encoder, z, pca_xy, explained) in zip(axes, loaded):
        xy = pca_xy if embedding == "pca" else _umap_latents(z, seed=seed)
        time_count, clip_count, _ = xy.shape
        time_color = np.repeat(np.arange(time_count), clip_count)
        scatter = ax.scatter(
            xy[:, :, 0].reshape(-1),
            xy[:, :, 1].reshape(-1),
            c=time_color,
            cmap="viridis",
            s=8,
            alpha=0.45,
            linewidths=0,
            rasterized=True,
        )
        title_suffix = f"PC1+PC2: {100.0 * explained:.1f}%" if embedding == "pca" else "UMAP"
        ax.set_title(f"{encoder} ({title_suffix})")
        ax.set_xlabel("PC1" if embedding == "pca" else "UMAP 1")
        ax.set_ylabel("PC2" if embedding == "pca" else "UMAP 2")
        ax.grid(color="#e5e7eb", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"{DATASET_LABELS[dataset]} latent trajectories", fontsize=14)
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="Time step", fraction=0.025, pad=0.02)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{dataset}_latent_{embedding}"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.png')}")
    print(f"Wrote {stem.with_suffix('.pdf')}")


def _make_dataset_figure(
    dataset: str,
    seed: int,
    split: str,
    window: int,
    bins: int,
    output_dir: Path,
    make_umap: bool,
) -> list[dict[str, object]]:
    records = []
    loaded = []
    for encoder, namespace_fn in ENCODERS:
        namespace = namespace_fn(dataset)
        path = _payload_path(namespace, seed, split, window, bins)
        z = _load(path)["z"]
        xy, explained = _pca_latents(z)
        loaded.append((encoder, z, xy, explained))
        records.append(
            {
                "dataset": dataset,
                "encoder": encoder,
                "seed": seed,
                "split": split,
                "time_steps": int(z.shape[0]),
                "clips": int(z.shape[1]),
                "latent_dim": int(z.shape[2]),
                "pca_explained_variance": explained,
                "payload": str(path),
            }
        )

    _plot_embedding(dataset, loaded, embedding="pca", output_dir=output_dir, seed=seed)
    if make_umap:
        _plot_embedding(dataset, loaded, embedding="umap", output_dir=output_dir, seed=seed)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("images/latent_spaces"))
    parser.add_argument("--umap", action="store_true", help="Also generate UMAP figures (requires umap-learn).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    records = []
    for dataset in datasets:
        records.extend(
            _make_dataset_figure(
                dataset,
                seed=args.seed,
                split=args.split,
                window=args.window,
                bins=args.bins,
                output_dir=args.output_dir,
                make_umap=args.umap,
            )
        )
    pd.DataFrame(records).to_csv(args.output_dir / "latent_pca_sources.csv", index=False)
    print(f"Wrote {args.output_dir / 'latent_pca_sources.csv'}")


if __name__ == "__main__":
    main()
