"""Prepare the processed human myogenic single-cell trajectory for geometry analysis.

The source contains genes by cells, cell metadata, and gene metadata.  We retain
well-observed ordering genes, apply the supplied cell size factors, log-transform
counts, standardize genes, and save a compact 32-dimensional PCA point cloud.
No artificial longitudinal cell tracks are created: cells are independent
samples collected at discrete developmental time points.
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path("datasets/2D/myogenic_human")


def main() -> None:
    """Create aligned cell embeddings and metadata under ``processed/``."""
    cells_path = ROOT / "GSE233605_Day0_Day28_trajectory_cells.csv.gz"
    genes_path = ROOT / "GSE233605_Day0_Day28_trajectory_features.csv.gz"
    matrix_path = ROOT / "GSE233605_Day0_Day28_trajectory_matrix.csv.gz"
    cells = pd.read_csv(cells_path, sep=";", index_col=0)
    genes = pd.read_csv(genes_path, sep=";", index_col=0)
    selected = set(
        genes.index[genes["use_for_ordering"].astype(bool) & (genes["num_cells_expressed"] >= 500)]
    )

    rows, gene_ids = [], []
    with gzip.open(matrix_path, "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        matrix_cells = next(reader)
        for row in reader:
            if row[0] in selected:
                gene_ids.append(row[0])
                rows.append(np.asarray(row[1:], dtype=np.float32))
    counts = np.stack(rows).T
    if list(cells.index) != matrix_cells:
        cells = cells.loc[matrix_cells]
    size_factors = cells["Size_Factor"].to_numpy(np.float32)
    normalized = np.log1p(counts / np.maximum(size_factors[:, None], 1e-6))
    normalized = StandardScaler().fit_transform(normalized).astype(np.float32)
    n_components = min(32, normalized.shape[0] - 1, normalized.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    embedding = pca.fit_transform(normalized).astype(np.float32)

    output = ROOT / "processed"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "myogenic_human_pca32.npz",
        x=embedding,
        cell_ids=np.asarray(matrix_cells),
        gene_ids=np.asarray(gene_ids),
        time_point=cells["time.point"].to_numpy(np.int16),
        genotype=cells["genotype"].astype(str).to_numpy(),
        cluster=cells["Cluster"].to_numpy(np.int16),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )
    cells.to_csv(output / "cell_metadata.csv")
    print({
        "cells": len(cells), "selected_genes": len(gene_ids),
        "pca_dimensions": n_components,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "time_points": sorted(cells["time.point"].unique().tolist()),
    })


if __name__ == "__main__":
    main()
