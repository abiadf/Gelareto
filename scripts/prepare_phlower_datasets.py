"""Create compact point clouds from the small PHLOWER example datasets."""

from pathlib import Path

import anndata as ad
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path("datasets/2D/phlower")


def main() -> None:
    """Standardize DLA10 and normalize/PCA-project Fib2Neuro."""
    output = ROOT / "processed"
    output.mkdir(parents=True, exist_ok=True)

    mat = loadmat(ROOT / "DLA_10_TreeData.mat")
    dla = StandardScaler().fit_transform(mat["M"]).astype(np.float32)
    np.savez_compressed(output / "dla10.npz", x=dla, branch=mat["C"].ravel())

    adata = ad.read_h5ad(ROOT / "fib2neuro.h5ad")
    counts = np.asarray(adata.X, dtype=np.float32)
    library = counts.sum(axis=1, keepdims=True)
    normalized = np.log1p(counts / np.maximum(library, 1e-6) * 1e4)
    n_genes = min(2_000, normalized.shape[1])
    variable = np.argpartition(normalized.var(axis=0), -n_genes)[-n_genes:]
    normalized = StandardScaler().fit_transform(normalized[:, variable])
    pca = PCA(n_components=32, random_state=0)
    embedding = pca.fit_transform(normalized).astype(np.float32)
    np.savez_compressed(
        output / "fib2neuro_pca32.npz", x=embedding,
        group=adata.obs["group"].astype(str).to_numpy(),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )
    print({"dla10": dla.shape, "fib2neuro": embedding.shape})


if __name__ == "__main__":
    main()
