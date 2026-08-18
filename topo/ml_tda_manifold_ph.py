"""Manifold-aware Vietoris--Rips persistence for mixed-geometry latents.

Unlike the standard latent-TDA pipeline, this module does not center or scale
the tangent coordinates as Euclidean vectors.  For every temporal window it
constructs the full product-manifold geodesic distance matrix, normalizes that
matrix by its mean off-diagonal distance, and gives it directly to Ripser.
The resulting payload has the same shape as ordinary latent-TDA features, so
forecasting code can compare the two metrics without changing the predictor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ripser import ripser

from topo import ml_tda_latent
from topo.ml_tda_mixedgeo import parse_manifold_signature, product_pairwise_distances


VR_DISTANCES = {"euclidean", "product_manifold"}


def vr_distance_matrix(
    points: torch.Tensor,
    signature: str,
    *,
    vr_distance: str = "product_manifold",
    eps: float = 1e-6,
) -> np.ndarray:
    """Return the normalized distance matrix used to build one VR filtration.

    Both choices share this function and therefore identical normalization.
    ``euclidean`` uses cdist on tangent coordinates; ``product_manifold`` uses
    the geodesic metric of the encoder's fixed H/S/E product signature.
    """
    if points.ndim != 2:
        raise ValueError(f"Expected [points, latent_dim], got {tuple(points.shape)}")
    parse_manifold_signature(signature, points.shape[-1])
    vr_distance = str(vr_distance).lower().strip()
    if vr_distance not in VR_DISTANCES:
        raise ValueError(f"Unknown VR distance {vr_distance!r}; choose from {sorted(VR_DISTANCES)}")
    distances = (
        torch.cdist(points, points)
        if vr_distance == "euclidean"
        else product_pairwise_distances(points, signature, eps=eps)
    )
    if len(points) > 1:
        upper = torch.triu_indices(len(points), len(points), offset=1, device=points.device)
        scale = distances[upper[0], upper[1]].mean().clamp_min(eps)
        distances = distances / scale
    return distances.detach().cpu().numpy().astype(np.float32)


def manifold_distance_matrix(points: torch.Tensor, signature: str, eps: float = 1e-6) -> np.ndarray:
    """Backward-compatible product-manifold distance helper."""
    return vr_distance_matrix(points, signature, vr_distance="product_manifold", eps=eps)


def manifold_window_betti_features(
    z_features: torch.Tensor,
    signature: str,
    *,
    window: int = 6,
    n_bins: int = 16,
    return_diagrams: bool = False,
    vr_distance: str = "product_manifold",
):
    """Compute H0/H1 window features using the selected VR distance."""
    parse_manifold_signature(signature, z_features.shape[-1])
    T, B, _ = z_features.shape
    h0 = np.zeros((T, B, n_bins), dtype=np.float32)
    h1 = np.zeros((T, B, n_bins), dtype=np.float32)
    diagrams_h0 = [[torch.empty((0, 2)) for _ in range(B)] for _ in range(T)]
    diagrams_h1 = [[torch.empty((0, 2)) for _ in range(B)] for _ in range(T)]
    z_cpu = z_features.detach().cpu().float()
    for t in range(T):
        start = max(0, t - int(window) + 1)
        for b in range(B):
            points = z_cpu[start:t + 1, b]
            if len(points) < 2:
                continue
            diagrams = ripser(
                vr_distance_matrix(points, signature, vr_distance=vr_distance),
                maxdim=1,
                distance_matrix=True,
            )["dgms"]
            dgm0 = ml_tda_latent._finite_diagram(diagrams[0])
            dgm1 = (
                ml_tda_latent._finite_diagram(diagrams[1])
                if len(diagrams) > 1 else np.empty((0, 2), dtype=np.float64)
            )
            diagrams_h0[t][b] = torch.from_numpy(dgm0.astype(np.float32))
            diagrams_h1[t][b] = torch.from_numpy(dgm1.astype(np.float32))
            h0[t, b] = ml_tda_latent.betti_curve_from_diag(dgm0, n_bins=n_bins)
            h1[t, b] = ml_tda_latent.betti_curve_from_diag(dgm1, n_bins=n_bins)
        if t in {0, T - 1}:
            print(f"  manifold latent TDA frame {t + 1}/{T}")
    if return_diagrams:
        return torch.from_numpy(h0), torch.from_numpy(h1), {"h0": diagrams_h0, "h1": diagrams_h1}
    return torch.from_numpy(h0), torch.from_numpy(h1)


def _cache_path(seed, split_name, x_subset, signature, vr_distance) -> Path:
    """Use a distinct cache so Euclidean and manifold persistence never mix."""
    cache_dir = Path("models") / ml_tda_latent.DATASET / "manifold_latent_tda_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_{split_name}_T{x_subset.shape[0]}_B{x_subset.shape[1]}_"
        f"H{x_subset.shape[-2]}_W{x_subset.shape[-1]}_latent{ml_tda_latent.LATENT_DIM}_"
        f"win{ml_tda_latent.LATENT_TDA_WINDOW}_bins{ml_tda_latent.LATENT_TDA_BINS}_"
        f"{signature}_vrdist{vr_distance}"
    )
    return cache_dir / f"{tag}.pt"


def load_or_compute_manifold_tda_features(
    seed: int,
    split_name: str,
    x_subset: torch.Tensor,
    encoder,
    signature: str,
    vr_distance: str = "product_manifold",
):
    """Load or compute latent features whose PH uses the requested VR distance."""
    if vr_distance not in VR_DISTANCES:
        raise ValueError(f"Unknown VR distance {vr_distance!r}; choose from {sorted(VR_DISTANCES)}")
    cache_path = _cache_path(seed, split_name, x_subset, signature, vr_distance)
    if cache_path.exists() and not ml_tda_latent.RECOMPUTE_LATENT_TDA_FEATURES:
        print(f"Loading manifold latent-TDA cache: {cache_path}")
        try:
            return torch.load(cache_path, map_location="cpu")
        except Exception:
            return torch.load(cache_path, map_location="cpu", weights_only=False)
    print(f"Computing z + manifold-window TDA for {split_name}...")
    z = ml_tda_latent.encode_video_to_z(x_subset, encoder)
    h0, h1, diagrams = manifold_window_betti_features(
        z,
        signature,
        window=ml_tda_latent.LATENT_TDA_WINDOW,
        n_bins=ml_tda_latent.LATENT_TDA_BINS,
        return_diagrams=True,
        vr_distance=vr_distance,
    )
    payload = {"z": z, "h0": h0, "h1": h1, "diagrams": diagrams}
    torch.save(payload, cache_path)
    print(f"Saved manifold latent-TDA cache: {cache_path}")
    return payload
