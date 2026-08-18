"""Add curvature persistence to latent forecasting without changing the encoder.

Each rolling latent window becomes a kNN flag complex whose vertex filtration is
an unsigned local-PCA curvature proxy. GUDHI lower-star persistence then produces
the optional ``B_curv`` features used in the VR/curvature ablation modes below.
"""

from __future__ import annotations

import numpy as np
import torch


CURVATURE_MODES = {
    "z_curv_h0",
    "z_curv_h1",
    "z_curv_both",
    "z_both_curv_h0",
    "z_both_curv_h1",
    "z_both_curv_both",
}


def is_curvature_mode(mode: str) -> bool:
    return str(mode) in CURVATURE_MODES


def curvature_mode_needs_vr(mode: str) -> bool:
    """Whether a curvature mode also includes the existing VR Betti features."""
    if not is_curvature_mode(mode):
        return False
    return str(mode).startswith("z_both_curv_")


def local_pca_curvature_scores(
    points: np.ndarray,
    *,
    knn: int = 5,
    tangent_dim: int = 1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return unsigned local non-flatness as PCA energy off the tangent space.

    This is a curvature proxy, not signed Ricci/sectional curvature.  For a
    trajectory, ``tangent_dim=1`` measures how strongly each neighborhood bends
    away from its best-fitting line.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"Expected [n_points, n_dimensions], got {points.shape}")
    n_points, ambient_dim = points.shape
    if n_points < 2:
        return np.zeros(n_points, dtype=np.float32)
    if not 1 <= int(tangent_dim) <= ambient_dim:
        raise ValueError(f"tangent_dim must be in [1, {ambient_dim}], got {tangent_dim}")

    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    k = min(max(2, int(knn)), n_points - 1)
    scores = np.zeros(n_points, dtype=np.float64)
    for index in range(n_points):
        neighbours = np.argsort(distances[index], kind="stable")[1:k + 1]
        local = np.concatenate(([index], neighbours))
        offsets = points[local] - points[index]
        covariance = offsets.T @ offsets / max(len(local) - 1, 1)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        total = float(eigenvalues.sum())
        if total > eps:
            scores[index] = float(eigenvalues[:-int(tangent_dim)].sum()) / total
    return scores.astype(np.float32)


def symmetric_knn_edges(points: np.ndarray, *, knn: int = 5) -> list[tuple[int, int]]:
    points = np.asarray(points, dtype=np.float64)
    n_points = len(points)
    if n_points < 2:
        return []
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    k = min(max(1, int(knn)), n_points - 1)
    edges: set[tuple[int, int]] = set()
    for index in range(n_points):
        for neighbour in np.argsort(distances[index], kind="stable")[1:k + 1]:
            edge = tuple(sorted((index, int(neighbour))))
            edges.add(edge)
    return sorted(edges)


def curvature_lower_star_diagrams(
    points: np.ndarray,
    *,
    knn: int = 5,
    tangent_dim: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute H0/H1 lower-star persistence of local curvature on a kNN flag complex."""
    try:
        import gudhi
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Curvature-persistence modes require gudhi.") from exc

    points = np.asarray(points, dtype=np.float64)
    scores = local_pca_curvature_scores(points, knn=knn, tangent_dim=tangent_dim)
    simplex_tree = gudhi.SimplexTree()
    for index, score in enumerate(scores):
        simplex_tree.insert([index], filtration=float(score))
    for left, right in symmetric_knn_edges(points, knn=knn):
        simplex_tree.insert([left, right], filtration=float(max(scores[left], scores[right])))
    # Triangles make H1 deaths meaningful; expansion preserves the flag-complex filtration.
    simplex_tree.expansion(2)
    simplex_tree.make_filtration_non_decreasing()
    simplex_tree.persistence(homology_coeff_field=2, persistence_dim_max=True)

    diagrams = []
    for dimension in (0, 1):
        diagram = np.asarray(simplex_tree.persistence_intervals_in_dimension(dimension), dtype=np.float32)
        if diagram.size == 0:
            diagram = np.empty((0, 2), dtype=np.float32)
        else:
            diagram = diagram.reshape(-1, 2)
            diagram = diagram[np.isfinite(diagram).all(axis=1)]
            diagram = diagram[diagram[:, 1] > diagram[:, 0]]
        diagrams.append(diagram)
    return diagrams[0], diagrams[1], scores


def _betti_curve(diagram: np.ndarray, n_bins: int) -> np.ndarray:
    """Vectorize a [0, 1]-valued curvature filtration on a shared fixed grid."""
    output = np.zeros(int(n_bins), dtype=np.float32)
    if len(diagram) == 0:
        return output
    grid = np.linspace(0.0, 1.0, int(n_bins), dtype=np.float32)
    births = diagram[:, 0:1]
    deaths = diagram[:, 1:2]
    return ((births <= grid) & (grid < deaths)).sum(axis=0).astype(np.float32)


def latent_window_curvature_features(
    z: torch.Tensor,
    *,
    window: int = 20,
    n_bins: int = 16,
    knn: int = 5,
    tangent_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rolling curvature-filtration H0/H1 Betti curves shaped [T, B, bins]."""
    if z.ndim != 3:
        raise ValueError(f"Expected latent trajectories [T, B, D], got {tuple(z.shape)}")
    time_steps, batches, _ = z.shape
    h0 = np.zeros((time_steps, batches, int(n_bins)), dtype=np.float32)
    h1 = np.zeros_like(h0)
    z_numpy = z.detach().cpu().numpy().astype(np.float32)
    for time_index in range(time_steps):
        start = max(0, time_index - int(window) + 1)
        for batch_index in range(batches):
            points = z_numpy[start:time_index + 1, batch_index]
            if len(points) < 3:
                continue
            points = points - points.mean(axis=0, keepdims=True)
            scale = float(points.std(axis=0).mean()) + 1e-6
            diagram_h0, diagram_h1, _ = curvature_lower_star_diagrams(
                points / scale,
                knn=knn,
                tangent_dim=tangent_dim,
            )
            h0[time_index, batch_index] = _betti_curve(diagram_h0, n_bins)
            h1[time_index, batch_index] = _betti_curve(diagram_h1, n_bins)
    return torch.from_numpy(h0), torch.from_numpy(h1)


def _payload_curvature(payload: dict, *, window: int, n_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = f"_curvature_win{int(window)}_bins{int(n_bins)}"
    if cache_key not in payload:
        payload[cache_key] = latent_window_curvature_features(
            payload["z"], window=window, n_bins=n_bins
        )
    return payload[cache_key]


def features_for_curvature_mode(payload: dict, mode: str, *, window: int, n_bins: int) -> torch.Tensor:
    """Assemble z + optional VR + curvature features for one payload."""
    if not is_curvature_mode(mode):
        raise ValueError(f"Unknown curvature mode: {mode}")
    curv_h0, curv_h1 = _payload_curvature(payload, window=window, n_bins=n_bins)
    selector = str(mode).rsplit("_", 1)[-1]
    if selector == "h0":
        curvature = curv_h0
    elif selector == "h1":
        curvature = curv_h1
    else:
        curvature = torch.cat([curv_h0, curv_h1], dim=-1)

    parts = [payload["z"]]
    if curvature_mode_needs_vr(mode):
        if "h0" not in payload or "h1" not in payload:
            raise ValueError(f"Mode {mode} requires existing VR H0/H1 features")
        parts.extend([payload["h0"], payload["h1"]])
    parts.append(curvature)
    return torch.cat(parts, dim=-1)
