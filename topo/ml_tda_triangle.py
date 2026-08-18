"""Gu-style graph triangle-curvature diagnostic for input point clouds."""

from __future__ import annotations

import numpy as np
import re
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


def _sample_points(x, max_points: int, seed: int) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    points = np.asarray(x, dtype=np.float32).reshape(-1, int(np.prod(x.shape[2:])))
    if len(points) > int(max_points):
        rng = np.random.default_rng(int(seed))
        points = points[np.sort(rng.choice(len(points), int(max_points), replace=False))]
    return points


def knn_graph(points: np.ndarray, knn: int = 8) -> csr_matrix:
    """Build an undirected, unweighted kNN graph."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 3:
        raise ValueError("Triangle curvature requires at least three points")
    k = min(max(1, int(knn)), n - 1)
    squared_norms = np.sum(points * points, axis=1, keepdims=True)
    distances2 = squared_norms + squared_norms.T - 2.0 * points @ points.T
    np.fill_diagonal(distances2, np.inf)
    neighbours = np.argpartition(distances2, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = neighbours.reshape(-1)
    adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.data[:] = 1.0
    return adjacency


def gu_triangle_curvature_samples(
    points: np.ndarray,
    *,
    knn: int = 8,
    n_samples: int = 10_000,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Sample Gu et al.'s normalized median-deviation proxy xi_G (Algorithm 3)."""
    adjacency = knn_graph(points, knn=knn)
    n_components, labels = connected_components(adjacency, directed=False)
    counts = np.bincount(labels)
    largest_label = int(np.argmax(counts))
    keep = np.flatnonzero(labels == largest_label)
    retained_fraction = len(keep) / adjacency.shape[0]
    adjacency = adjacency[keep][:, keep]
    distances = shortest_path(adjacency, directed=False, unweighted=True)
    neighbours = [adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]] for i in range(len(keep))]
    centers = np.asarray([i for i, values in enumerate(neighbours) if len(values) >= 2], dtype=int)
    if len(centers) == 0:
        raise ValueError("kNN graph has no vertices with two distinct neighbors")

    rng = np.random.default_rng(int(seed))
    values = np.empty(int(n_samples), dtype=np.float64)
    for sample_index in range(int(n_samples)):
        m = int(rng.choice(centers))
        b, c = rng.choice(neighbours[m], size=2, replace=False)
        candidates = np.delete(np.arange(len(keep)), m)
        a = int(rng.choice(candidates))
        d_am = distances[a, m]
        xi = d_am**2 + distances[b, c]**2 / 4.0
        xi -= (distances[a, b]**2 + distances[a, c]**2) / 2.0
        values[sample_index] = xi / (2.0 * d_am)
    metadata = {
        "n_input_points": int(len(points)),
        "n_graph_points": int(len(keep)),
        "n_components": int(n_components),
        "largest_component_fraction": float(retained_fraction),
    }
    return values, metadata


def suggested_signature(negative: float, flat: float, positive: float, latent_dim: int) -> str:
    """Convert sign proportions into one reproducible dimension-allocation heuristic."""
    proportions = np.asarray([negative, positive, flat], dtype=np.float64)
    active = proportions >= 0.05
    if not active.any():
        return f"e{int(latent_dim)}"
    proportions = proportions * active
    proportions /= proportions.sum()
    raw = proportions * int(latent_dim)
    dims = np.floor(raw).astype(int)
    for index in np.flatnonzero(active & (dims == 0)):
        dims[index] = 1
    while dims.sum() < int(latent_dim):
        dims[int(np.argmax(raw - dims))] += 1
    while dims.sum() > int(latent_dim):
        eligible = np.flatnonzero(dims > 1)
        dims[int(eligible[np.argmax(dims[eligible] - raw[eligible])])] -= 1
    names = ("h", "s", "e")
    return "_".join(f"{name}{dim}" for name, dim in zip(names, dims) if dim > 0)


def softened_signature_candidates(
    diagnostic_signature: str,
    latent_dim: int,
    strengths=(0.0, 0.5, 0.75, 1.0),
) -> list[str]:
    """Shrink diagnosed H/S dimensions toward E using fixed strengths.

    Curved dimensions are multiplied by each strength and rounded to the
    nearest integer; every released dimension is assigned to the Euclidean
    factor. This applies one pre-registered candidate rule to every dataset
    while retaining its diagnosed hyperbolic-to-spherical ratio.
    """
    parsed = {
        name: int(dim)
        for name, dim in re.findall(r"([hse])(\d+)", diagnostic_signature.lower())
    }
    if sum(parsed.values()) != int(latent_dim):
        raise ValueError(
            f"Diagnostic signature {diagnostic_signature!r} has dimension "
            f"{sum(parsed.values())}, expected {latent_dim}"
        )
    candidates = []
    for strength in strengths:
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Signature shrinkage must lie in [0, 1], got {strength}")
        h_dim = int(np.floor(strength * parsed.get("h", 0) + 0.5))
        s_dim = int(np.floor(strength * parsed.get("s", 0) + 0.5))
        e_dim = int(latent_dim) - h_dim - s_dim
        parts = (
            f"h{h_dim}" if h_dim else "",
            f"s{s_dim}" if s_dim else "",
            f"e{e_dim}" if e_dim else "",
        )
        signature = "_".join(part for part in parts if part)
        if signature not in candidates:
            candidates.append(signature)
    return candidates


def triangle_curvature_diagnostic(
    x,
    *,
    latent_dim: int,
    knn: int = 8,
    n_samples: int = 10_000,
    max_points: int = 512,
    flat_threshold: float = 1e-6,
    seed: int = 0,
) -> dict:
    points = _sample_points(x, max_points=max_points, seed=seed)
    values, metadata = gu_triangle_curvature_samples(
        points, knn=knn, n_samples=n_samples, seed=seed
    )
    threshold = float(flat_threshold)
    negative = float(np.mean(values < -threshold))
    positive = float(np.mean(values > threshold))
    flat = 1.0 - negative - positive
    return {
        **metadata,
        "seed": int(seed),
        "knn": int(knn),
        "n_triangle_samples": int(len(values)),
        "k_mean": float(values.mean()),
        "k_std": float(values.std(ddof=1)),
        "k_q05": float(np.quantile(values, 0.05)),
        "k_q25": float(np.quantile(values, 0.25)),
        "k_median": float(np.median(values)),
        "k_q75": float(np.quantile(values, 0.75)),
        "k_q95": float(np.quantile(values, 0.95)),
        "negative_fraction": negative,
        "flat_fraction": flat,
        "positive_fraction": positive,
        "suggested_signature": suggested_signature(negative, flat, positive, latent_dim),
    }
