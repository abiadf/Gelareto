"""Latent-trajectory TDA helpers for video forecasting experiments."""

from pathlib import Path
import contextlib
import hashlib
import random

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from persim import bottleneck
except ImportError:  # pragma: no cover - optional diagnostic dependency
    bottleneck = None

try:
    from sklearn.manifold import trustworthiness
except ImportError:  # pragma: no cover - optional diagnostic dependency
    trustworthiness = None

try:
    from gudhi.representations import Landscape, PersistenceImage
except ImportError:  # pragma: no cover - optional vectorization dependency
    Landscape = None
    PersistenceImage = None

import topo.ml_tda as ml_tda
from topo.utils import tqdm_progress_bar
from topo.ml_tda import (
    SpatialEncoder,
    SpatialDecoder,
    TopologicalPredictor,
    make_predictor,
    pretrain_spatial_encoder,
    summarize_metric_runs,
)


DATASET = "default"
LATENT_DIM = 128
HIDDEN_DIM = 128
RETRAIN_ENCODER = False
HORIZON = 1
LATENT_TDA_WINDOW = 20
LATENT_TDA_BINS = 16
LATENT_TDA_PREDICTOR_EPOCHS = 10
LATENT_TDA_LR = 3e-4
RETRAIN_LATENT_TDA_PREDICTOR = True
RECOMPUTE_LATENT_TDA_FEATURES = True
STANDARDIZE_LATENT_PREDICTOR = True
PERSISTENCE_IMAGE_RESOLUTION = (8, 8)
PERSISTENCE_IMAGE_BANDWIDTH = 0.1
PERSISTENCE_LANDSCAPE_NUM = 5
PERSISTENCE_LANDSCAPE_RESOLUTION = 32
PERSLAY_OUT_DIM = 64
PERSLAY_SIGMA = 0.25
ACTIVE_PROFILER = None


def _profile_phase(name):
    if ACTIVE_PROFILER is None:
        return contextlib.nullcontext()
    return ACTIVE_PROFILER.phase(name)


def configure_runtime(**kwargs):
    """Update notebook-controlled globals used by latent-TDA helpers."""
    globals().update(kwargs)
    ml_tda.configure_runtime(
        DATASET=globals().get("DATASET", DATASET),
        LATENT_DIM=globals().get("LATENT_DIM", LATENT_DIM),
        HORIZON=globals().get("HORIZON", HORIZON),
    )


def take_batch_subset(X, max_batch, seed=0):
    if max_batch is None or X.shape[1] <= max_batch:
        return X
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[1], generator=g)[:max_batch]
    return X[:, idx]


def shared_encoder_path_for_video(seed, X):
    encoder_tag = f"seed{seed}_T{X.shape[0]}_B{X.shape[1]}_H{X.shape[-2]}_W{X.shape[-1]}_latent{LATENT_DIM}"
    return Path("models") / DATASET / "encoders" / f"encoder_{encoder_tag}.pt"


def load_or_train_shared_encoder_for_latent_tda(seed, X_train_subset, X_train_for_tag=None):
    encoder = SpatialEncoder(latent_dim=LATENT_DIM)
    decoder = SpatialDecoder(latent_dim=LATENT_DIM, output_size=X_train_subset.shape[-2:])
    tag_source = X_train_subset if X_train_for_tag is None else X_train_for_tag
    encoder_path = shared_encoder_path_for_video(seed, tag_source)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)

    if encoder_path.exists() and not RETRAIN_ENCODER:
        print(f"Loading shared encoder: {encoder_path}")
        encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    else:
        reason = "retraining" if encoder_path.exists() else "missing; training once"
        print(f"Shared encoder {reason}: {encoder_path}")
        with _profile_phase("encoder_train"):
            pretrain_spatial_encoder(encoder, decoder, X_train_subset, ae_epochs=int(getattr(ml_tda, "AE_EPOCHS", 3)))
        torch.save(encoder.state_dict(), encoder_path)

    encoder.to(ml_tda.get_runtime_device())
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


@torch.no_grad()
def encode_video_to_z(X, encoder, frame_batch_size=1024):
    with _profile_phase("encoding"):
        device = ml_tda.get_runtime_device()
        encoder.to(device)
        T, B = X.shape[:2]
        frames = X.reshape(T * B, *X.shape[2:])
        chunks = []
        batches = range(0, len(frames), frame_batch_size)
        n_batches = (len(frames) + frame_batch_size - 1) // frame_batch_size
        for start in tqdm_progress_bar(batches, desc="Encode z batches", total=n_batches):
            frame_batch = ml_tda.tensor_to_model_float(frames[start:start + frame_batch_size]).to(device)
            chunks.append(encoder(frame_batch).cpu())
        return torch.cat(chunks, dim=0).reshape(T, B, -1)


def betti_curve_from_diag(diag, n_bins=16):
    if diag is None or len(diag) == 0:
        return np.zeros(n_bins, dtype=np.float32)
    diag = np.asarray(diag, dtype=np.float32)
    finite = diag[np.isfinite(diag).all(axis=1)]
    finite = finite[finite[:, 1] > finite[:, 0]]
    if len(finite) == 0:
        return np.zeros(n_bins, dtype=np.float32)
    max_death = float(finite[:, 1].max())
    if max_death <= 0:
        return np.zeros(n_bins, dtype=np.float32)
    grid = np.linspace(0.0, max_death, n_bins, dtype=np.float32)
    births = finite[:, 0:1]
    deaths = finite[:, 1:2]
    return ((births <= grid) & (grid < deaths)).sum(axis=0).astype(np.float32)


def _finite_diagram(diag):
    if diag is None or len(diag) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if torch.is_tensor(diag):
        diag = diag.detach().cpu().numpy()
    diag = np.asarray(diag, dtype=np.float64)
    finite = diag[np.isfinite(diag).all(axis=1)]
    finite = finite[finite[:, 1] > finite[:, 0]]
    return finite.astype(np.float64, copy=False)


def latent_window_betti_features(z_features, window=6, n_bins=16, return_diagrams=False):
    T, B, _ = z_features.shape
    h0 = np.zeros((T, B, n_bins), dtype=np.float32)
    h1 = np.zeros((T, B, n_bins), dtype=np.float32)
    diagrams_h0 = [[torch.empty((0, 2), dtype=torch.float32) for _ in range(B)] for _ in range(T)]
    diagrams_h1 = [[torch.empty((0, 2), dtype=torch.float32) for _ in range(B)] for _ in range(T)]
    z_np = z_features.detach().cpu().numpy().astype(np.float32)

    for t in range(T):
        start = max(0, t - window + 1)
        for b in range(B):
            pts = z_np[start:t + 1, b, :]
            if len(pts) < 2:
                continue
            pts = pts - pts.mean(axis=0, keepdims=True)
            scale = pts.std(axis=0, keepdims=True).mean() + 1e-6
            pts = pts / scale
            diffs = pts[:, None, :] - pts[None, :, :]
            dmat = np.sqrt(np.sum(diffs * diffs, axis=-1)).astype(np.float32)
            dgms = ripser(dmat, maxdim=1, distance_matrix=True)["dgms"]
            dgm0 = _finite_diagram(dgms[0])
            dgm1 = _finite_diagram(dgms[1]) if len(dgms) > 1 else np.empty((0, 2), dtype=np.float64)
            diagrams_h0[t][b] = torch.from_numpy(dgm0.astype(np.float32))
            diagrams_h1[t][b] = torch.from_numpy(dgm1.astype(np.float32))
            h0[t, b] = betti_curve_from_diag(dgm0, n_bins=n_bins)
            h1[t, b] = betti_curve_from_diag(dgm1, n_bins=n_bins)
        if t in {0, T - 1}:
            print(f"  latent TDA frame {t + 1}/{T}")

    if return_diagrams:
        return torch.from_numpy(h0), torch.from_numpy(h1), {"h0": diagrams_h0, "h1": diagrams_h1}
    return torch.from_numpy(h0), torch.from_numpy(h1)


def _standardized_latent_window_points(z_np, t, b, window):
    start = max(0, t - window + 1)
    pts = z_np[start:t + 1, b, :].astype(np.float32, copy=True)
    if len(pts) < 2:
        return None
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = pts.std(axis=0, keepdims=True).mean() + 1e-6
    return pts / scale


def _hausdorff_distance(points_a, points_b):
    if points_a is None or points_b is None or len(points_a) == 0 or len(points_b) == 0:
        return np.nan
    diffs = points_a[:, None, :] - points_b[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=-1))
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def _window_persistence_diagrams(points):
    if points is None or len(points) < 2:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    diffs = points[:, None, :] - points[None, :, :]
    dmat = np.sqrt(np.sum(diffs * diffs, axis=-1)).astype(np.float32)
    diagrams = ripser(dmat, maxdim=1, distance_matrix=True)["dgms"]
    h0 = _finite_diagram(diagrams[0])
    h1 = _finite_diagram(diagrams[1]) if len(diagrams) > 1 else np.empty((0, 2), dtype=np.float64)
    return h0, h1


def _bottleneck_distance(diagram_a, diagram_b):
    if bottleneck is None:
        raise ImportError("latent_stability requires persim. Install it with `pip install persim`.")
    diagram_a = _finite_diagram(diagram_a)
    diagram_b = _finite_diagram(diagram_b)
    if len(diagram_a) == 0 and len(diagram_b) == 0:
        return 0.0
    return float(bottleneck(diagram_a, diagram_b))


def mean_normalized_distance_matrix(points, eps=1e-12):
    """Pairwise Euclidean distances normalized by their mean off-diagonal value."""
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"Expected a 2D point matrix, got shape={points.shape}")
    n_points = int(points.shape[0])
    if n_points < 2:
        return np.zeros((n_points, n_points), dtype=np.float32), 0.0
    distances = squareform(pdist(points, metric="euclidean"))
    off_diagonal = distances[np.triu_indices(n_points, k=1)]
    scale = float(off_diagonal.mean())
    if not np.isfinite(scale) or scale <= eps:
        return np.zeros_like(distances, dtype=np.float32), scale
    return (distances / scale).astype(np.float32), scale


def metric_distortion(distance_x, distance_z):
    """RMS discrepancy between corresponding upper-triangular distances."""
    distance_x = np.asarray(distance_x, dtype=np.float64)
    distance_z = np.asarray(distance_z, dtype=np.float64)
    if distance_x.shape != distance_z.shape:
        raise ValueError(f"Distance-matrix shapes differ: {distance_x.shape} != {distance_z.shape}")
    if distance_x.ndim != 2 or distance_x.shape[0] != distance_x.shape[1]:
        raise ValueError(f"Expected matching square matrices, got shape={distance_x.shape}")
    if distance_x.shape[0] < 2:
        return 0.0
    upper = np.triu_indices(distance_x.shape[0], k=1)
    return float(np.sqrt(np.mean((distance_x[upper] - distance_z[upper]) ** 2)))


def _diagrams_from_distance_matrix(distances):
    distances = np.asarray(distances, dtype=np.float32)
    if distances.shape[0] < 2:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    diagrams = ripser(distances, maxdim=1, distance_matrix=True)["dgms"]
    h0 = _finite_diagram(diagrams[0])
    h1 = _finite_diagram(diagrams[1]) if len(diagrams) > 1 else np.empty((0, 2), dtype=np.float64)
    return h0, h1


def representation_fidelity_for_window(frames, latents):
    """Compare input- and latent-space metric/VR persistence for one window."""
    if torch.is_tensor(frames):
        frames = ml_tda.tensor_to_model_float(frames).detach().cpu().numpy()
    if torch.is_tensor(latents):
        latents = latents.detach().cpu().numpy()
    frames = np.asarray(frames)
    latents = np.asarray(latents)
    if len(frames) != len(latents):
        raise ValueError(f"Window lengths differ: {len(frames)} != {len(latents)}")

    x_points = frames.reshape(len(frames), -1)
    z_points = latents.reshape(len(latents), -1)
    distance_x, x_scale = mean_normalized_distance_matrix(x_points)
    distance_z, z_scale = mean_normalized_distance_matrix(z_points)
    x_h0, x_h1 = _diagrams_from_distance_matrix(distance_x)
    z_h0, z_h1 = _diagrams_from_distance_matrix(distance_z)
    return {
        "metric_distortion": metric_distortion(distance_x, distance_z),
        "h0_bottleneck": _bottleneck_distance(x_h0, z_h0),
        "h1_bottleneck": _bottleneck_distance(x_h1, z_h1),
        "x_distance_scale": float(x_scale),
        "z_distance_scale": float(z_scale),
        "x_h0_count": int(len(x_h0)),
        "z_h0_count": int(len(z_h0)),
        "x_h1_count": int(len(x_h1)),
        "z_h1_count": int(len(z_h1)),
        "h1_any_nonempty": bool(len(x_h1) or len(z_h1)),
        "h1_both_nonempty": bool(len(x_h1) and len(z_h1)),
    }


def _sample_full_window_indices(sequence_length, n_clips, window, n_windows, seed):
    window = int(window)
    if window < 2:
        raise ValueError(f"representation fidelity requires window >= 2, got {window}")
    if sequence_length < window:
        raise ValueError(
            f"Sequence length {sequence_length} is shorter than fidelity window {window}"
        )
    candidates = [(end, clip) for end in range(window - 1, sequence_length) for clip in range(n_clips)]
    if n_windows is None or int(n_windows) >= len(candidates):
        return candidates
    if int(n_windows) <= 0:
        raise ValueError(f"fidelity n_windows must be positive, got {n_windows}")
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(len(candidates), size=int(n_windows), replace=False))
    return [candidates[int(index)] for index in selected]


def run_representation_fidelity_diagnostic(
    X_test,
    run_seeds=None,
    encoder_loader=None,
    encoder_label="ae",
    window=20,
    n_windows=100,
):
    """Measure input-to-latent metric distortion and H0/H1 persistence fidelity."""
    if X_test is None:
        empty = pd.DataFrame()
        return empty, empty
    if bottleneck is None:
        raise ImportError("representation_fidelity requires persim. Install it with `pip install persim`.")

    run_seeds = list(range(5)) if run_seeds is None else list(run_seeds)
    rows = []
    T, B = X_test.shape[:2]
    for seed in tqdm_progress_bar(run_seeds, desc="Representation fidelity seeds", total=len(run_seeds), leave=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if encoder_loader is None:
            raise ValueError("representation_fidelity requires an encoder_loader")
        encoder = encoder_loader(seed)
        z_test = encode_video_to_z(X_test, encoder)
        sample_seed = _stable_int_seed(DATASET, seed, window, "representation_fidelity_windows")
        indices = _sample_full_window_indices(T, B, window, n_windows, sample_seed)
        for window_id, (end, clip) in enumerate(indices):
            start = end - int(window) + 1
            metrics = representation_fidelity_for_window(
                X_test[start:end + 1, clip],
                z_test[start:end + 1, clip],
            )
            rows.append(
                {
                    "dataset": DATASET,
                    "encoder": encoder_label,
                    "seed": int(seed),
                    "window_id": int(window_id),
                    "clip": int(clip),
                    "window_start": int(start),
                    "window_end": int(end),
                    "window_size": int(window),
                    **metrics,
                }
            )

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df, pd.DataFrame()
    summary_rows = []
    metric_cols = ["metric_distortion", "h0_bottleneck", "h1_bottleneck"]
    for (dataset, encoder, seed), group in results_df.groupby(
        ["dataset", "encoder", "seed"], sort=False
    ):
        row = {
            "dataset": dataset,
            "encoder": encoder,
            "seed": int(seed),
            "window_size": int(group["window_size"].iloc[0]),
            "n_windows": int(len(group)),
            "h1_any_nonempty_fraction": float(group["h1_any_nonempty"].mean()),
            "h1_both_nonempty_fraction": float(group["h1_both_nonempty"].mean()),
        }
        for metric in metric_cols:
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        summary_rows.append(row)
    return results_df, pd.DataFrame(summary_rows)


def run_latent_stability_diagnostic(
    X_train=None,
    X_test=None,
    run_seeds=None,
    noise_levels=None,
    max_train=None,
    max_test=None,
    encoder_loader=None,
    encoder_label="ae",
):
    """Measure latent point-cloud and VR diagram stability under input noise."""
    if X_train is None or X_test is None:
        empty = pd.DataFrame()
        return empty, empty

    run_seeds = list(range(3)) if run_seeds is None else list(run_seeds)
    noise_levels = [0.01, 0.03, 0.05, 0.10] if noise_levels is None else list(noise_levels)
    rows = []

    for seed in tqdm_progress_bar(run_seeds, desc="Latent stability seeds", total=len(run_seeds), leave=True):
        print(f"\n================ latent stability seed={seed} ================")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        Xtr = take_batch_subset(X_train, max_train, seed=seed)
        Xte = take_batch_subset(X_test, max_test, seed=seed + 1)
        if encoder_loader is None:
            encoder = load_or_train_shared_encoder_for_latent_tda(seed, Xtr, X_train)
        else:
            encoder = encoder_loader(seed, Xtr, X_train)
        clean_x = ml_tda.tensor_to_model_float(Xte).clamp(0.0, 1.0)
        clean_z = encode_video_to_z(clean_x, encoder).detach().cpu().numpy().astype(np.float32)

        for sigma in noise_levels:
            generator = torch.Generator().manual_seed(_stable_int_seed(DATASET, seed, sigma, "latent_stability"))
            noise = torch.randn(clean_x.shape, generator=generator, dtype=clean_x.dtype) * float(sigma)
            noisy_x = (clean_x + noise).clamp(0.0, 1.0)
            noisy_z = encode_video_to_z(noisy_x, encoder).detach().cpu().numpy().astype(np.float32)

            T, B = clean_z.shape[:2]
            window_rows = []
            for t in range(T):
                for b in range(B):
                    clean_pts = _standardized_latent_window_points(clean_z, t, b, LATENT_TDA_WINDOW)
                    noisy_pts = _standardized_latent_window_points(noisy_z, t, b, LATENT_TDA_WINDOW)
                    if clean_pts is None or noisy_pts is None:
                        continue
                    clean_h0, clean_h1 = _window_persistence_diagrams(clean_pts)
                    noisy_h0, noisy_h1 = _window_persistence_diagrams(noisy_pts)
                    window_rows.append(
                        {
                            "latent_hausdorff": _hausdorff_distance(clean_pts, noisy_pts),
                            "h0_bottleneck": _bottleneck_distance(clean_h0, noisy_h0),
                            "h1_bottleneck": _bottleneck_distance(clean_h1, noisy_h1),
                        }
                    )
            if window_rows:
                window_df = pd.DataFrame(window_rows)
                latent_hausdorff = float(window_df["latent_hausdorff"].mean())
                h0_bottleneck = float(window_df["h0_bottleneck"].mean())
                h1_bottleneck = float(window_df["h1_bottleneck"].mean())
                rows.append(
                    {
                        "dataset": DATASET,
                        "encoder": encoder_label,
                        "seed": seed,
                        "sigma": float(sigma),
                        "n_windows": int(len(window_df)),
                        "latent_hausdorff": latent_hausdorff,
                        "h0_bottleneck": h0_bottleneck,
                        "h1_bottleneck": h1_bottleneck,
                        "h0_ratio": h0_bottleneck / latent_hausdorff if latent_hausdorff > 1e-12 else np.nan,
                        "h1_ratio": h1_bottleneck / latent_hausdorff if latent_hausdorff > 1e-12 else np.nan,
                    }
                )
            print(f"latent stability sigma={float(sigma):.3f} done")

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df, results_df
    summary_df = summarize_metric_runs(
        results_df,
        group_cols=["encoder", "sigma"],
        metric_cols=["latent_hausdorff", "h0_bottleneck", "h1_bottleneck", "h0_ratio", "h1_ratio"],
    )
    print("\nlatent_stability per-seed results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nlatent_stability mean +/- std:")
    print(summary_df.to_string(float_format=lambda value: f"{value:.4f}"))
    return results_df, summary_df


def latent_tda_cache_path(seed, split_name, X_subset):
    cache_dir = Path("models") / DATASET / "latent_tda_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_{split_name}_T{X_subset.shape[0]}_B{X_subset.shape[1]}_"
        f"H{X_subset.shape[-2]}_W{X_subset.shape[-1]}_latent{LATENT_DIM}_"
        f"win{LATENT_TDA_WINDOW}_bins{LATENT_TDA_BINS}"
    )
    return cache_dir / f"{tag}.pt"


def latent_z_cache_path(seed, split_name, X_subset):
    cache_dir = Path("models") / DATASET / "latent_tda_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_{split_name}_T{X_subset.shape[0]}_B{X_subset.shape[1]}_"
        f"H{X_subset.shape[-2]}_W{X_subset.shape[-1]}_latent{LATENT_DIM}_zonly"
    )
    return cache_dir / f"{tag}.pt"


def load_or_compute_latent_z_features(seed, split_name, X_subset, encoder):
    cache_path = latent_z_cache_path(seed, split_name, X_subset)
    if cache_path.exists() and not RECOMPUTE_LATENT_TDA_FEATURES:
        print(f"Loading z-only cache: {cache_path}")
        try:
            return torch.load(cache_path, map_location="cpu")
        except Exception:
            return torch.load(cache_path, map_location="cpu", weights_only=False)

    print(f"Computing z-only features for {split_name}...")
    payload = {"z": encode_video_to_z(X_subset, encoder)}
    torch.save(payload, cache_path)
    print(f"Saved z-only cache: {cache_path}")
    return payload


def load_or_compute_latent_tda_features(seed, split_name, X_subset, encoder, require_persistence=True):
    if not require_persistence:
        return load_or_compute_latent_z_features(seed, split_name, X_subset, encoder)

    cache_path = latent_tda_cache_path(seed, split_name, X_subset)
    if cache_path.exists() and not RECOMPUTE_LATENT_TDA_FEATURES:
        print(f"Loading latent TDA cache: {cache_path}")
        try:
            return torch.load(cache_path, map_location="cpu")
        except Exception:
            return torch.load(cache_path, map_location="cpu", weights_only=False)

    print(f"Computing z + latent-window TDA for {split_name}...")
    z = encode_video_to_z(X_subset, encoder)
    h0, h1, diagrams = latent_window_betti_features(
        z,
        window=LATENT_TDA_WINDOW,
        n_bins=LATENT_TDA_BINS,
        return_diagrams=True,
    )
    payload = {"z": z, "h0": h0, "h1": h1, "diagrams": diagrams}
    torch.save(payload, cache_path)
    print(f"Saved latent TDA cache: {cache_path}")
    return payload


def latent_window_temporal_stats(z, window=6):
    """Simple non-TDA trajectory summaries for each latent time point."""
    T, B, D = z.shape
    rolling_mean = torch.zeros_like(z)
    rolling_std = torch.zeros_like(z)
    velocity = torch.zeros_like(z)
    acceleration = torch.zeros_like(z)

    for t in range(T):
        start = max(0, t - window + 1)
        window_z = z[start:t + 1]
        rolling_mean[t] = window_z.mean(dim=0)
        rolling_std[t] = window_z.std(dim=0, unbiased=False) if window_z.shape[0] > 1 else 0.0
        if t >= 1:
            velocity[t] = z[t] - z[t - 1]
        if t >= 2:
            acceleration[t] = z[t] - 2 * z[t - 1] + z[t - 2]

    return torch.cat([rolling_mean, rolling_std, velocity, acceleration], dim=-1)


def latent_window_pca_eigenvalue_features(z, window=6, n_components=16):
    """Top PCA eigenvalues of each latent trajectory window as non-topological controls."""
    T, B, _ = z.shape
    n_components = int(n_components)
    features = np.zeros((T, B, n_components), dtype=np.float32)
    z_np = z.detach().cpu().numpy().astype(np.float32)

    for t in range(T):
        start = max(0, t - window + 1)
        for b in range(B):
            pts = z_np[start:t + 1, b, :]
            if len(pts) < 2:
                continue
            pts = pts - pts.mean(axis=0, keepdims=True)
            scale = pts.std(axis=0, keepdims=True).mean() + 1e-6
            pts = pts / scale
            denom = max(len(pts) - 1, 1)
            # The window is usually much shorter than latent_dim, so the Gram
            # matrix gives the same nonzero eigenvalues as the full covariance.
            gram = (pts @ pts.T) / float(denom)
            eigvals = np.linalg.eigvalsh(gram).astype(np.float32)
            eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
            k = min(n_components, len(eigvals))
            features[t, b, :k] = eigvals[:k]
    return torch.from_numpy(features).float()


def latent_window_kpca_eigenvalue_features(z, window=6, n_components=16, kernel="rbf", gamma=None):
    """Top kernel-PCA eigenvalues of each latent trajectory window as nonlinear controls."""
    T, B, D = z.shape
    n_components = int(n_components)
    features = np.zeros((T, B, n_components), dtype=np.float32)
    z_np = z.detach().cpu().numpy().astype(np.float32)
    kernel = str(kernel).lower()

    for t in range(T):
        start = max(0, t - window + 1)
        for b in range(B):
            pts = z_np[start:t + 1, b, :]
            if len(pts) < 2:
                continue
            pts = pts - pts.mean(axis=0, keepdims=True)
            scale = pts.std(axis=0, keepdims=True).mean() + 1e-6
            pts = pts / scale
            if kernel == "rbf":
                gamma_value = float(gamma) if gamma is not None else 1.0 / max(int(D), 1)
                sq_norm = np.sum(pts * pts, axis=1, keepdims=True)
                sqdist = np.maximum(sq_norm + sq_norm.T - 2.0 * (pts @ pts.T), 0.0)
                gram = np.exp(-gamma_value * sqdist).astype(np.float32)
            elif kernel == "poly":
                gamma_value = float(gamma) if gamma is not None else 1.0 / max(int(D), 1)
                gram = (gamma_value * (pts @ pts.T) + 1.0) ** 3
            elif kernel == "linear":
                gram = pts @ pts.T
            else:
                raise ValueError(f"Unknown KPCA kernel: {kernel}")

            n = gram.shape[0]
            one = np.ones((n, n), dtype=np.float32) / float(n)
            gram = gram - one @ gram - gram @ one + one @ gram @ one
            eigvals = np.linalg.eigvalsh(gram).astype(np.float32)
            eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
            k = min(n_components, len(eigvals))
            features[t, b, :k] = eigvals[:k]
    return torch.from_numpy(features).float()


def _window_rbf_affinity(pts, gamma=None):
    n, d = pts.shape
    sq_norm = np.sum(pts * pts, axis=1, keepdims=True)
    sqdist = np.maximum(sq_norm + sq_norm.T - 2.0 * (pts @ pts.T), 0.0)
    if gamma is None:
        positive = sqdist[sqdist > 1e-12]
        sigma2 = float(np.median(positive)) if len(positive) else 1.0
        gamma_value = 1.0 / max(2.0 * sigma2, 1e-6)
    else:
        gamma_value = float(gamma)
    affinity = np.exp(-gamma_value * sqdist).astype(np.float32)
    np.fill_diagonal(affinity, 0.0)
    return affinity


def latent_window_laplacian_eigenvalue_features(z, window=6, n_components=16, gamma=None):
    """Smallest normalized graph-Laplacian eigenvalues of each latent trajectory window."""
    T, B, _ = z.shape
    n_components = int(n_components)
    features = np.zeros((T, B, n_components), dtype=np.float32)
    z_np = z.detach().cpu().numpy().astype(np.float32)

    for t in range(T):
        start = max(0, t - window + 1)
        for b in range(B):
            pts = z_np[start:t + 1, b, :]
            if len(pts) < 2:
                continue
            pts = pts - pts.mean(axis=0, keepdims=True)
            scale = pts.std(axis=0, keepdims=True).mean() + 1e-6
            pts = pts / scale
            affinity = _window_rbf_affinity(pts, gamma=gamma)
            degree = affinity.sum(axis=1)
            inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-6))
            normalized_adj = affinity * inv_sqrt_degree[:, None] * inv_sqrt_degree[None, :]
            laplacian = np.eye(len(pts), dtype=np.float32) - normalized_adj
            eigvals = np.linalg.eigvalsh(laplacian).astype(np.float32)
            eigvals = np.sort(np.maximum(eigvals, 0.0))
            k = min(n_components, len(eigvals))
            features[t, b, :k] = eigvals[:k]
    return torch.from_numpy(features).float()


def latent_window_diffusion_eigenvalue_features(z, window=6, n_components=16, gamma=None):
    """Leading diffusion-map eigenvalues of each latent trajectory window."""
    T, B, _ = z.shape
    n_components = int(n_components)
    features = np.zeros((T, B, n_components), dtype=np.float32)
    z_np = z.detach().cpu().numpy().astype(np.float32)

    for t in range(T):
        start = max(0, t - window + 1)
        for b in range(B):
            pts = z_np[start:t + 1, b, :]
            if len(pts) < 2:
                continue
            pts = pts - pts.mean(axis=0, keepdims=True)
            scale = pts.std(axis=0, keepdims=True).mean() + 1e-6
            pts = pts / scale
            affinity = _window_rbf_affinity(pts, gamma=gamma)
            degree = affinity.sum(axis=1)
            inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-6))
            symmetric_markov = affinity * inv_sqrt_degree[:, None] * inv_sqrt_degree[None, :]
            eigvals = np.linalg.eigvalsh(symmetric_markov).astype(np.float32)
            eigvals = np.sort(eigvals)[::-1]
            k = min(n_components, len(eigvals))
            features[t, b, :k] = eigvals[:k]
    return torch.from_numpy(features).float()


def latent_window_random_fourier_features(z, window=6, n_components=16, seed=0):
    """Fixed RBF random Fourier features of standardized latent windows."""
    T, B, D = z.shape
    window = int(window)
    n_components = int(n_components)
    features = np.zeros((T, B, n_components), dtype=np.float32)
    z_np = z.detach().cpu().numpy().astype(np.float32)
    rng = np.random.default_rng(int(seed))
    input_dim = window * D
    frequencies = rng.normal(
        loc=0.0,
        scale=np.sqrt(2.0 / max(input_dim, 1)),
        size=(input_dim, n_components),
    ).astype(np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_components).astype(np.float32)
    output_scale = np.sqrt(2.0 / max(n_components, 1))

    for t in range(T):
        start = max(0, t - window + 1)
        for b in range(B):
            pts = z_np[start:t + 1, b, :]
            pts = pts - pts.mean(axis=0, keepdims=True)
            scale = pts.std(axis=0, keepdims=True).mean() + 1e-6
            pts = pts / scale
            padded = np.zeros((window, D), dtype=np.float32)
            padded[-len(pts):] = pts
            features[t, b] = output_scale * np.cos(padded.reshape(-1) @ frequencies + phases)
    return torch.from_numpy(features).float()


def _control_feature_dim_for_homology(homology):
    if homology in {"h0", "h1"}:
        return int(LATENT_TDA_BINS)
    if homology == "both":
        return int(2 * LATENT_TDA_BINS)
    raise ValueError(f"Unknown control-feature homology-width selector: {homology}")


def _parse_latent_control_mode(base_mode):
    base_mode = canonicalize_latent_tda_mode(base_mode)
    if base_mode.startswith("z_fuse_"):
        spec = base_mode[len("z_fuse_"):]
    elif base_mode.startswith("topo_"):
        spec = base_mode[len("topo_"):]
    elif base_mode.startswith("z_"):
        spec = base_mode[len("z_"):]
    else:
        return None
    for homology in ("both", "h0", "h1"):
        if spec == f"pca_{homology}":
            return "pca", homology
        if spec == f"kpca_{homology}":
            return "kpca", homology
        if spec == f"laplacian_{homology}":
            return "laplacian", homology
        if spec == f"lap_{homology}":
            return "laplacian", homology
        if spec == f"diffusion_{homology}":
            return "diffusion", homology
        if spec == f"diffmap_{homology}":
            return "diffusion", homology
        if spec == f"rff_{homology}":
            return "rff", homology
        if spec == f"random_fourier_{homology}":
            return "rff", homology
    return None


def _pca_tensor(payload, homology):
    return latent_window_pca_eigenvalue_features(
        payload["z"],
        window=LATENT_TDA_WINDOW,
        n_components=_control_feature_dim_for_homology(homology),
    )


def _kpca_tensor(payload, homology):
    return latent_window_kpca_eigenvalue_features(
        payload["z"],
        window=LATENT_TDA_WINDOW,
        n_components=_control_feature_dim_for_homology(homology),
    )


def _laplacian_tensor(payload, homology):
    return latent_window_laplacian_eigenvalue_features(
        payload["z"],
        window=LATENT_TDA_WINDOW,
        n_components=_control_feature_dim_for_homology(homology),
    )


def _diffusion_tensor(payload, homology):
    return latent_window_diffusion_eigenvalue_features(
        payload["z"],
        window=LATENT_TDA_WINDOW,
        n_components=_control_feature_dim_for_homology(homology),
    )


def _random_fourier_tensor(payload, homology, seed=0):
    return latent_window_random_fourier_features(
        payload["z"],
        window=LATENT_TDA_WINDOW,
        n_components=_control_feature_dim_for_homology(homology),
        seed=seed,
    )


def _latent_control_tensor(payload, control_kind, homology, seed=0):
    if control_kind == "pca":
        return _pca_tensor(payload, homology)
    if control_kind == "kpca":
        return _kpca_tensor(payload, homology)
    if control_kind == "laplacian":
        return _laplacian_tensor(payload, homology)
    if control_kind == "diffusion":
        return _diffusion_tensor(payload, homology)
    if control_kind == "rff":
        return _random_fourier_tensor(payload, homology, seed=seed)
    raise ValueError(f"Unknown latent control kind: {control_kind}")


def _latent_control_tensor_pair(
    train_payload,
    test_payload,
    control_kind,
    homology,
    train_control_seed=0,
    test_control_seed=10_000,
    control="real",
):
    # A random-feature control must use one fixed map for both splits.
    feature_seed = int(train_control_seed)
    train_features = _latent_control_tensor(train_payload, control_kind, homology, seed=feature_seed)
    test_features = _latent_control_tensor(test_payload, control_kind, homology, seed=feature_seed)
    train_features = ml_tda.apply_tda_control(train_features, control=control, seed=int(train_control_seed), shift=1)
    test_features = ml_tda.apply_tda_control(test_features, control=control, seed=int(test_control_seed), shift=1)
    return train_features, test_features


def _split_control_suffix(mode):
    control_names = {"zero", "shuffle", "noise", "shift"}
    parts = mode.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in control_names:
        return parts[0], parts[1]
    return mode, "real"


def canonicalize_latent_tda_mode(mode):
    """Prefer compact mode names while accepting the older z_latent_* aliases."""
    if mode.startswith("z_temporal_stats_latent_"):
        return "z_temporal_stats_" + mode.removeprefix("z_temporal_stats_latent_")
    if mode.startswith("z_fuse_latent_"):
        return "z_fuse_" + mode.removeprefix("z_fuse_latent_")
    if mode.startswith("topo_latent_"):
        return "topo_" + mode.removeprefix("topo_latent_")
    if mode.startswith("z_latent_"):
        return "z_" + mode.removeprefix("z_latent_")
    return mode


def canonicalize_latent_tda_modes(modes):
    canonical_modes = []
    seen = set()
    for mode in modes:
        canonical_mode = canonicalize_latent_tda_mode(mode)
        if canonical_mode in seen:
            continue
        seen.add(canonical_mode)
        canonical_modes.append(canonical_mode)
    return canonical_modes


def _parse_vectorized_latent_mode(base_mode):
    base_mode = canonicalize_latent_tda_mode(base_mode)
    if base_mode.startswith("z_fuse_"):
        spec = base_mode[len("z_fuse_"):]
    elif base_mode.startswith("topo_"):
        spec = base_mode[len("topo_"):]
    elif base_mode.startswith("z_"):
        spec = base_mode[len("z_"):]
    else:
        return None
    for homology in ("both", "h0", "h1"):
        suffix = f"_{homology}"
        if spec.endswith(suffix):
            vectorizer = spec[:-len(suffix)]
            aliases = {
                "pi": "pi",
                "image": "pi",
                "images": "pi",
                "persistence_image": "pi",
                "persistence_images": "pi",
                "landscape": "landscape",
                "landscapes": "landscape",
                "perslay": "perslay",
            }
            if vectorizer in aliases:
                return aliases[vectorizer], homology
    return None


def _latent_mode_needs_persistence(mode):
    mode = canonicalize_latent_tda_mode(mode)
    if mode in {"z", "z_temporal_stats"}:
        return False
    if mode.startswith("z_temporal_stats_"):
        suffix_mode = "z_" + mode.removeprefix("z_temporal_stats_")
        return _latent_mode_needs_persistence(suffix_mode)
    base_mode, _ = _split_control_suffix(mode)
    if _parse_latent_control_mode(base_mode) is not None:
        return False
    if _parse_vectorized_latent_mode(base_mode) is not None:
        return True
    if base_mode.startswith("z_fuse_"):
        base_mode = "z_" + base_mode.removeprefix("z_fuse_")
    elif base_mode.startswith("z_gate_"):
        base_mode = "z_" + base_mode.removeprefix("z_gate_")
    elif base_mode.startswith("topo_"):
        base_mode = "z_" + base_mode.removeprefix("topo_")
    return base_mode in {"z_h0", "z_h1", "z_both"}


def _latent_modes_need_persistence(modes):
    return any(_latent_mode_needs_persistence(mode) for mode in modes)


def _require_diagrams(payload):
    if "diagrams" not in payload:
        raise ValueError(
            "This latent mode needs persistence diagrams, but the cache only has Betti curves. "
            "Rerun with --recompute-latent-tda-features."
        )


def _diagram_list(payload, homology):
    _require_diagrams(payload)
    return [_finite_diagram(diag) for row in payload["diagrams"][homology] for diag in row]


def _diagram_tensor_from_features(features, payload):
    T, B = payload["z"].shape[:2]
    return torch.as_tensor(features, dtype=torch.float32).reshape(T, B, -1)


def _fit_persistence_image(train_diagrams):
    if PersistenceImage is None:
        raise ImportError("Install gudhi to use z_latent_pi_* modes.")
    transformer = PersistenceImage(
        bandwidth=float(PERSISTENCE_IMAGE_BANDWIDTH),
        weight=lambda point: point[1],
        resolution=list(PERSISTENCE_IMAGE_RESOLUTION),
    )
    transformer.fit(train_diagrams)
    return transformer


def _fit_landscape(train_diagrams):
    if Landscape is None:
        raise ImportError("Install gudhi to use z_latent_landscape_* modes.")
    transformer = Landscape(
        num_landscapes=int(PERSISTENCE_LANDSCAPE_NUM),
        resolution=int(PERSISTENCE_LANDSCAPE_RESOLUTION),
    )
    transformer.fit(train_diagrams)
    return transformer


class PersLayFeaturizer:
    """Train-fitted, permutation-invariant diagram embedding inspired by PersLay."""

    def __init__(self, out_dim=64, sigma=0.25):
        self.out_dim = int(out_dim)
        self.sigma = float(sigma)
        self.centers = None

    def fit(self, diagrams):
        points = []
        for diag in diagrams:
            diag = _finite_diagram(diag)
            if len(diag):
                bp = diag.copy()
                bp[:, 1] = bp[:, 1] - bp[:, 0]
                points.append(bp)
        if points:
            points = np.concatenate(points, axis=0)
            idx = np.linspace(0, len(points) - 1, self.out_dim, dtype=int)
            order = np.argsort(points[:, 1], kind="mergesort")
            self.centers = points[order][idx].astype(np.float32)
        else:
            self.centers = np.zeros((self.out_dim, 2), dtype=np.float32)
        return self

    def transform(self, diagrams):
        if self.centers is None:
            raise RuntimeError("PersLayFeaturizer must be fit before transform.")
        outputs = np.zeros((len(diagrams), self.out_dim), dtype=np.float32)
        centers = self.centers[None, :, :]
        sigma2 = max(self.sigma ** 2, 1e-8)
        for i, diag in enumerate(diagrams):
            diag = _finite_diagram(diag)
            if len(diag) == 0:
                continue
            bp = diag.astype(np.float32, copy=True)
            bp[:, 1] = bp[:, 1] - bp[:, 0]
            weights = bp[:, 1:2]
            dist2 = np.sum((bp[:, None, :] - centers) ** 2, axis=-1)
            outputs[i] = np.sum(weights * np.exp(-0.5 * dist2 / sigma2), axis=0)
        return outputs


def _vectorized_diagram_tensor_pair(train_payload, test_payload, vectorizer, homology):
    train_parts, test_parts = [], []
    homologies = ("h0", "h1") if homology == "both" else (homology,)
    for h in homologies:
        train_diagrams = _diagram_list(train_payload, h)
        test_diagrams = _diagram_list(test_payload, h)
        if vectorizer == "pi":
            transformer = _fit_persistence_image(train_diagrams)
        elif vectorizer == "landscape":
            transformer = _fit_landscape(train_diagrams)
        elif vectorizer == "perslay":
            transformer = PersLayFeaturizer(out_dim=PERSLAY_OUT_DIM, sigma=PERSLAY_SIGMA).fit(train_diagrams)
        else:
            raise ValueError(f"Unknown diagram vectorizer: {vectorizer}")
        train_parts.append(_diagram_tensor_from_features(transformer.transform(train_diagrams), train_payload))
        test_parts.append(_diagram_tensor_from_features(transformer.transform(test_diagrams), test_payload))
    return torch.cat(train_parts, dim=-1), torch.cat(test_parts, dim=-1)


def _betti_tda_tensor(payload, homology):
    if homology == "h0":
        return payload["h0"]
    if homology == "h1":
        return payload["h1"]
    if homology == "both":
        return torch.cat([payload["h0"], payload["h1"]], dim=-1)
    raise ValueError(f"Unknown homology selector: {homology}")


def _topology_tensor_pair(train_payload, test_payload, topo_spec, train_control_seed=0, test_control_seed=10_000):
    topo_spec, control = _split_control_suffix(canonicalize_latent_tda_mode(topo_spec))
    if topo_spec.startswith("z_fuse_"):
        topo_spec = "z_" + topo_spec.removeprefix("z_fuse_")
    elif topo_spec.startswith("z_gate_"):
        topo_spec = "z_" + topo_spec.removeprefix("z_gate_")
    elif topo_spec.startswith("topo_"):
        topo_spec = "z_" + topo_spec.removeprefix("topo_")

    parsed_control = _parse_latent_control_mode(topo_spec)
    if parsed_control is not None:
        control_kind, homology = parsed_control
        train_tda, test_tda = _latent_control_tensor_pair(
            train_payload,
            test_payload,
            control_kind,
            homology,
            train_control_seed=train_control_seed,
            test_control_seed=test_control_seed,
            control=control,
        )
        return train_tda, test_tda

    parsed = _parse_vectorized_latent_mode(topo_spec)
    if parsed is not None:
        vectorizer, homology = parsed
        train_tda, test_tda = _vectorized_diagram_tensor_pair(train_payload, test_payload, vectorizer, homology)
    elif topo_spec in {"z_h0", "z_h1", "z_both"}:
        homology = topo_spec.removeprefix("z_")
        train_tda = _betti_tda_tensor(train_payload, homology)
        test_tda = _betti_tda_tensor(test_payload, homology)
    else:
        raise ValueError(f"Unknown topology mode: {topo_spec}")

    train_tda = ml_tda.apply_tda_control(train_tda, control=control, seed=int(train_control_seed), shift=1)
    test_tda = ml_tda.apply_tda_control(test_tda, control=control, seed=int(test_control_seed), shift=1)
    return train_tda, test_tda


def features_for_latent_tda_mode(payload, mode):
    mode = canonicalize_latent_tda_mode(mode)
    z = payload["z"]
    if mode == "z":
        return z
    if mode == "z_temporal_stats":
        temporal_stats = latent_window_temporal_stats(z, window=LATENT_TDA_WINDOW)
        return torch.cat([z, temporal_stats], dim=-1)
    if mode.startswith("z_temporal_stats_"):
        temporal_stats = latent_window_temporal_stats(z, window=LATENT_TDA_WINDOW)
        latent_mode = "z_" + mode.removeprefix("z_temporal_stats_")
        latent_features = features_for_latent_tda_mode(payload, latent_mode)
        latent_tda = latent_features[..., z.shape[-1]:]
        return torch.cat([z, temporal_stats, latent_tda], dim=-1)

    base_mode, control = _split_control_suffix(mode)
    parsed_control = _parse_latent_control_mode(base_mode)
    if parsed_control is not None:
        control_kind, homology = parsed_control
        control_seed = globals().get("CONTROL_SEED", 0)
        tda = _latent_control_tensor(payload, control_kind, homology, seed=int(control_seed))
        tda = ml_tda.apply_tda_control(tda, control=control, seed=int(control_seed), shift=1)
        return torch.cat([z, tda], dim=-1)

    if base_mode == "z_h0":
        tda = payload["h0"]
    elif base_mode == "z_h1":
        tda = payload["h1"]
    elif base_mode == "z_both":
        tda = torch.cat([payload["h0"], payload["h1"]], dim=-1)
    else:
        raise ValueError(f"Unknown latent TDA mode: {mode}")

    control_seed = globals().get("CONTROL_SEED", 0)
    tda = ml_tda.apply_tda_control(tda, control=control, seed=int(control_seed), shift=1)
    return torch.cat([z, tda], dim=-1)


def _features_for_latent_tda_mode_pair_impl(train_payload, test_payload, mode, train_control_seed=0, test_control_seed=10_000):
    mode = canonicalize_latent_tda_mode(mode)
    if mode == "z":
        return train_payload["z"], test_payload["z"]

    if mode == "z_temporal_stats":
        train_z = train_payload["z"]
        test_z = test_payload["z"]
        return (
            torch.cat([train_z, latent_window_temporal_stats(train_z, window=LATENT_TDA_WINDOW)], dim=-1),
            torch.cat([test_z, latent_window_temporal_stats(test_z, window=LATENT_TDA_WINDOW)], dim=-1),
        )

    if mode.startswith("z_temporal_stats_"):
        latent_mode = "z_" + mode.removeprefix("z_temporal_stats_")
        train_latent, test_latent = _features_for_latent_tda_mode_pair_impl(
            train_payload,
            test_payload,
            latent_mode,
            train_control_seed=train_control_seed,
            test_control_seed=test_control_seed,
        )
        train_z = train_payload["z"]
        test_z = test_payload["z"]
        train_tda = train_latent[..., train_z.shape[-1]:]
        test_tda = test_latent[..., test_z.shape[-1]:]
        return (
            torch.cat([train_z, latent_window_temporal_stats(train_z, window=LATENT_TDA_WINDOW), train_tda], dim=-1),
            torch.cat([test_z, latent_window_temporal_stats(test_z, window=LATENT_TDA_WINDOW), test_tda], dim=-1),
        )

    if mode.startswith("topo_"):
        return _topology_tensor_pair(
            train_payload,
            test_payload,
            mode,
            train_control_seed=train_control_seed,
            test_control_seed=test_control_seed,
        )

    if mode.startswith("z_fuse_"):
        train_tda, test_tda = _topology_tensor_pair(
            train_payload,
            test_payload,
            mode,
            train_control_seed=train_control_seed,
            test_control_seed=test_control_seed,
        )
        return (
            torch.cat([train_payload["z"], train_tda], dim=-1),
            torch.cat([test_payload["z"], test_tda], dim=-1),
        )

    if mode.startswith("z_gate_"):
        if mode != "z_gate_both":
            raise ValueError(
                "Group-wise topology gating requires both homology groups; "
                "use mode='z_gate_both'."
            )
        train_tda, test_tda = _topology_tensor_pair(
            train_payload,
            test_payload,
            mode,
            train_control_seed=train_control_seed,
            test_control_seed=test_control_seed,
        )
        return (
            torch.cat([train_payload["z"], train_tda], dim=-1),
            torch.cat([test_payload["z"], test_tda], dim=-1),
        )

    base_mode, control = _split_control_suffix(mode)
    parsed_control = _parse_latent_control_mode(base_mode)
    if parsed_control is not None:
        control_kind, homology = parsed_control
        train_control_features, test_control_features = _latent_control_tensor_pair(
            train_payload,
            test_payload,
            control_kind,
            homology,
            train_control_seed=train_control_seed,
            test_control_seed=test_control_seed,
            control=control,
        )
        return (
            torch.cat([train_payload["z"], train_control_features], dim=-1),
            torch.cat([test_payload["z"], test_control_features], dim=-1),
        )

    parsed = _parse_vectorized_latent_mode(base_mode)
    if parsed is None:
        configure_runtime(CONTROL_SEED=train_control_seed)
        train_features = features_for_latent_tda_mode(train_payload, mode)
        configure_runtime(CONTROL_SEED=test_control_seed)
        test_features = features_for_latent_tda_mode(test_payload, mode)
        return train_features, test_features

    vectorizer, homology = parsed
    train_tda, test_tda = _vectorized_diagram_tensor_pair(train_payload, test_payload, vectorizer, homology)
    train_tda = ml_tda.apply_tda_control(train_tda, control=control, seed=int(train_control_seed), shift=1)
    test_tda = ml_tda.apply_tda_control(test_tda, control=control, seed=int(test_control_seed), shift=1)
    return (
        torch.cat([train_payload["z"], train_tda], dim=-1),
        torch.cat([test_payload["z"], test_tda], dim=-1),
    )


def features_for_latent_tda_mode_pair(train_payload, test_payload, mode, train_control_seed=0, test_control_seed=10_000):
    with _profile_phase("fusion"):
        return _features_for_latent_tda_mode_pair_impl(
            train_payload,
            test_payload,
            mode,
            train_control_seed=train_control_seed,
            test_control_seed=test_control_seed,
        )


def _fit_standardizer(x, eps=1e-6):
    """Fit one scalar standardizer to remove arbitrary encoder-level scale."""
    mean = x.mean().reshape(1, 1, 1)
    std = x.std().clamp_min(eps).reshape(1, 1, 1)
    return mean, std


def _fit_z_h0_h1_standardizer(x, latent_dim, eps=1e-6):
    """Fit independent scalar scales for the z, H0, and H1 blocks."""
    topo_dim = int(x.shape[-1]) - int(latent_dim)
    if topo_dim <= 0 or topo_dim % 2:
        raise ValueError(
            "Separate z/H0/H1 normalization requires equal-width H0 and H1 "
            f"blocks; got input width {x.shape[-1]} and latent width {latent_dim}"
        )
    widths = [int(latent_dim), topo_dim // 2, topo_dim // 2]
    means, stds = [], []
    start = 0
    for width in widths:
        block = x[..., start:start + width]
        mean, std = _fit_standardizer(block, eps=eps)
        means.append(mean.expand(1, 1, width))
        stds.append(std.expand(1, 1, width))
        start += width
    return torch.cat(means, dim=-1), torch.cat(stds, dim=-1)


def _uses_z_h0_h1_standardizer(mode):
    return mode in {"z_both", "z_fuse_both", "z_gate_both"}


def _standardize(x, mean, std):
    return (x - mean) / std


def _attach_latent_standardizers(model, feature_mean, feature_std, target_mean, target_std):
    model.feature_mean = feature_mean.detach().cpu()
    model.feature_std = feature_std.detach().cpu()
    model.target_mean = target_mean.detach().cpu()
    model.target_std = target_std.detach().cpu()
    return model


class TopologicalFusion(nn.Module):
    """Fuse latent visual state and topology features into one learned representation."""

    def __init__(self, latent_dim, topo_dim, fused_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + topo_dim, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(self, z, topo):
        return self.net(torch.cat([z, topo], dim=-1))


class FusedTopologicalPredictor(nn.Module):
    """Jointly train a topology fusion MLP and the temporal predictor."""

    def __init__(self, latent_dim, topo_dim, fused_dim=128, hidden_dim=128):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.topo_dim = int(topo_dim)
        self.fusion = TopologicalFusion(latent_dim=latent_dim, topo_dim=topo_dim, fused_dim=fused_dim)
        self.predictor = make_predictor(input_dim=fused_dim, hidden_dim=hidden_dim)

    def forward(self, x):
        z = x[..., :self.latent_dim]
        topo = x[..., self.latent_dim:]
        h = self.fusion(z, topo)
        return self.predictor(h)


class GroupGatedTopologicalPredictor(nn.Module):
    """Apply one geometry-conditioned gate to H0 and one to H1."""

    def __init__(self, latent_dim, h0_dim, h1_dim, gate_hidden_dim=64, hidden_dim=128):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.h0_dim = int(h0_dim)
        self.h1_dim = int(h1_dim)
        if min(self.latent_dim, self.h0_dim, self.h1_dim) <= 0:
            raise ValueError("latent_dim, h0_dim, and h1_dim must all be positive")
        self.gate = nn.Sequential(
            nn.Linear(self.latent_dim + self.h0_dim + self.h1_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 2),
            nn.Sigmoid(),
        )
        # Begin as an almost-open gate so training starts close to the direct
        # [z, H0, H1] baseline and learns only the suppression it needs.
        initial_gate = 0.9
        initial_logit = float(np.log(initial_gate / (1.0 - initial_gate)))
        nn.init.zeros_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, initial_logit)
        self.predictor = make_predictor(
            input_dim=self.latent_dim + self.h0_dim + self.h1_dim,
            hidden_dim=hidden_dim,
        )
        self.last_gate_weights = None

    def forward(self, x):
        expected_dim = self.latent_dim + self.h0_dim + self.h1_dim
        if x.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected {expected_dim} input features, got {x.shape[-1]}"
            )
        z, h0, h1 = torch.split(
            x,
            [self.latent_dim, self.h0_dim, self.h1_dim],
            dim=-1,
        )
        gate_weights = self.gate(torch.cat([z, h0, h1], dim=-1))
        gated_h0 = gate_weights[..., 0:1] * h0
        gated_h1 = gate_weights[..., 1:2] * h1
        if not self.training:
            self.last_gate_weights = gate_weights.detach().cpu()
        return self.predictor(torch.cat([z, gated_h0, gated_h1], dim=-1))


def topology_gate_statistics(model):
    """Return held-out H0/H1 gate summaries for the results table."""
    if not isinstance(model, GroupGatedTopologicalPredictor):
        return {}
    weights = model.last_gate_weights
    if weights is None or weights.numel() == 0:
        return {
            "gate_h0_mean": np.nan,
            "gate_h0_std": np.nan,
            "gate_h1_mean": np.nan,
            "gate_h1_std": np.nan,
        }
    flat = weights.reshape(-1, 2).float()
    return {
        "gate_h0_mean": float(flat[:, 0].mean().item()),
        "gate_h0_std": float(flat[:, 0].std(unbiased=False).item()),
        "gate_h1_mean": float(flat[:, 1].mean().item()),
        "gate_h1_std": float(flat[:, 1].std(unbiased=False).item()),
    }


def latent_result_metric_columns(results_df):
    """Core metrics for the shared mode table.

    Gate diagnostics are intentionally summarized separately because they are
    not applicable to the non-gated ablations.
    """
    return ["test_mse", "latent_r2"]


def _stable_int_seed(*parts):
    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _set_predictor_seed(seed, mode):
    stable_seed = _stable_int_seed(DATASET, seed, mode, HORIZON, LATENT_TDA_WINDOW, LATENT_TDA_BINS)
    random.seed(stable_seed)
    np.random.seed(stable_seed % (2**32))
    torch.manual_seed(stable_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(stable_seed)


def train_or_load_latent_tda_predictor(seed, mode, train_features, train_z):
    with _profile_phase("predictor"):
        if HORIZON >= train_features.shape[0]:
            raise ValueError(
                f"HORIZON={HORIZON} must be smaller than sequence length "
                f"{train_features.shape[0]}"
            )
        _set_predictor_seed(seed, mode)
        model_dir = Path("models") / DATASET / "latent_tda_predictors"
        model_dir.mkdir(parents=True, exist_ok=True)
        if not STANDARDIZE_LATENT_PREDICTOR:
            standardizer_tag = "raw"
        elif _uses_z_h0_h1_standardizer(mode):
            standardizer_tag = "zh0h1std"
        else:
            standardizer_tag = "gstdz"
        if mode.startswith("z_fuse_"):
            architecture_tag = "fusion"
        elif mode.startswith("z_gate_"):
            architecture_tag = "groupgate"
        else:
            architecture_tag = "direct"
        lr_tag = f"{LATENT_TDA_LR:g}".replace(".", "p").replace("-", "m")
        model_path = model_dir / (
            f"model_seed{seed}_pred{HORIZON}_{mode}_{architecture_tag}_{standardizer_tag}_{ml_tda.PREDICTOR_TYPE}_"
            f"hidden{HIDDEN_DIM}_epochs{LATENT_TDA_PREDICTOR_EPOCHS}_lr{lr_tag}_"
            f"win{LATENT_TDA_WINDOW}_bins{LATENT_TDA_BINS}.pt"
        )
        device = ml_tda.get_runtime_device()
        if mode.startswith("z_fuse_"):
            topo_dim = train_features.shape[-1] - LATENT_DIM
            if topo_dim <= 0:
                raise ValueError(f"Fusion mode {mode} needs topology features after z; got input_dim={train_features.shape[-1]}")
            model = FusedTopologicalPredictor(
                latent_dim=LATENT_DIM,
                topo_dim=topo_dim,
                fused_dim=HIDDEN_DIM,
                hidden_dim=HIDDEN_DIM,
            ).to(device)
        elif mode.startswith("z_gate_"):
            if mode != "z_gate_both":
                raise ValueError(
                    "Group-wise topology gating requires mode='z_gate_both'."
                )
            topo_dim = train_features.shape[-1] - LATENT_DIM
            if topo_dim <= 0 or topo_dim % 2:
                raise ValueError(
                    "z_gate_both expects equal-width H0 and H1 groups; "
                    f"got combined topology width {topo_dim}"
                )
            model = GroupGatedTopologicalPredictor(
                latent_dim=LATENT_DIM,
                h0_dim=topo_dim // 2,
                h1_dim=topo_dim // 2,
                gate_hidden_dim=min(64, HIDDEN_DIM),
                hidden_dim=HIDDEN_DIM,
            ).to(device)
        else:
            model = make_predictor(input_dim=train_features.shape[-1], hidden_dim=HIDDEN_DIM).to(device)
        source_features = train_features[:-HORIZON]
        source_targets = train_z[HORIZON:]
        if STANDARDIZE_LATENT_PREDICTOR:
            if _uses_z_h0_h1_standardizer(mode):
                feature_mean, feature_std = _fit_z_h0_h1_standardizer(
                    source_features,
                    LATENT_DIM,
                )
            else:
                feature_mean, feature_std = _fit_standardizer(source_features)
            target_mean, target_std = _fit_standardizer(source_targets)
        else:
            feature_mean = torch.zeros((1, 1, source_features.shape[-1]), dtype=source_features.dtype)
            feature_std = torch.ones((1, 1, source_features.shape[-1]), dtype=source_features.dtype)
            target_mean = torch.zeros((1, 1, source_targets.shape[-1]), dtype=source_targets.dtype)
            target_std = torch.ones((1, 1, source_targets.shape[-1]), dtype=source_targets.dtype)
        _attach_latent_standardizers(model, feature_mean, feature_std, target_mean, target_std)

        if model_path.exists() and not RETRAIN_LATENT_TDA_PREDICTOR:
            print(f"Loading latent TDA predictor: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                _attach_latent_standardizers(
                    model,
                    checkpoint["feature_mean"],
                    checkpoint["feature_std"],
                    checkpoint["target_mean"],
                    checkpoint["target_std"],
                )
            else:
                model.load_state_dict(checkpoint)
            return model

        reason = "retraining" if model_path.exists() else "missing; training once"
        print(f"Latent TDA predictor {reason}: {model_path}")
        optimizer = optim.AdamW(model.parameters(), lr=LATENT_TDA_LR)
        criterion = nn.MSELoss()
        train_x = _standardize(source_features, feature_mean, feature_std).to(device)
        train_y = _standardize(source_targets, target_mean, target_std).to(device)
        epochs = range(1, LATENT_TDA_PREDICTOR_EPOCHS + 1)
        for epoch in tqdm_progress_bar(epochs, desc=f"Latent predictor {mode}", total=LATENT_TDA_PREDICTOR_EPOCHS):
            model.train()
            optimizer.zero_grad()
            pred = model(train_x)
            loss = criterion(pred, train_y)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0 or epoch == LATENT_TDA_PREDICTOR_EPOCHS:
                print(
                    f"Epoch {epoch:02d}/{LATENT_TDA_PREDICTOR_EPOCHS:02d} | "
                    f"{mode} standardized train MSE: {loss.item():.6f}"
                )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_mean": model.feature_mean,
                "feature_std": model.feature_std,
                "target_mean": model.target_mean,
                "target_std": model.target_std,
                "standardized": bool(STANDARDIZE_LATENT_PREDICTOR),
            },
            model_path,
        )
        return model


def eval_latent_tda_predictor(model, test_features, test_z):
    with _profile_phase("predictor"):
        if HORIZON >= test_features.shape[0]:
            raise ValueError(
                f"HORIZON={HORIZON} must be smaller than sequence length "
                f"{test_features.shape[0]}"
            )
        model.eval()
        criterion = nn.MSELoss()
        device = ml_tda.get_runtime_device()
        model.to(device)
        with torch.no_grad():
            target = test_z[HORIZON:].to(device)
            feature_mean = model.feature_mean.to(device)
            feature_std = model.feature_std.to(device)
            target_mean = model.target_mean.to(device)
            target_std = model.target_std.to(device)
            features = _standardize(test_features[:-HORIZON].to(device), feature_mean, feature_std)
            target_standardized = _standardize(target, target_mean, target_std)
            pred_standardized = model(features)
            mse = criterion(pred_standardized, target_standardized).item()
            per_frame = ((pred_standardized - target_standardized) ** 2).mean(dim=(1, 2)).cpu()
    target_cpu = target_standardized.detach().float().cpu()
    target_var = torch.mean((target_cpu - target_cpu.mean()) ** 2).item()
    r2 = float(1.0 - (mse / max(target_var, 1e-12)))
    return mse, per_frame, r2


def _normalized_distance_matrix(points):
    distances = squareform(pdist(points))
    scale = float(distances.max())
    if scale > 1e-12:
        distances = distances / scale
    return distances.astype(np.float32)


def _sample_betti_curves(points, n_bins=16):
    if points.shape[0] < 4:
        zero = np.zeros(n_bins, dtype=np.float32)
        return zero, zero
    distances = _normalized_distance_matrix(points)
    diagrams = ripser(distances, maxdim=1, distance_matrix=True)["dgms"]
    h0 = betti_curve_from_diag(diagrams[0], n_bins=n_bins)
    h1 = betti_curve_from_diag(diagrams[1], n_bins=n_bins) if len(diagrams) > 1 else np.zeros(n_bins, dtype=np.float32)
    return h0, h1


def latent_geometry_diagnostics(video_tensor, z, max_points=512, max_tda_points=128):
    """Secondary diagnostics for explaining latent geometry quality."""
    with torch.no_grad():
        z_cpu = z.detach().cpu().float()
        dz = z_cpu[1:] - z_cpu[:-1]
        velocity = float(torch.linalg.vector_norm(dz, dim=-1).mean().item()) if dz.numel() else 0.0
        if z_cpu.shape[0] > 2:
            ddz = z_cpu[2:] - 2 * z_cpu[1:-1] + z_cpu[:-2]
            acceleration = float(torch.linalg.vector_norm(ddz, dim=-1).mean().item())
        else:
            acceleration = 0.0
        flat_z = z_cpu.reshape(-1, z_cpu.shape[-1]).numpy()
        frames = ml_tda.tensor_to_model_float(video_tensor).reshape(-1, *video_tensor.shape[2:]).float()
        flat_x = frames.reshape(frames.shape[0], -1).cpu().numpy()

    common_n = min(flat_z.shape[0], flat_x.shape[0])
    finite_rows = np.isfinite(flat_x[:common_n]).all(axis=1) & np.isfinite(flat_z[:common_n]).all(axis=1)
    valid_idx = np.flatnonzero(finite_rows)
    n = min(int(max_points), len(valid_idx))
    if n < 4:
        return {
            "latent_velocity": velocity,
            "latent_acceleration": acceleration,
            "distance_corr": np.nan,
            "trustworthiness": np.nan,
            "topology_h0_l2": np.nan,
            "topology_h1_l2": np.nan,
        }

    # Preserve temporal coverage while excluding invalid source rows. This is
    # a diagnostic only; it must not abort an otherwise completed experiment.
    idx = valid_idx[np.linspace(0, len(valid_idx) - 1, n, dtype=int)]
    x_sample = flat_x[idx]
    z_sample = flat_z[idx]
    dx = pdist(x_sample)
    dz_pair = pdist(z_sample)
    distance_corr = np.nan
    if np.std(dx) >= 1e-12 and np.std(dz_pair) >= 1e-12:
        distance_corr = float(np.corrcoef(dx, dz_pair)[0, 1])
    if trustworthiness is None:
        trust = np.nan
    else:
        # sklearn requires n_neighbors < n_samples / 2. Small biological clips can
        # hit this boundary exactly, so choose the largest valid diagnostic value.
        n_neighbors = min(10, max(1, (n - 1) // 2))
        trust = float(trustworthiness(x_sample, z_sample, n_neighbors=n_neighbors))
    n_tda = min(int(max_tda_points), n)
    tda_idx = np.linspace(0, n - 1, n_tda, dtype=int)
    x_h0, x_h1 = _sample_betti_curves(x_sample[tda_idx], n_bins=LATENT_TDA_BINS)
    z_h0, z_h1 = _sample_betti_curves(z_sample[tda_idx], n_bins=LATENT_TDA_BINS)
    return {
        "latent_velocity": velocity,
        "latent_acceleration": acceleration,
        "distance_corr": distance_corr,
        "trustworthiness": trust,
        "topology_h0_l2": float(np.linalg.norm(x_h0 - z_h0)),
        "topology_h1_l2": float(np.linalg.norm(x_h1 - z_h1)),
    }


def run_latent_tda_trajectory_experiment(
    X_train=None,
    X_test=None,
    run_seeds=None,
    modes=None,
    max_train=None,
    max_test=None,
    display_fn=None,
    compute_diagnostics=True,
):
    if X_train is None or X_test is None:
        print(
            "Skipping standalone latent-TDA cell: X_train/X_test are not defined. "
            "Provide X_train/X_test, or use the celltracking runner's built-in latent-TDA block."
        )
        empty = pd.DataFrame()
        return empty, empty, empty

    run_seeds = list(range(5)) if run_seeds is None else run_seeds
    modes = ["z", "z_h0", "z_h1", "z_both"] if modes is None else canonicalize_latent_tda_modes(modes)
    require_persistence = _latent_modes_need_persistence(modes)
    rows = []
    for seed in tqdm_progress_bar(run_seeds, desc="Latent TDA seeds", total=len(run_seeds), leave=True):
        print(f"\n================ latent TDA seed={seed} ================")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        Xtr = take_batch_subset(X_train, max_train, seed=seed)
        Xte = take_batch_subset(X_test, max_test, seed=seed + 1)
        encoder = load_or_train_shared_encoder_for_latent_tda(seed, Xtr, X_train)
        train_payload = load_or_compute_latent_tda_features(
            seed,
            "train",
            Xtr,
            encoder,
            require_persistence=require_persistence,
        )
        test_payload = load_or_compute_latent_tda_features(
            seed,
            "test",
            Xte,
            encoder,
            require_persistence=require_persistence,
        )
        diagnostics = latent_geometry_diagnostics(Xte, test_payload["z"]) if compute_diagnostics else {}

        for mode in modes:
            train_features, test_features = features_for_latent_tda_mode_pair(
                train_payload,
                test_payload,
                mode,
                train_control_seed=seed,
                test_control_seed=seed + 10_000,
            )
            model = train_or_load_latent_tda_predictor(seed, mode, train_features, train_payload["z"])
            test_mse, per_frame_mse, latent_r2 = eval_latent_tda_predictor(
                model,
                test_features,
                test_payload["z"],
            )
            row = {
                "dataset": DATASET,
                "seed": seed,
                "mode": mode,
                "horizon": HORIZON,
                "test_mse": float(test_mse),
                "latent_r2": float(latent_r2),
                **topology_gate_statistics(model),
                **diagnostics,
            }
            rows.append(row)
            print("latent TDA run summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=latent_result_metric_columns(results_df),
        sort_metric="test_mse",
    )
    duplicate_mask = results_df.duplicated(subset=["seed", "mode"], keep=False)
    if duplicate_mask.any():
        duplicate_counts = (
            results_df.loc[duplicate_mask]
            .groupby(["seed", "mode"])
            .size()
            .rename("n")
            .reset_index()
        )
        print("\nDuplicate latent TDA rows found; averaging duplicates for paired diagnostics:")
        print(duplicate_counts)
    paired_df = results_df.pivot_table(index="seed", columns="mode", values="test_mse", aggfunc="mean")
    if {"z", "z_h1"}.issubset(paired_df.columns):
        paired_df["h1_minus_z"] = paired_df["z_h1"] - paired_df["z"]
        print("\nPaired z_h1 - z differences; negative means TDA helped:")
        print(paired_df)

    print("\nLatent TDA trajectory summary:")
    print(summary_df)
    if display_fn is not None:
        display_fn(summary_df)
        display_fn(results_df.sort_values(["seed", "mode"]))
    return results_df, summary_df, paired_df
