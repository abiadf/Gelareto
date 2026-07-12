"""Latent-trajectory TDA helpers for video forecasting experiments."""

from pathlib import Path
import hashlib
import random

import numpy as np
import pandas as pd
from ripser import ripser
import torch
import torch.nn as nn
import torch.optim as optim

import topo.ml_tda as ml_tda
from topo.ml_tda import (
    SpatialEncoder,
    SpatialDecoder,
    TopologicalPredictor,
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
        pretrain_spatial_encoder(encoder, decoder, X_train_subset, ae_epochs=3)
        torch.save(encoder.state_dict(), encoder_path)

    encoder.to(ml_tda.get_runtime_device())
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


@torch.no_grad()
def encode_video_to_z(X, encoder, frame_batch_size=1024):
    device = ml_tda.get_runtime_device()
    encoder.to(device)
    T, B = X.shape[:2]
    frames = X.reshape(T * B, *X.shape[2:])
    chunks = []
    for start in range(0, len(frames), frame_batch_size):
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


def latent_window_betti_features(z_features, window=6, n_bins=16):
    T, B, _ = z_features.shape
    h0 = np.zeros((T, B, n_bins), dtype=np.float32)
    h1 = np.zeros((T, B, n_bins), dtype=np.float32)
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
            h0[t, b] = betti_curve_from_diag(dgms[0], n_bins=n_bins)
            h1[t, b] = betti_curve_from_diag(dgms[1], n_bins=n_bins) if len(dgms) > 1 else 0.0
        if t in {0, T - 1}:
            print(f"  latent TDA frame {t + 1}/{T}")

    return torch.from_numpy(h0), torch.from_numpy(h1)


def latent_tda_cache_path(seed, split_name, X_subset):
    cache_dir = Path("models") / DATASET / "latent_tda_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_{split_name}_T{X_subset.shape[0]}_B{X_subset.shape[1]}_"
        f"H{X_subset.shape[-2]}_W{X_subset.shape[-1]}_latent{LATENT_DIM}_"
        f"win{LATENT_TDA_WINDOW}_bins{LATENT_TDA_BINS}"
    )
    return cache_dir / f"{tag}.pt"


def load_or_compute_latent_tda_features(seed, split_name, X_subset, encoder):
    cache_path = latent_tda_cache_path(seed, split_name, X_subset)
    if cache_path.exists() and not RECOMPUTE_LATENT_TDA_FEATURES:
        print(f"Loading latent TDA cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f"Computing z + latent-window TDA for {split_name}...")
    z = encode_video_to_z(X_subset, encoder)
    h0, h1 = latent_window_betti_features(z, window=LATENT_TDA_WINDOW, n_bins=LATENT_TDA_BINS)
    payload = {"z": z, "h0": h0, "h1": h1}
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


def features_for_latent_tda_mode(payload, mode):
    z = payload["z"]
    if mode == "z":
        return z
    if mode == "z_temporal_stats":
        temporal_stats = latent_window_temporal_stats(z, window=LATENT_TDA_WINDOW)
        return torch.cat([z, temporal_stats], dim=-1)
    if mode.startswith("z_temporal_stats_latent_"):
        temporal_stats = latent_window_temporal_stats(z, window=LATENT_TDA_WINDOW)
        latent_mode = "z_latent_" + mode.removeprefix("z_temporal_stats_latent_")
        latent_features = features_for_latent_tda_mode(payload, latent_mode)
        latent_tda = latent_features[..., z.shape[-1]:]
        return torch.cat([z, temporal_stats, latent_tda], dim=-1)

    control_names = {"zero", "shuffle", "noise", "shift"}
    parts = mode.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in control_names:
        base_mode, control = parts
    else:
        base_mode, control = mode, "real"

    if base_mode == "z_latent_h0":
        tda = payload["h0"]
    elif base_mode == "z_latent_h1":
        tda = payload["h1"]
    elif base_mode == "z_latent_both":
        tda = torch.cat([payload["h0"], payload["h1"]], dim=-1)
    else:
        raise ValueError(f"Unknown latent TDA mode: {mode}")

    control_seed = globals().get("CONTROL_SEED", 0)
    tda = ml_tda.apply_tda_control(tda, control=control, seed=int(control_seed), shift=1)
    return torch.cat([z, tda], dim=-1)


def _fit_standardizer(x, eps=1e-6):
    """Fit one scalar standardizer to remove arbitrary encoder-level scale."""
    mean = x.mean().reshape(1, 1, 1)
    std = x.std().clamp_min(eps).reshape(1, 1, 1)
    return mean, std


def _standardize(x, mean, std):
    return (x - mean) / std


def _attach_latent_standardizers(model, feature_mean, feature_std, target_mean, target_std):
    model.feature_mean = feature_mean.detach().cpu()
    model.feature_std = feature_std.detach().cpu()
    model.target_mean = target_mean.detach().cpu()
    model.target_std = target_std.detach().cpu()
    return model


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
    if HORIZON >= train_features.shape[0]:
        raise ValueError(
            f"HORIZON={HORIZON} must be smaller than sequence length "
            f"{train_features.shape[0]}"
        )
    _set_predictor_seed(seed, mode)
    model_dir = Path("models") / DATASET / "latent_tda_predictors"
    model_dir.mkdir(parents=True, exist_ok=True)
    standardizer_tag = "gstdz" if STANDARDIZE_LATENT_PREDICTOR else "raw"
    model_path = model_dir / (
        f"model_seed{seed}_pred{HORIZON}_{mode}_{standardizer_tag}_"
        f"win{LATENT_TDA_WINDOW}_bins{LATENT_TDA_BINS}.pt"
    )
    device = ml_tda.get_runtime_device()
    model = TopologicalPredictor(input_dim=train_features.shape[-1], hidden_dim=HIDDEN_DIM).to(device)
    source_features = train_features[:-HORIZON]
    source_targets = train_z[HORIZON:]
    if STANDARDIZE_LATENT_PREDICTOR:
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
    for epoch in range(1, LATENT_TDA_PREDICTOR_EPOCHS + 1):
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


def run_latent_tda_trajectory_experiment(
    X_train=None,
    X_test=None,
    run_seeds=None,
    modes=None,
    max_train=None,
    max_test=None,
    display_fn=None,
):
    if X_train is None or X_test is None:
        print(
            "Skipping standalone latent-TDA cell: X_train/X_test are not defined. "
            "Provide X_train/X_test, or use the celltracking runner's built-in latent-TDA block."
        )
        empty = pd.DataFrame()
        return empty, empty, empty

    run_seeds = list(range(5)) if run_seeds is None else run_seeds
    modes = ["z", "z_latent_h0", "z_latent_h1", "z_latent_both"] if modes is None else modes
    rows = []
    for seed in run_seeds:
        print(f"\n================ latent TDA seed={seed} ================")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        Xtr = take_batch_subset(X_train, max_train, seed=seed)
        Xte = take_batch_subset(X_test, max_test, seed=seed + 1)
        encoder = load_or_train_shared_encoder_for_latent_tda(seed, Xtr, X_train)
        train_payload = load_or_compute_latent_tda_features(seed, "train", Xtr, encoder)
        test_payload = load_or_compute_latent_tda_features(seed, "test", Xte, encoder)

        for mode in modes:
            configure_runtime(CONTROL_SEED=seed)
            train_features = features_for_latent_tda_mode(train_payload, mode)
            configure_runtime(CONTROL_SEED=seed + 10_000)
            test_features = features_for_latent_tda_mode(test_payload, mode)
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
            }
            rows.append(row)
            print("latent TDA run summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=["test_mse", "latent_r2"],
        sort_metric="test_mse",
    )
    paired_df = results_df.pivot(index="seed", columns="mode", values="test_mse")
    if {"z", "z_latent_h1"}.issubset(paired_df.columns):
        paired_df["latent_h1_minus_z"] = paired_df["z_latent_h1"] - paired_df["z"]
        print("\nPaired latent_h1 - z differences; negative means latent TDA helped:")
        print(paired_df)

    print("\nLatent TDA trajectory summary:")
    print(summary_df)
    if display_fn is not None:
        display_fn(summary_df)
        display_fn(results_df.sort_values(["seed", "mode"]))
    return results_df, summary_df, paired_df
