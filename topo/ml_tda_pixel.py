"""Pixel prediction helpers for z + real/frame-space TDA video experiments."""

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from topo import ml_tda


DATASET = "default"
LATENT_DIM = 128
HIDDEN_DIM = 128
HORIZON = 1
REAL_TDA_BINS = 25
REAL_TDA_SCALE = 15
PIXEL_TDA_PREDICTOR_EPOCHS = 4
PIXEL_TDA_LR = 3e-4
PIXEL_TDA_BATCH_SIZE = 32
PIXEL_TDA_FG_WEIGHT = 10.0
PIXEL_TDA_FG_THRESHOLD = 0.05
PIXEL_TDA_RETRAIN_ENCODER = False
PIXEL_TDA_RETRAIN_PREDICTOR = True


def configure_runtime(**kwargs):
    """Update notebook-controlled globals used by pixel TDA helpers."""
    globals().update(kwargs)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PixelFuturePredictor(nn.Module):
    """Predict future image frames from latent/TDA feature histories."""

    def __init__(self, input_dim, output_size, hidden_dim=128):
        super().__init__()
        self.output_size = tuple(output_size)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, self.output_size[0] * self.output_size[1]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        y = self.head(h)
        return y.reshape(x.shape[0], x.shape[1], 1, self.output_size[0], self.output_size[1])


def mode_input_dim(mode):
    base, _ = ml_tda.parse_tda_control_mode(mode)
    if base == "none":
        return LATENT_DIM
    if base in {"h0", "h1"}:
        return LATENT_DIM + REAL_TDA_BINS
    if base == "both":
        return LATENT_DIM + 2 * REAL_TDA_BINS
    raise ValueError(f"Unknown pixel TDA mode: {mode}")


def load_or_train_pixel_encoder(X_train, seed, output_size):
    encoder = ml_tda.SpatialEncoder(latent_dim=LATENT_DIM)
    decoder = ml_tda.SpatialDecoder(latent_dim=LATENT_DIM, output_size=output_size)
    encoder_dir = Path("models") / DATASET / "encoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    encoder_tag = (
        f"seed{seed}_T{X_train.shape[0]}_B{X_train.shape[1]}_"
        f"H{X_train.shape[-2]}_W{X_train.shape[-1]}_latent{LATENT_DIM}"
    )
    encoder_path = encoder_dir / f"encoder_{encoder_tag}.pt"

    if encoder_path.exists() and not PIXEL_TDA_RETRAIN_ENCODER:
        print(f"Loading shared pixel encoder: {encoder_path}")
        encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    else:
        reason = "retraining" if encoder_path.exists() else "missing; training once"
        print(f"Pixel encoder {reason}: {encoder_path}")
        ml_tda.pretrain_spatial_encoder(encoder, decoder, X_train, ae_epochs=3)
        torch.save(encoder.state_dict(), encoder_path)
        print(f"Saved shared pixel encoder: {encoder_path}")

    device = ml_tda.get_runtime_device()
    encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder, encoder_path


def compute_z_h0_h1(video_tensor, encoder, split_name):
    old_mode = getattr(ml_tda, "TDA_MODE", "none")
    ml_tda.configure_runtime(TDA_MODE="both")
    both_features, z_seq = ml_tda.build_real_tda_features(
        video_tensor,
        encoder,
        use_tda=True,
        split_name=f"{split_name} pixel h0+h1 cache",
    )
    h0 = both_features[..., LATENT_DIM:LATENT_DIM + REAL_TDA_BINS]
    h1 = both_features[..., LATENT_DIM + REAL_TDA_BINS:LATENT_DIM + 2 * REAL_TDA_BINS]
    ml_tda.configure_runtime(TDA_MODE=old_mode)
    return z_seq, h0, h1


def make_mode_features(z_seq, h0, h1, mode, seed):
    return ml_tda.build_controlled_tda_features(z_seq, h0, h1, mode, seed=seed, shift=1)


def predictor_path(seed, mode, train_features, output_size):
    model_dir = Path("models") / DATASET / "pixel_tda_predictors"
    model_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_mode{mode}_pred{HORIZON}_"
        f"T{train_features.shape[0]}_B{train_features.shape[1]}_"
        f"H{output_size[0]}_W{output_size[1]}_latent{LATENT_DIM}_realtdabins{REAL_TDA_BINS}_"
        f"fgw{PIXEL_TDA_FG_WEIGHT}"
    )
    return model_dir / f"{tag}.pt"


def pixel_losses(pred, target):
    err = (pred - target) ** 2
    fg_mask = target > PIXEL_TDA_FG_THRESHOLD
    weight = torch.ones_like(target)
    weight = weight + fg_mask.float() * PIXEL_TDA_FG_WEIGHT
    weighted_mse = (err * weight).mean()
    pixel_mse = err.mean()
    if fg_mask.any():
        foreground_mse = err[fg_mask].mean()
    else:
        foreground_mse = torch.tensor(float("nan"), device=pred.device)
    background_mask = ~fg_mask
    if background_mask.any():
        background_mse = err[background_mask].mean()
    else:
        background_mse = torch.tensor(float("nan"), device=pred.device)
    return weighted_mse, pixel_mse, foreground_mse, background_mse


def target_frames(video_tensor):
    return ml_tda.tensor_to_model_float(video_tensor)


def train_or_load_predictor(seed, mode, train_features, X_train):
    device = ml_tda.get_runtime_device()
    output_size = X_train.shape[-2:]
    model = PixelFuturePredictor(
        input_dim=mode_input_dim(mode),
        output_size=output_size,
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    path = predictor_path(seed, mode, train_features, output_size)

    if path.exists() and not PIXEL_TDA_RETRAIN_PREDICTOR:
        print(f"Loading pixel predictor: {path}")
        model.load_state_dict(torch.load(path, map_location=device))
        return model, path

    x_all = train_features[:-HORIZON]
    y_all = target_frames(X_train[HORIZON:])
    batch_size = min(PIXEL_TDA_BATCH_SIZE, x_all.shape[1])
    opt = optim.AdamW(model.parameters(), lr=PIXEL_TDA_LR)
    generator = torch.Generator().manual_seed(seed + 1234)

    for epoch in range(1, PIXEL_TDA_PREDICTOR_EPOCHS + 1):
        model.train()
        perm = torch.randperm(x_all.shape[1], generator=generator)
        total_weighted, total_pixel, total_seen = 0.0, 0.0, 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            x_batch = x_all[:, idx].to(device, non_blocking=True)
            y_batch = y_all[:, idx].to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(x_batch)
            loss, pixel_mse, _, _ = pixel_losses(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            seen = len(idx)
            total_weighted += loss.item() * seen
            total_pixel += pixel_mse.item() * seen
            total_seen += seen
        if epoch == 1 or epoch == PIXEL_TDA_PREDICTOR_EPOCHS:
            print(
                f"pixel {mode} seed={seed} epoch {epoch}/{PIXEL_TDA_PREDICTOR_EPOCHS} | "
                f"weighted_mse={total_weighted / total_seen:.6f} "
                f"pixel_mse={total_pixel / total_seen:.6f}"
            )

    torch.save(model.state_dict(), path)
    print(f"Saved pixel predictor: {path}")
    return model, path


def evaluate_predictor(model, test_features, X_test):
    device = ml_tda.get_runtime_device()
    x_all = test_features[:-HORIZON]
    y_all = target_frames(X_test[HORIZON:])
    batch_size = min(PIXEL_TDA_BATCH_SIZE, x_all.shape[1])
    sums = {"weighted": 0.0, "pixel": 0.0, "foreground": 0.0, "background": 0.0}
    counts = {"weighted": 0, "pixel": 0, "foreground": 0, "background": 0}
    per_time_sse = torch.zeros(x_all.shape[0], dtype=torch.float64)
    per_time_count = torch.zeros(x_all.shape[0], dtype=torch.float64)
    total_sse = 0.0
    total_target_sum = 0.0
    total_target_sq_sum = 0.0
    total_target_count = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, x_all.shape[1], batch_size):
            idx = torch.arange(start, min(start + batch_size, x_all.shape[1]))
            x_batch = x_all[:, idx].to(device, non_blocking=True)
            y_batch = y_all[:, idx].to(device, non_blocking=True)
            pred = model(x_batch)
            weighted_mse, pixel_mse, fg_mse, bg_mse = pixel_losses(pred, y_batch)
            seen = len(idx)
            sums["weighted"] += weighted_mse.item() * seen
            sums["pixel"] += pixel_mse.item() * seen
            counts["weighted"] += seen
            counts["pixel"] += seen
            if torch.isfinite(fg_mse):
                sums["foreground"] += fg_mse.item() * seen
                counts["foreground"] += seen
            if torch.isfinite(bg_mse):
                sums["background"] += bg_mse.item() * seen
                counts["background"] += seen
            err = ((pred - y_batch) ** 2).detach().cpu().double()
            per_time_sse += err.sum(dim=(1, 2, 3, 4))
            per_time_count += err[0].numel()
            y_cpu = y_batch.detach().cpu().double()
            total_sse += float(err.sum())
            total_target_sum += float(y_cpu.sum())
            total_target_sq_sum += float((y_cpu ** 2).sum())
            total_target_count += y_cpu.numel()

    per_time_pixel_mse = (per_time_sse / per_time_count.clamp_min(1)).float()
    target_mean = total_target_sum / max(total_target_count, 1)
    target_var = total_target_sq_sum / max(total_target_count, 1) - target_mean ** 2
    pixel_mse_global = total_sse / max(total_target_count, 1)
    pixel_r2 = 1.0 - (pixel_mse_global / max(target_var, 1e-12))
    return {
        "weighted_mse": sums["weighted"] / max(counts["weighted"], 1),
        "pixel_mse": sums["pixel"] / max(counts["pixel"], 1),
        "pixel_r2": pixel_r2,
        "foreground_mse": sums["foreground"] / counts["foreground"] if counts["foreground"] else float("nan"),
        "background_mse": sums["background"] / counts["background"] if counts["background"] else float("nan"),
        "warmup_excluded_pixel_mse": float(per_time_pixel_mse[1:].mean()) if len(per_time_pixel_mse) > 1 else float(per_time_pixel_mse.mean()),
        "per_time_pixel_mse": per_time_pixel_mse,
    }


def run_pixel_tda_experiment(X_train, X_test, seeds, modes, display_fn=None):
    rows = []
    runs = {}
    ml_tda.configure_runtime(
        DATASET=DATASET,
        LATENT_DIM=LATENT_DIM,
        REAL_TDA_SCALE=REAL_TDA_SCALE,
        REAL_TDA_BINS=REAL_TDA_BINS,
        HORIZON=HORIZON,
    )
    print(f"Pixel TDA device: {ml_tda.get_runtime_device()}")

    for seed in seeds:
        print(f"\n================ pixel TDA seed={seed} ================")
        set_all_seeds(seed)
        encoder, encoder_path = load_or_train_pixel_encoder(X_train, seed, output_size=X_train.shape[-2:])
        train_z, train_h0, train_h1 = compute_z_h0_h1(X_train, encoder, split_name="train")
        test_z, test_h0, test_h1 = compute_z_h0_h1(X_test, encoder, split_name="test")

        for mode in modes:
            print(f"\n--- pixel mode={mode} seed={seed} ---")
            train_features = make_mode_features(train_z, train_h0, train_h1, mode, seed)
            test_features = make_mode_features(test_z, test_h0, test_h1, mode, seed + 10_000)
            model, model_path = train_or_load_predictor(seed, mode, train_features, X_train)
            metrics = evaluate_predictor(model, test_features, X_test)
            row = {
                "dataset": DATASET,
                "seed": seed,
                "mode": mode,
                "horizon": HORIZON,
                "pixel_mse": float(metrics["pixel_mse"]),
                "pixel_r2": float(metrics["pixel_r2"]),
                "weighted_mse": float(metrics["weighted_mse"]),
                "foreground_mse": float(metrics["foreground_mse"]),
                "background_mse": float(metrics["background_mse"]),
                "warmup_excluded_pixel_mse": float(metrics["warmup_excluded_pixel_mse"]),
                "fg_weight": PIXEL_TDA_FG_WEIGHT,
                "fg_threshold": PIXEL_TDA_FG_THRESHOLD,
                "encoder_path": str(encoder_path),
                "model_path": str(model_path),
            }
            rows.append(row)
            runs[(seed, mode)] = {"row": row, "per_time_pixel_mse": metrics["per_time_pixel_mse"]}
            print("pixel summary:", row)

    import pandas as pd

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=[
            "pixel_mse",
            "pixel_r2",
            "weighted_mse",
            "foreground_mse",
            "background_mse",
            "warmup_excluded_pixel_mse",
        ],
        sort_metric="weighted_mse",
    )
    print("\nPixel TDA per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nPixel TDA mean +/- std:")
    print(summary_df)
    if display_fn is not None:
        display_fn(summary_df)
        display_fn(results_df.sort_values(["seed", "mode"]))
    return results_df, summary_df, runs
