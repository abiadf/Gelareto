"""Auxiliary topology prediction helpers for video forecasting experiments."""

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from topo import ml_tda
from topo.utils import tqdm_progress_bar


DATASET = "default"
LATENT_DIM = 128
HIDDEN_DIM = 128
HORIZON = 1
REAL_TDA_BINS = 25
REAL_TDA_SCALE = 15
AUX_TDA_LAMBDA = 0.1
AUX_TDA_PREDICTOR_EPOCHS = 6
AUX_TDA_LR = 3e-4
AUX_TDA_RETRAIN_ENCODER = False
AUX_TDA_RETRAIN_PREDICTOR = True


def configure_runtime(**kwargs):
    """Update notebook-controlled globals used by auxiliary TDA helpers."""
    globals().update(kwargs)


def _round_metric(value, digits=4):
    """Round finite scalar metrics while preserving NaN."""
    scalar = float(value)
    if not np.isfinite(scalar):
        return scalar
    return round(scalar, digits)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AuxTopologyPredictor(nn.Module):
    """Predict future latent z and, optionally, future Betti features."""

    def __init__(self, latent_dim, hidden_dim=128, betti_dim=0):
        super().__init__()
        self.betti_dim = int(betti_dim)
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, batch_first=False)
        self.z_head = nn.Linear(hidden_dim, latent_dim)
        self.b_head = nn.Linear(hidden_dim, betti_dim) if betti_dim > 0 else None

    def forward(self, z_history):
        h, _ = self.lstm(z_history)
        pred_z = self.z_head(h)
        pred_b = self.b_head(h) if self.b_head is not None else None
        return pred_z, pred_b


def aux_mode_betti_dim(mode):
    if mode == "none":
        return 0
    if mode in {"aux_h0", "aux_h1"}:
        return REAL_TDA_BINS
    if mode == "aux_both":
        return 2 * REAL_TDA_BINS
    raise ValueError(f"Unknown auxiliary TDA mode: {mode}")


def select_betti_target(h0, h1, mode):
    if mode == "none":
        return None
    if mode == "aux_h0":
        return h0
    if mode == "aux_h1":
        return h1
    if mode == "aux_both":
        return torch.cat([h0, h1], dim=-1)
    raise ValueError(f"Unknown auxiliary TDA mode: {mode}")


def load_or_train_aux_encoder(X_train, seed, output_size):
    encoder = ml_tda.SpatialEncoder(latent_dim=LATENT_DIM)
    decoder = ml_tda.SpatialDecoder(latent_dim=LATENT_DIM, output_size=output_size)
    model_dir = Path("models") / DATASET
    encoder_dir = model_dir / "encoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    encoder_tag = (
        f"seed{seed}_T{X_train.shape[0]}_B{X_train.shape[1]}_"
        f"H{X_train.shape[-2]}_W{X_train.shape[-1]}_latent{LATENT_DIM}"
    )
    encoder_path = encoder_dir / f"encoder_{encoder_tag}.pt"

    if encoder_path.exists() and not AUX_TDA_RETRAIN_ENCODER:
        print(f"Loading shared auxiliary encoder: {encoder_path}")
        encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    else:
        reason = "retraining" if encoder_path.exists() else "missing; training once"
        print(f"Auxiliary encoder {reason}: {encoder_path}")
        ml_tda.pretrain_spatial_encoder(encoder, decoder, X_train, ae_epochs=int(getattr(ml_tda, "AE_EPOCHS", 3)))
        torch.save(encoder.state_dict(), encoder_path)
        print(f"Saved auxiliary encoder: {encoder_path}")

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
        split_name=f"{split_name} aux h0+h1 cache",
    )
    h0 = both_features[..., LATENT_DIM:LATENT_DIM + REAL_TDA_BINS]
    h1 = both_features[..., LATENT_DIM + REAL_TDA_BINS:LATENT_DIM + 2 * REAL_TDA_BINS]
    ml_tda.configure_runtime(TDA_MODE=old_mode)
    return z_seq, h0, h1


def make_supervised_sequences(z_seq, b_seq, horizon):
    x = z_seq[:-horizon]
    y_z = z_seq[horizon:]
    y_b = b_seq[horizon:] if b_seq is not None else None
    return x, y_z, y_b


def predictor_path(seed, mode, train_z):
    model_dir = Path("models") / DATASET / "aux_tda_predictors"
    model_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_mode{mode}_pred{HORIZON}_"
        f"T{train_z.shape[0]}_B{train_z.shape[1]}_latent{LATENT_DIM}_"
        f"realtdabins{REAL_TDA_BINS}_lambda{AUX_TDA_LAMBDA}"
    )
    return model_dir / f"{tag}.pt"


def train_or_load_predictor(seed, mode, train_z, train_b):
    device = ml_tda.get_runtime_device()
    betti_dim = aux_mode_betti_dim(mode)
    model = AuxTopologyPredictor(LATENT_DIM, hidden_dim=HIDDEN_DIM, betti_dim=betti_dim).to(device)
    path = predictor_path(seed, mode, train_z)

    if path.exists() and not AUX_TDA_RETRAIN_PREDICTOR:
        print(f"Loading auxiliary predictor: {path}")
        model.load_state_dict(torch.load(path, map_location=device))
        return model, path

    x_train, y_z_train, y_b_train = make_supervised_sequences(train_z, train_b, HORIZON)
    opt = optim.AdamW(model.parameters(), lr=AUX_TDA_LR)
    loss_fn = nn.MSELoss()
    epochs = range(1, AUX_TDA_PREDICTOR_EPOCHS + 1)
    for epoch in tqdm_progress_bar(epochs, desc=f"Aux predictor {mode}", total=AUX_TDA_PREDICTOR_EPOCHS):
        model.train()
        opt.zero_grad()
        pred_z, pred_b = model(x_train)
        z_loss = loss_fn(pred_z, y_z_train)
        if y_b_train is not None:
            b_loss = loss_fn(pred_b, y_b_train)
            loss = z_loss + AUX_TDA_LAMBDA * b_loss
        else:
            b_loss = torch.tensor(0.0, device=device)
            loss = z_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if epoch == 1 or epoch == AUX_TDA_PREDICTOR_EPOCHS:
            print(
                f"aux {mode} seed={seed} epoch {epoch}/{AUX_TDA_PREDICTOR_EPOCHS} | "
                f"z_mse={z_loss.item():.4f} b_mse={b_loss.item():.4f} total={loss.item():.4f}"
            )

    torch.save(model.state_dict(), path)
    print(f"Saved auxiliary predictor: {path}")
    return model, path


def evaluate_predictor(model, mode, test_z, test_b):
    loss_fn = nn.MSELoss()
    x_test, y_z_test, y_b_test = make_supervised_sequences(test_z, test_b, HORIZON)
    model.eval()
    with torch.no_grad():
        pred_z, pred_b = model(x_test)
        z_mse = loss_fn(pred_z, y_z_test).item()
        per_frame_z_mse = ((pred_z - y_z_test) ** 2).mean(dim=(1, 2)).detach().cpu()
        target_z = y_z_test.detach().float().cpu()
        target_var = torch.mean((target_z - target_z.mean()) ** 2).item()
        z_r2 = float(1.0 - (z_mse / max(target_var, 1e-12)))
        if y_b_test is not None:
            b_mse = loss_fn(pred_b, y_b_test).item()
        else:
            b_mse = float("nan")
    return z_mse, b_mse, per_frame_z_mse, z_r2


def run_aux_tda_experiment(X_train, X_test, seeds, modes, display_fn=None):
    rows = []
    runs = {}
    ml_tda.configure_runtime(
        DATASET=DATASET,
        LATENT_DIM=LATENT_DIM,
        REAL_TDA_SCALE=REAL_TDA_SCALE,
        REAL_TDA_BINS=REAL_TDA_BINS,
        HORIZON=HORIZON,
    )
    print(f"Aux TDA device: {ml_tda.get_runtime_device()}")

    for seed in seeds:
        print(f"\n================ aux TDA seed={seed} ================")
        set_all_seeds(seed)
        encoder, encoder_path = load_or_train_aux_encoder(X_train, seed, output_size=X_train.shape[-2:])
        train_z, train_h0, train_h1 = compute_z_h0_h1(X_train, encoder, split_name="train")
        test_z, test_h0, test_h1 = compute_z_h0_h1(X_test, encoder, split_name="test")

        for mode in modes:
            print(f"\n--- aux mode={mode} seed={seed} ---")
            train_b = select_betti_target(train_h0, train_h1, mode)
            test_b = select_betti_target(test_h0, test_h1, mode)
            model, model_path = train_or_load_predictor(seed, mode, train_z, train_b)
            z_mse, b_mse, per_frame_z_mse, z_r2 = evaluate_predictor(model, mode, test_z, test_b)
            row = {
                "dataset": DATASET,
                "seed": seed,
                "mode": mode,
                "horizon": HORIZON,
                "lambda_topo": AUX_TDA_LAMBDA,
                "z_mse": _round_metric(z_mse),
                "z_r2": _round_metric(z_r2),
                "b_mse": _round_metric(b_mse),
                "encoder_path": str(encoder_path),
                "model_path": str(model_path),
            }
            rows.append(row)
            runs[(seed, mode)] = {"row": row, "per_frame_z_mse": per_frame_z_mse}
            print("aux summary:", row)

    import pandas as pd

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=["z_mse", "z_r2", "b_mse"],
        sort_metric="z_mse",
    )
    print("\nAux TDA per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nAux TDA mean +/- std:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    if display_fn is not None:
        display_fn(summary_df)
        display_fn(results_df.sort_values(["seed", "mode"]))
    return results_df, summary_df, runs
