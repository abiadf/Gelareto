"""SimVP-style frame prediction with optional latent/TDA conditioning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from topo import ml_tda
from topo import ml_tda_pixel
from topo.utils import tqdm_progress_bar


DATASET = "default"
HORIZON = 1
SIMVP_INPUT_FRAMES = 5
SIMVP_HIDDEN_DIM = 64
SIMVP_EPOCHS = 10
SIMVP_LR = 3e-4
SIMVP_BATCH_SIZE = 16
SIMVP_RETRAIN = True


def configure_runtime(**kwargs):
    globals().update(kwargs)


class SimVPBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class SimVPFramePredictor(nn.Module):
    """Compact SimVP-style encoder-translator-decoder for one future frame."""

    def __init__(self, in_channels: int, hidden_dim: int, output_size: tuple[int, int], cond_dim: int = 0):
        super().__init__()
        self.output_size = tuple(output_size)
        self.cond_dim = int(cond_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.translator = nn.Sequential(
            SimVPBlock(hidden_dim),
            SimVPBlock(hidden_dim),
            SimVPBlock(hidden_dim),
        )
        self.cond_proj = (
            nn.Sequential(nn.LayerNorm(cond_dim), nn.Linear(cond_dim, hidden_dim))
            if cond_dim > 0
            else None
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # x: [N, L, C, H, W]
        n, length, channels, height, width = x.shape
        h = self.encoder(x.reshape(n * length, channels, height, width))
        h = h.reshape(n, length, h.shape[1], h.shape[2], h.shape[3]).permute(0, 2, 1, 3, 4)
        h = self.translator(h)
        h_last = h[:, :, -1]
        if self.cond_proj is not None:
            if cond is None:
                raise ValueError("Conditioned SimVP model requires cond features.")
            h_last = h_last + self.cond_proj(cond)[:, :, None, None]
        pred = self.decoder(h_last)
        if pred.shape[-2:] != self.output_size:
            pred = nn.functional.interpolate(pred, size=self.output_size, mode="bilinear", align_corners=False)
        return pred


def is_unconditioned_mode(mode: str) -> bool:
    return str(mode).lower().strip() in {"frames", "none", "simvp"}


def _sample_indices(video_tensor: torch.Tensor, input_frames: int, horizon: int) -> np.ndarray:
    n_times, n_clips = video_tensor.shape[:2]
    last_source = n_times - horizon - 1
    if last_source < input_frames - 1:
        raise ValueError(
            f"Need at least input_frames+horizon frames; got T={n_times}, "
            f"input_frames={input_frames}, horizon={horizon}"
        )
    pairs = [(t, b) for t in range(input_frames - 1, last_source + 1) for b in range(n_clips)]
    return np.asarray(pairs, dtype=np.int64)


def _batch_from_indices(
    video_tensor: torch.Tensor,
    indices: np.ndarray,
    input_frames: int,
    horizon: int,
    cond_features: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    xs = []
    ys = []
    conds = [] if cond_features is not None else None
    video = ml_tda.tensor_to_model_float(video_tensor)
    for source_t, clip_idx in indices:
        source_t = int(source_t)
        clip_idx = int(clip_idx)
        xs.append(video[source_t - input_frames + 1 : source_t + 1, clip_idx])
        ys.append(video[source_t + horizon, clip_idx])
        if conds is not None:
            conds.append(cond_features[source_t, clip_idx])
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    cond = torch.stack(conds, dim=0) if conds is not None else None
    return x, y, cond


def _model_path(seed: int, mode: str, cond_dim: int, output_size: tuple[int, int]) -> Path:
    model_dir = Path("models") / DATASET / "simvp_predictors"
    model_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_mode{mode}_pred{HORIZON}_in{SIMVP_INPUT_FRAMES}_"
        f"cond{cond_dim}_hid{SIMVP_HIDDEN_DIM}_H{output_size[0]}_W{output_size[1]}"
    )
    return model_dir / f"model_{tag}.pt"


def train_or_load_simvp(
    *,
    seed: int,
    mode: str,
    train_video: torch.Tensor,
    cond_features: torch.Tensor | None = None,
) -> tuple[SimVPFramePredictor, Path]:
    device = ml_tda.get_runtime_device()
    cond_dim = 0 if cond_features is None else int(cond_features.shape[-1])
    model = SimVPFramePredictor(
        in_channels=int(train_video.shape[2]),
        hidden_dim=int(SIMVP_HIDDEN_DIM),
        output_size=tuple(train_video.shape[-2:]),
        cond_dim=cond_dim,
    ).to(device)
    path = _model_path(seed, mode, cond_dim, tuple(train_video.shape[-2:]))
    if path.exists() and not SIMVP_RETRAIN:
        print(f"Loading SimVP predictor: {path}")
        model.load_state_dict(torch.load(path, map_location=device))
        return model, path

    print(f"Training SimVP predictor: {path}")
    indices = _sample_indices(train_video, int(SIMVP_INPUT_FRAMES), int(HORIZON))
    rng = np.random.default_rng(seed + 2024)
    optimizer = optim.AdamW(model.parameters(), lr=float(SIMVP_LR))
    batch_size = min(int(SIMVP_BATCH_SIZE), len(indices))
    epochs = range(1, int(SIMVP_EPOCHS) + 1)
    for epoch in tqdm_progress_bar(epochs, desc=f"SimVP {mode}", total=int(SIMVP_EPOCHS)):
        model.train()
        rng.shuffle(indices)
        total_loss = 0.0
        total_seen = 0
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            x, y, cond = _batch_from_indices(train_video, batch_idx, int(SIMVP_INPUT_FRAMES), int(HORIZON), cond_features)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True) if cond is not None else None
            optimizer.zero_grad()
            pred = model(x, cond)
            loss, _, _, _ = ml_tda_pixel.pixel_losses(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_idx)
            total_seen += len(batch_idx)
        if epoch == 1 or epoch == int(SIMVP_EPOCHS):
            print(f"SimVP {mode} seed={seed} epoch {epoch}/{SIMVP_EPOCHS} weighted_mse={total_loss / max(total_seen, 1):.6f}")
    torch.save(model.state_dict(), path)
    print(f"Saved SimVP predictor: {path}")
    return model, path


def _encode_frames(encoder: nn.Module, frames: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    device = ml_tda.get_runtime_device()
    encoder.to(device).eval()
    latents = []
    with torch.no_grad():
        for start in range(0, frames.shape[0], batch_size):
            latents.append(encoder(frames[start:start + batch_size].to(device)).cpu())
    return torch.cat(latents, dim=0)


def evaluate_simvp(
    *,
    model: SimVPFramePredictor,
    test_video: torch.Tensor,
    cond_features: torch.Tensor | None = None,
    encoder: nn.Module | None = None,
) -> dict[str, float]:
    device = ml_tda.get_runtime_device()
    indices = _sample_indices(test_video, int(SIMVP_INPUT_FRAMES), int(HORIZON))
    batch_size = min(int(SIMVP_BATCH_SIZE), len(indices))
    total_sse = 0.0
    total_target_sum = 0.0
    total_target_sq_sum = 0.0
    total_count = 0
    weighted_sum = 0.0
    fg_sum = 0.0
    fg_count = 0
    bg_sum = 0.0
    bg_count = 0
    pred_frames = []
    target_frames = []
    model.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            x, y, cond = _batch_from_indices(test_video, batch_idx, int(SIMVP_INPUT_FRAMES), int(HORIZON), cond_features)
            x = x.to(device, non_blocking=True)
            y_device = y.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True) if cond is not None else None
            pred = model(x, cond)
            weighted_mse, _, fg_mse, bg_mse = ml_tda_pixel.pixel_losses(pred, y_device)
            err = ((pred - y_device) ** 2).detach().cpu().double()
            y_cpu = y.double()
            n = y_cpu.numel()
            total_sse += float(err.sum())
            total_target_sum += float(y_cpu.sum())
            total_target_sq_sum += float((y_cpu ** 2).sum())
            total_count += n
            weighted_sum += float(weighted_mse.item()) * len(batch_idx)
            if torch.isfinite(fg_mse):
                fg_sum += float(fg_mse.item()) * len(batch_idx)
                fg_count += len(batch_idx)
            if torch.isfinite(bg_mse):
                bg_sum += float(bg_mse.item()) * len(batch_idx)
                bg_count += len(batch_idx)
            if encoder is not None:
                pred_frames.append(pred.detach().cpu())
                target_frames.append(y.detach().cpu())
    pixel_mse = total_sse / max(total_count, 1)
    target_mean = total_target_sum / max(total_count, 1)
    target_var = total_target_sq_sum / max(total_count, 1) - target_mean ** 2
    metrics = {
        "pixel_mse": float(pixel_mse),
        "pixel_r2": float(1.0 - pixel_mse / max(target_var, 1e-12)),
        "weighted_mse": float(weighted_sum / max(len(indices), 1)),
        "foreground_mse": float(fg_sum / fg_count) if fg_count else float("nan"),
        "background_mse": float(bg_sum / bg_count) if bg_count else float("nan"),
    }
    if encoder is not None and pred_frames:
        pred_z = _encode_frames(encoder, torch.cat(pred_frames, dim=0))
        target_z = _encode_frames(encoder, torch.cat(target_frames, dim=0))
        latent_mse = torch.mean((pred_z - target_z) ** 2).item()
        latent_var = torch.mean((target_z - target_z.mean()) ** 2).item()
        metrics["latent_mse"] = float(latent_mse)
        metrics["latent_r2"] = float(1.0 - latent_mse / max(latent_var, 1e-12))
    return metrics
