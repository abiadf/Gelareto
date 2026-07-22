"""Streaming 3D cubical persistence features for video volumes.

This module treats each clip as a growing ``(t, y, x)`` cubical volume. The
public API is intentionally streaming: frames are appended one at a time and a
feature vector is emitted after each append. The current backend computes exact
H0/H1/H2 cubical persistence of the current prefix with CRiSPER. It keeps the
streaming state and boundary bookkeeping separate so a future local-update H2
backend can replace the full-prefix backend without changing the ML interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import cripser
except Exception:  # pragma: no cover - optional dependency checked at runtime.
    cripser = None

from topo import ml_tda
from topo import ml_tda_pixel
from topo.utils import tqdm_progress_bar


VIDEO3D_TDA_BINS = 16
VIDEO3D_TDA_SCALE = 15.0
VIDEO3D_TDA_BOUNDARY_SLICES = 2
VIDEO3D_TDA_PREDICTOR_EPOCHS = 4
VIDEO3D_TDA_LR = 3e-4
VIDEO3D_TDA_BATCH_SIZE = 32
VIDEO3D_TDA_RETRAIN_PREDICTOR = True
DATASET = "default"
LATENT_DIM = 128
HIDDEN_DIM = 128
HORIZON = 1


CONTROL_SUFFIXES = ("zero", "shuffle", "noise", "shift")
VIDEO3D_MODES = {
    "none",
    "h0",
    "h1",
    "h2",
    "h0_h1",
    "h0_h2",
    "h1_h2",
    "all",
}


def configure_runtime(**kwargs) -> None:
    """Update runner-controlled globals."""
    globals().update(kwargs)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DiagramSnapshot3D:
    """Persistence snapshot after appending one video frame."""

    t: int
    h0: np.ndarray
    h1: np.ndarray
    h2: np.ndarray
    volume_shape: tuple[int, int, int]
    damage_region_shape: tuple[int, int, int]


@dataclass
class StreamingCubicalPersistence3D:
    """Append frames and emit exact prefix H0/H1/H2 diagrams.

    The state stores all frames because the active backend recomputes the exact
    current-prefix cubical diagram. ``boundary_slices`` records the intended
    local-update damage radius for the future incremental backend.
    """

    boundary_slices: int = VIDEO3D_TDA_BOUNDARY_SLICES
    maxdim: int = 2
    frames: list[np.ndarray] = field(default_factory=list)

    def update(self, frame: np.ndarray) -> DiagramSnapshot3D:
        """Append one ``(H, W)`` frame and return current prefix diagrams."""
        if cripser is None:
            raise ImportError("video3d_tda requires cripser. Install it with `pip install cripser`.")
        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim != 2:
            raise ValueError(f"expected one 2D frame, got shape {frame.shape}")
        if self.frames and frame.shape != self.frames[0].shape:
            raise ValueError(f"expected frame shape {self.frames[0].shape}, got {frame.shape}")
        self.frames.append(frame)
        volume = np.stack(self.frames, axis=0).astype(np.float32, copy=False)
        start = max(0, len(self.frames) - int(self.boundary_slices) - 1)
        damage_region = volume[start:]
        h0, h1, h2 = compute_volume_diagrams(volume, maxdim=self.maxdim)
        return DiagramSnapshot3D(
            t=len(self.frames) - 1,
            h0=h0,
            h1=h1,
            h2=h2,
            volume_shape=tuple(int(v) for v in volume.shape),
            damage_region_shape=tuple(int(v) for v in damage_region.shape),
        )

    def diagram_snapshots(self) -> list[DiagramSnapshot3D]:
        """Return exact prefix snapshots for the frames already appended."""
        engine = StreamingCubicalPersistence3D(
            boundary_slices=self.boundary_slices,
            maxdim=self.maxdim,
        )
        return [engine.update(frame) for frame in self.frames]


def compute_volume_diagrams(volume: np.ndarray, maxdim: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute H0/H1/H2 diagrams for a 3D cubical volume."""
    if cripser is None:
        raise ImportError("3D cubical persistence requires cripser.")
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"expected volume with shape (T,H,W), got {volume.shape}")
    ph = cripser.compute_ph(volume, maxdim=maxdim)
    diagrams = []
    max_v = float(np.nanmax(volume)) if volume.size else 1.0
    for dim in (0, 1, 2):
        diag = ph[ph[:, 0] == dim][:, 1:3].astype(np.float32, copy=False)
        if diag.size == 0:
            diag = np.zeros((0, 2), dtype=np.float32)
        else:
            diag = diag.copy()
            diag[~np.isfinite(diag)] = max_v
        diagrams.append(diag)
    return diagrams[0], diagrams[1], diagrams[2]


def diagram_to_betti_curve(
    diagram: np.ndarray,
    *,
    num_steps: int = VIDEO3D_TDA_BINS,
    scale: float = VIDEO3D_TDA_SCALE,
    min_v: float = 0.0,
    max_v: float = 1.0,
) -> np.ndarray:
    """Vectorize one persistence diagram as a normalized Betti curve."""
    diagram = np.asarray(diagram, dtype=np.float32)
    curve = np.zeros((num_steps,), dtype=np.float32)
    if diagram.size == 0:
        return curve
    diagram = diagram.reshape(-1, 2)
    diagram = diagram.copy()
    diagram[~np.isfinite(diagram)] = max_v
    thresholds = np.linspace(min_v, max_v, num_steps, dtype=np.float32)
    alive = (diagram[:, 0][:, None] <= thresholds) & (diagram[:, 1][:, None] > thresholds)
    curve = alive.sum(axis=0).astype(np.float32)
    return np.clip(curve / float(scale), 0.0, 1.0)


def streaming_video_betti_features(
    frames: np.ndarray,
    *,
    num_steps: int = VIDEO3D_TDA_BINS,
    scale: float = VIDEO3D_TDA_SCALE,
    boundary_slices: int = VIDEO3D_TDA_BOUNDARY_SLICES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[DiagramSnapshot3D]]:
    """Return per-prefix H0/H1/H2 Betti curves for one ``(T,H,W)`` clip."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError(f"expected frames with shape (T,H,W), got {frames.shape}")
    engine = StreamingCubicalPersistence3D(boundary_slices=boundary_slices)
    h0 = np.zeros((frames.shape[0], num_steps), dtype=np.float32)
    h1 = np.zeros_like(h0)
    h2 = np.zeros_like(h0)
    snapshots = []
    for t, frame in enumerate(frames):
        snapshot = engine.update(frame)
        snapshots.append(snapshot)
        h0[t] = diagram_to_betti_curve(snapshot.h0, num_steps=num_steps, scale=scale)
        h1[t] = diagram_to_betti_curve(snapshot.h1, num_steps=num_steps, scale=scale)
        h2[t] = diagram_to_betti_curve(snapshot.h2, num_steps=num_steps, scale=scale)
    return h0, h1, h2, snapshots


def _video_tensor_to_numpy_frames(video_tensor: torch.Tensor) -> np.ndarray:
    """Convert ``(T,B,C,H,W)`` or ``(T,B,H,W)`` to float32 ``(T,B,H,W)``."""
    x = ml_tda.tensor_to_model_float(video_tensor).detach().cpu()
    if x.ndim == 5:
        x = x[:, :, 0]
    if x.ndim != 4:
        raise ValueError(f"expected video tensor as (T,B,C,H,W) or (T,B,H,W), got {tuple(x.shape)}")
    return x.numpy().astype(np.float32, copy=False)


def compute_video3d_tda_features(
    video_tensor: torch.Tensor,
    *,
    split_name: str,
    cache_path: Path | None = None,
    recompute: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute or load streaming 3D H0/H1/H2 features for a video split."""
    if cache_path is not None and cache_path.exists() and not recompute:
        print(f"Loading video3d TDA cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    frames = _video_tensor_to_numpy_frames(video_tensor)
    T, B, _, _ = frames.shape
    h0 = np.zeros((T, B, VIDEO3D_TDA_BINS), dtype=np.float32)
    h1 = np.zeros_like(h0)
    h2 = np.zeros_like(h0)
    damage_shapes = []
    for b in tqdm_progress_bar(range(B), desc=f"{split_name} video3d clips", total=B):
        h0_b, h1_b, h2_b, snapshots = streaming_video_betti_features(
            frames[:, b],
            num_steps=VIDEO3D_TDA_BINS,
            scale=VIDEO3D_TDA_SCALE,
            boundary_slices=VIDEO3D_TDA_BOUNDARY_SLICES,
        )
        h0[:, b] = h0_b
        h1[:, b] = h1_b
        h2[:, b] = h2_b
        if b == 0:
            damage_shapes = [snapshot.damage_region_shape for snapshot in snapshots]

    payload = {
        "h0": torch.from_numpy(h0).float(),
        "h1": torch.from_numpy(h1).float(),
        "h2": torch.from_numpy(h2).float(),
        "damage_region_shapes": damage_shapes,
        "bins": VIDEO3D_TDA_BINS,
        "scale": VIDEO3D_TDA_SCALE,
        "boundary_slices": VIDEO3D_TDA_BOUNDARY_SLICES,
        "backend": "cripser_full_prefix",
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
        print(f"Saved video3d TDA cache: {cache_path}")
    return payload


def feature_stats(features: torch.Tensor, prefix: str = "") -> dict[str, float]:
    """Return compact diagnostics for one TDA feature tensor."""
    x = features.detach().cpu().float()
    prefix = f"{prefix}_" if prefix else ""
    if x.numel() == 0:
        return {
            f"{prefix}mean": float("nan"),
            f"{prefix}std": float("nan"),
            f"{prefix}max": float("nan"),
            f"{prefix}nonzero_frac": float("nan"),
        }
    return {
        f"{prefix}mean": float(x.mean().item()),
        f"{prefix}std": float(x.std(unbiased=False).item()),
        f"{prefix}max": float(x.max().item()),
        f"{prefix}nonzero_frac": float((x != 0).float().mean().item()),
    }


def payload_feature_stats(payload: dict[str, torch.Tensor], prefix: str = "") -> dict[str, float]:
    """Return H0/H1/H2 diagnostics for a video3d TDA payload."""
    stats: dict[str, float] = {}
    base_prefix = f"{prefix}_" if prefix else ""
    for dim in ("h0", "h1", "h2"):
        if dim in payload:
            stats.update(feature_stats(payload[dim], prefix=f"{base_prefix}{dim}"))
    return stats


def print_payload_feature_stats(payload: dict[str, torch.Tensor], label: str) -> None:
    """Print whether H0/H1/H2 features are active or effectively constant."""
    print(f"{label} video3d TDA feature diagnostics:")
    for dim in ("h0", "h1", "h2"):
        if dim not in payload:
            continue
        stats = feature_stats(payload[dim])
        shape = tuple(payload[dim].shape)
        print(
            f"  {dim}: shape={shape} "
            f"mean={stats['mean']:.6f} std={stats['std']:.6f} "
            f"max={stats['max']:.6f} nonzero_frac={stats['nonzero_frac']:.4f}"
        )


def cache_path_for_split(video_tensor: torch.Tensor, *, seed: int, split_name: str) -> Path:
    """Build a shape/config-specific cache path."""
    tag = (
        f"seed{seed}_{split_name}_T{video_tensor.shape[0]}_B{video_tensor.shape[1]}_"
        f"H{video_tensor.shape[-2]}_W{video_tensor.shape[-1]}_"
        f"bins{VIDEO3D_TDA_BINS}_scale{VIDEO3D_TDA_SCALE:g}_bdry{VIDEO3D_TDA_BOUNDARY_SLICES}"
    )
    return Path("models") / DATASET / "video3d_tda_features" / f"{tag}.pt"


def compute_z_h0_h1_h2(video_tensor: torch.Tensor, encoder: nn.Module, *, split_name: str, seed: int, recompute: bool = False):
    """Encode frames and compute streaming 3D video TDA features."""
    device = ml_tda.get_runtime_device()
    encoder.to(device).eval()
    z_history = []
    with torch.no_grad():
        for t in tqdm_progress_bar(range(video_tensor.shape[0]), desc=f"{split_name} encode", total=video_tensor.shape[0]):
            frame_t = ml_tda.tensor_to_model_float(video_tensor[t]).to(device, non_blocking=True)
            z_history.append(encoder(frame_t).detach().cpu())
    z = torch.stack(z_history, dim=0)
    payload = compute_video3d_tda_features(
        video_tensor,
        split_name=split_name,
        cache_path=cache_path_for_split(video_tensor, seed=seed, split_name=split_name),
        recompute=recompute,
    )
    return z, payload["h0"], payload["h1"], payload["h2"], payload


def split_video3d_mode(mode: str) -> tuple[str, str]:
    """Split modes like ``h2_noise`` into base mode and control suffix."""
    mode = mode.strip()
    for suffix in CONTROL_SUFFIXES:
        marker = f"_{suffix}"
        if mode.endswith(marker):
            return mode[: -len(marker)], suffix
    return mode, "real"


def select_tda_block(h0: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor, base_mode: str) -> torch.Tensor | None:
    """Select H0/H1/H2 blocks for a base video3d mode."""
    if base_mode == "none":
        return None
    if base_mode == "h0":
        return h0
    if base_mode == "h1":
        return h1
    if base_mode == "h2":
        return h2
    if base_mode == "h0_h1":
        return torch.cat([h0, h1], dim=-1)
    if base_mode == "h0_h2":
        return torch.cat([h0, h2], dim=-1)
    if base_mode == "h1_h2":
        return torch.cat([h1, h2], dim=-1)
    if base_mode == "all":
        return torch.cat([h0, h1, h2], dim=-1)
    valid = ", ".join(sorted(VIDEO3D_MODES))
    raise ValueError(f"Unknown video3d TDA mode {base_mode!r}. Valid: {valid}")


def make_mode_features(z: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor, mode: str, seed: int) -> torch.Tensor:
    """Concatenate z with selected real or perturbed 3D TDA features."""
    base, control = split_video3d_mode(mode)
    tda = select_tda_block(h0, h1, h2, base)
    if tda is None:
        return z
    tda = ml_tda.apply_tda_control(tda, control=control, seed=seed, shift=1)
    return torch.cat([z, tda], dim=-1)


def mode_input_dim(mode: str) -> int:
    base, _ = split_video3d_mode(mode)
    if base == "none":
        return LATENT_DIM
    if base in {"h0", "h1", "h2"}:
        return LATENT_DIM + VIDEO3D_TDA_BINS
    if base in {"h0_h1", "h0_h2", "h1_h2"}:
        return LATENT_DIM + 2 * VIDEO3D_TDA_BINS
    if base == "all":
        return LATENT_DIM + 3 * VIDEO3D_TDA_BINS
    valid = ", ".join(sorted(VIDEO3D_MODES))
    raise ValueError(f"Unknown video3d TDA mode {base!r}. Valid: {valid}")


def predictor_path(seed: int, mode: str, train_features: torch.Tensor, output_size: tuple[int, int]) -> Path:
    model_dir = Path("models") / DATASET / "video3d_tda_predictors"
    model_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_mode{mode}_pred{HORIZON}_T{train_features.shape[0]}_B{train_features.shape[1]}_"
        f"H{output_size[0]}_W{output_size[1]}_latent{LATENT_DIM}_bins{VIDEO3D_TDA_BINS}"
    )
    return model_dir / f"{tag}.pt"


def train_or_load_predictor(seed: int, mode: str, train_features: torch.Tensor, x_train: torch.Tensor):
    """Train/load a future-frame predictor from z + video3d TDA features."""
    device = ml_tda.get_runtime_device()
    output_size = tuple(int(v) for v in x_train.shape[-2:])
    model = ml_tda_pixel.PixelFuturePredictor(
        input_dim=mode_input_dim(mode),
        output_size=output_size,
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    path = predictor_path(seed, mode, train_features, output_size)
    if path.exists() and not VIDEO3D_TDA_RETRAIN_PREDICTOR:
        print(f"Loading video3d predictor: {path}")
        model.load_state_dict(torch.load(path, map_location=device))
        return model, path

    x_all = train_features[:-HORIZON]
    y_all = ml_tda_pixel.target_frames(x_train[HORIZON:])
    batch_size = min(VIDEO3D_TDA_BATCH_SIZE, x_all.shape[1])
    opt = optim.AdamW(model.parameters(), lr=VIDEO3D_TDA_LR)
    generator = torch.Generator().manual_seed(seed + 3300)
    epochs = range(1, VIDEO3D_TDA_PREDICTOR_EPOCHS + 1)
    for epoch in tqdm_progress_bar(epochs, desc=f"video3d predictor {mode}", total=VIDEO3D_TDA_PREDICTOR_EPOCHS):
        model.train()
        perm = torch.randperm(x_all.shape[1], generator=generator)
        total_weighted, total_pixel, total_seen = 0.0, 0.0, 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            x_batch = x_all[:, idx].to(device, non_blocking=True)
            y_batch = y_all[:, idx].to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(x_batch)
            loss, pixel_mse, _, _ = ml_tda_pixel.pixel_losses(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_weighted += loss.item() * len(idx)
            total_pixel += pixel_mse.item() * len(idx)
            total_seen += len(idx)
        if epoch == 1 or epoch == VIDEO3D_TDA_PREDICTOR_EPOCHS:
            print(
                f"video3d {mode} seed={seed} epoch {epoch}/{VIDEO3D_TDA_PREDICTOR_EPOCHS} | "
                f"weighted_mse={total_weighted / max(total_seen, 1):.6f} "
                f"pixel_mse={total_pixel / max(total_seen, 1):.6f}"
            )
    torch.save(model.state_dict(), path)
    print(f"Saved video3d predictor: {path}")
    return model, path


def evaluate_predictor(model: nn.Module, test_features: torch.Tensor, x_test: torch.Tensor) -> dict[str, float]:
    """Evaluate future-frame prediction metrics."""
    old_batch = ml_tda_pixel.PIXEL_TDA_BATCH_SIZE
    old_horizon = ml_tda_pixel.HORIZON
    try:
        ml_tda_pixel.configure_runtime(
            PIXEL_TDA_BATCH_SIZE=VIDEO3D_TDA_BATCH_SIZE,
            HORIZON=HORIZON,
        )
        return ml_tda_pixel.evaluate_predictor(model, test_features, x_test)
    finally:
        ml_tda_pixel.configure_runtime(PIXEL_TDA_BATCH_SIZE=old_batch, HORIZON=old_horizon)
