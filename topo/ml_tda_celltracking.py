"""Celltracking-specific preprocessing and TDA ablation helpers."""

from pathlib import Path
import glob
import os
import random

import cripser
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import torch.optim as optim
from ripser import ripser

try:
    import cv2
except ImportError:  # pragma: no cover - notebook dependency fallback
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - notebook dependency fallback
    Image = None

from topo.ml_tda import (
    SpatialEncoder,
    SpatialDecoder,
    prepare_clip_tensor_for_sequence_pipeline,
    pretrain_spatial_encoder_on_clips,
)


CELLTRACKING_MODEL_NAME = "celltracking_clips"
PREDICT_STEPS_AHEAD = 5
N_STEPS_LOCAL = 25
BETTI_SCALE_LOCAL = 15
LATENT_DIM_LOCAL = 128
AE_EPOCHS = 2
AE_MAX_FRAMES_PER_EPOCH = 4096
MLP_EPOCHS = 6
MLP_BATCH_SIZE = 512
LEARNING_RATE = 5e-4


def configure_runtime(**kwargs):
    """Update notebook-controlled globals used by celltracking helpers."""
    globals().update(kwargs)


def load_raw_video_stack(folder_path):
    """Read a CTC-style raw frame folder into (T, 1, H, W)."""
    file_paths = sorted(glob.glob(os.path.join(str(folder_path), "*.tif")))
    if not file_paths:
        raise FileNotFoundError(f"No .tif files found in: {folder_path}")

    frames = []
    for fp in file_paths:
        try:
            frame = tiff.imread(fp)
        except ValueError as exc:
            if "imagecodecs" not in str(exc) and "COMPRESSION" not in str(exc):
                raise
            frame = None
            if cv2 is not None:
                frame = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            if frame is None:
                if Image is None:
                    raise ImportError("Install imagecodecs, Pillow, or OpenCV to read this TIFF compression")
                frame = np.asarray(Image.open(fp))
        if frame.ndim > 2:
            frame = np.squeeze(frame)
            if frame.ndim > 2:
                frame = frame[0]
        frames.append(frame.astype(np.float32))

    video_stack = np.stack(frames, axis=0)[:, np.newaxis, :, :]
    print(f"Loaded raw sequence {folder_path}: {video_stack.shape}")
    return video_stack


def normalize_video_stack(video_stack, mode="minmax", percentiles=(1, 99.8)):
    video_stack = video_stack.astype(np.float32)
    if mode == "percentile":
        lo, hi = np.percentile(video_stack, percentiles)
        if hi <= lo:
            hi = lo + 1.0
        video_stack = np.clip((video_stack - lo) / (hi - lo), 0.0, 1.0)
    elif mode == "minmax":
        lo, hi = float(video_stack.min()), float(video_stack.max())
        if hi <= lo:
            hi = lo + 1.0
        video_stack = (video_stack - lo) / (hi - lo)
    elif mode in {None, "none"}:
        if video_stack.max() > 1.0:
            video_stack = video_stack / video_stack.max()
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")
    return video_stack.astype(np.float32)


def slice_video_into_spatial_patches(video_stack, patch_size=128, spatial_stride=None):
    """Split a (T, 1, H, W) stack into overlapping patch timelines."""
    _, _, H, W = video_stack.shape
    spatial_stride = patch_size if spatial_stride is None else spatial_stride
    if H < patch_size or W < patch_size:
        raise ValueError(f"Frame size {(H, W)} is smaller than patch_size={patch_size}")

    row_starts = list(range(0, H - patch_size + 1, spatial_stride))
    col_starts = list(range(0, W - patch_size + 1, spatial_stride))
    if row_starts[-1] != H - patch_size:
        row_starts.append(H - patch_size)
    if col_starts[-1] != W - patch_size:
        col_starts.append(W - patch_size)

    timelines = []
    for r in row_starts:
        for c in col_starts:
            timelines.append(video_stack[:, :, r:r + patch_size, c:c + patch_size])
    print(f"Spatial patch timelines: {len(timelines)} of size {patch_size}x{patch_size}")
    return timelines


def generate_sliding_window_clips(
    spatial_timelines,
    win_len=20,
    stride=5,
    min_std=0.01,
    min_mean=0.01,
    empty_patch_filter=True,
):
    """Window each patch timeline into clips shaped (win_len, 1, H, W)."""
    clips = []
    for timeline in spatial_timelines:
        T = timeline.shape[0]
        for t in range(0, T - win_len + 1, stride):
            clip = timeline[t:t + win_len]
            if empty_patch_filter and (clip.std() < min_std or clip.mean() < min_mean):
                continue
            clips.append(torch.from_numpy(clip.astype(np.float32)))
    return clips


def build_clips_from_sequence_dirs(sequence_dirs, cfg, split_name):
    all_clips = []
    for seq_dir in sequence_dirs:
        stack = load_raw_video_stack(seq_dir)
        stack = normalize_video_stack(
            stack,
            mode=cfg.get("normalize", "minmax"),
            percentiles=cfg.get("normalize_percentiles", (1, 99.8)),
        )
        timelines = slice_video_into_spatial_patches(
            stack,
            patch_size=cfg.get("patch_size", 128),
            spatial_stride=cfg.get("spatial_stride", cfg.get("patch_size", 128)),
        )
        clips = generate_sliding_window_clips(
            timelines,
            win_len=cfg.get("win_len", 20),
            stride=cfg.get("temporal_stride", 5),
            min_std=cfg.get("min_temporal_std", 0.01),
            min_mean=cfg.get("min_mean_intensity", 0.01),
            empty_patch_filter=cfg.get("empty_patch_filter", True),
        )
        print(f"{split_name} {seq_dir}: kept {len(clips)} clips")
        all_clips.extend(clips)
    if not all_clips:
        raise ValueError(f"No clips generated for {split_name}; relax empty-patch thresholds or check paths.")
    return torch.stack(all_clips).contiguous()


def tensor_range_text(name, tensor):
    return f"{name} min={tensor.min().item():.4f} max={tensor.max().item():.4f} mean={tensor.mean().item():.4f}"


def load_or_build_celltracking_tensors(cfg, force_rebuild=False):
    train_path = Path(cfg["preprocessed_train_tensor"])
    test_path = Path(cfg["preprocessed_test_tensor"])
    if train_path.exists() and test_path.exists() and not force_rebuild:
        print(f"Loading cached celltracking tensors:\n  {train_path}\n  {test_path}")
        train_tensor = torch.load(train_path, map_location="cpu")
        test_tensor = torch.load(test_path, map_location="cpu")
    else:
        if cfg.get("kind") != "ctc_tif_clips":
            raise ValueError(f"Cache build expects kind='ctc_tif_clips', got {cfg.get('kind')!r}")
        reason = "forced rebuild" if force_rebuild else "cache missing"
        print(f"Building celltracking tensors ({reason})...")
        train_tensor = build_clips_from_sequence_dirs(cfg["train_sequence_dirs"], cfg, "train")
        test_tensor = build_clips_from_sequence_dirs(cfg["test_sequence_dirs"], cfg, "test")

        max_train = cfg.get("max_train_clips")
        max_test = cfg.get("max_test_clips")
        if max_train is not None and train_tensor.shape[0] > max_train:
            train_tensor = train_tensor[:max_train]
        if max_test is not None and test_tensor.shape[0] > max_test:
            test_tensor = test_tensor[:max_test]

        train_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            "Preprocessed ranges: "
            f"{tensor_range_text('train', train_tensor)} | {tensor_range_text('test', test_tensor)}"
        )
        torch.save(train_tensor, train_path)
        torch.save(test_tensor, test_path)
        print(f"Saved train tensor {tuple(train_tensor.shape)} -> {train_path}")
        print(f"Saved test tensor  {tuple(test_tensor.shape)} -> {test_path}")
    return train_tensor.float().contiguous(), test_tensor.float().contiguous()


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encoder_cache_path(dataset_name, seed, train_clips, test_clips, latent_dim=None):
    latent_dim = LATENT_DIM_LOCAL if latent_dim is None else latent_dim
    cache_dir = Path("models") / dataset_name / "encoders"
    cache_dir.mkdir(parents=True, exist_ok=True)
    shape_tag = f"T{train_clips.shape[1]}_H{train_clips.shape[-2]}_W{train_clips.shape[-1]}"
    clip_tag = f"train{train_clips.shape[0]}_test{test_clips.shape[0]}"
    return cache_dir / f"encoder_seed{seed}_{shape_tag}_{clip_tag}_latent{latent_dim}.pt"


def load_or_train_shared_encoder(train_clips, test_clips, seed, dataset_name=None):
    dataset_name = CELLTRACKING_MODEL_NAME if dataset_name is None else dataset_name
    encoder = SpatialEncoder(latent_dim=LATENT_DIM_LOCAL)
    decoder = SpatialDecoder(latent_dim=LATENT_DIM_LOCAL, output_size=train_clips.shape[-2:])
    path = encoder_cache_path(dataset_name, seed, train_clips, test_clips)
    if path.exists():
        encoder.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded shared encoder: {path}")
    else:
        print(f"Training shared encoder: {path}")
        pretrain_spatial_encoder_on_clips(
            encoder,
            decoder,
            train_clips,
            epochs=AE_EPOCHS,
            frame_batch_size=256,
            max_frames_per_epoch=AE_MAX_FRAMES_PER_EPOCH,
        )
        torch.save(encoder.state_dict(), path)
        print(f"Saved shared encoder: {path}")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, path


@torch.no_grad()
def encode_clips(encoder, clips, frame_batch_size=256):
    frames = clips.reshape(-1, 1, clips.shape[-2], clips.shape[-1])
    zs = []
    for start in range(0, len(frames), frame_batch_size):
        zs.append(encoder(frames[start:start + frame_batch_size]).cpu())
    return torch.cat(zs, dim=0).reshape(clips.shape[0], clips.shape[1], -1)


def betti_curve(diagram, num_steps=None, min_v=0.0, max_v=1.0):
    num_steps = N_STEPS_LOCAL if num_steps is None else num_steps
    thresholds = np.linspace(min_v, max_v, num_steps, dtype=np.float32)
    diagram = np.asarray(diagram, dtype=np.float32)
    curve = np.zeros(num_steps, dtype=np.float32)
    if diagram.size == 0:
        return curve
    diagram = diagram.reshape(-1, 2)
    diagram[~np.isfinite(diagram)] = max_v
    alive = (diagram[:, 0][:, None] <= thresholds) & (diagram[:, 1][:, None] > thresholds)
    curve = alive.sum(axis=0).astype(np.float32)
    return np.clip(curve / BETTI_SCALE_LOCAL, 0.0, 1.0)


def tda_vector(frame, homology_dim):
    ph = cripser.compute_ph(frame.astype(np.float32), maxdim=1)
    diagram = ph[ph[:, 0] == homology_dim][:, 1:3]
    return betti_curve(diagram)


def compute_tda_features(clips, homology_dim=1):
    features = np.zeros((clips.shape[0], clips.shape[1], N_STEPS_LOCAL), dtype=np.float32)
    arr = clips[:, :, 0].detach().cpu().numpy()
    for n in range(arr.shape[0]):
        if n == 0 or (n + 1) % 8 == 0 or n == arr.shape[0] - 1:
            print(f"Computing H{homology_dim} TDA clip {n + 1:03d}/{arr.shape[0]:03d}")
        for t in range(arr.shape[1]):
            features[n, t] = tda_vector(arr[n, t], homology_dim)
    return torch.from_numpy(features).float()


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim=None, hidden_dim=128):
        super().__init__()
        output_dim = LATENT_DIM_LOCAL if output_dim is None else output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_features(z, b_h1=None, b_h0=None, input_variant="z", predict_steps_ahead=None):
    predict_steps_ahead = PREDICT_STEPS_AHEAD if predict_steps_ahead is None else predict_steps_ahead
    if predict_steps_ahead < 1 or predict_steps_ahead >= z.shape[1]:
        raise ValueError(f"PREDICT_STEPS_AHEAD must be in [1, T-1], got {predict_steps_ahead} for T={z.shape[1]}")
    z_now = z[:, :-predict_steps_ahead]
    z_future = z[:, predict_steps_ahead:]
    z_delta = z_future - z_now

    if input_variant == "z":
        x = z_now
    elif input_variant == "z_h1":
        x = torch.cat([z_now, b_h1[:, :-predict_steps_ahead]], dim=-1)
    elif input_variant == "z_h0":
        x = torch.cat([z_now, b_h0[:, :-predict_steps_ahead]], dim=-1)
    elif input_variant == "z_h0_h1":
        x = torch.cat([z_now, b_h0[:, :-predict_steps_ahead], b_h1[:, :-predict_steps_ahead]], dim=-1)
    else:
        raise ValueError(f"Unknown input_variant: {input_variant}")

    return {
        "x": x.reshape(-1, x.shape[-1]),
        "z_now": z_now.reshape(-1, z_now.shape[-1]),
        "z_future": z_future.reshape(-1, z_future.shape[-1]),
        "z_delta": z_delta.reshape(-1, z_delta.shape[-1]),
    }


def make_supervised(z, b_h1=None, b_h0=None, input_variant="z", task="next", predict_steps_ahead=None):
    data = make_features(
        z,
        b_h1=b_h1,
        b_h0=b_h0,
        input_variant=input_variant,
        predict_steps_ahead=predict_steps_ahead,
    )
    target = data["z_future"] if task == "next" else data["z_delta"]
    return data["x"], target, data["z_now"], data["z_future"]


def train_mlp_variant(name, train_data, test_data, seed, task):
    set_all_seeds(seed)
    x_train, y_train, _, _ = train_data
    x_test, _, z_now_test, z_future_test = test_data
    model = MLPRegressor(input_dim=x_train.shape[1], output_dim=y_train.shape[1])
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(ds, batch_size=MLP_BATCH_SIZE, shuffle=True)
    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            seen += len(xb)
        if epoch == 1 or epoch == MLP_EPOCHS:
            print(f"{name} epoch {epoch}/{MLP_EPOCHS} train target MSE {total / seen:.6f}")
    model.eval()
    with torch.no_grad():
        pred_target = model(x_test)
        pred_future = z_now_test + pred_target if task == "delta" else pred_target
        future_z_mse = loss_fn(pred_future, z_future_test).item()
        target_mse = loss_fn(pred_target, z_future_test - z_now_test).item() if task == "delta" else loss_fn(pred_target, z_future_test).item()
    return future_z_mse, target_mse


def _latent_betti_curve_from_diag(diag, n_bins=16):
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


def compute_latent_window_tda_features(z_features, window=20, n_bins=16):
    """Compute H0/H1 on each clip's recent latent trajectory window."""
    N, T, _ = z_features.shape
    h0 = np.zeros((N, T, n_bins), dtype=np.float32)
    h1 = np.zeros((N, T, n_bins), dtype=np.float32)
    z_np = z_features.detach().cpu().numpy().astype(np.float32)

    for n in range(N):
        if n == 0 or (n + 1) % 8 == 0 or n == N - 1:
            print(f"Computing latent-window TDA clip {n + 1:03d}/{N:03d}")
        for t in range(T):
            start = max(0, t - window + 1)
            pts = z_np[n, start:t + 1, :]
            if len(pts) < 2:
                continue
            pts = pts - pts.mean(axis=0, keepdims=True)
            scale = pts.std(axis=0, keepdims=True).mean() + 1e-6
            pts = pts / scale
            diffs = pts[:, None, :] - pts[None, :, :]
            dmat = np.sqrt(np.sum(diffs * diffs, axis=-1)).astype(np.float32)
            dgms = ripser(dmat, maxdim=1, distance_matrix=True)["dgms"]
            h0[n, t] = _latent_betti_curve_from_diag(dgms[0], n_bins=n_bins)
            h1[n, t] = _latent_betti_curve_from_diag(dgms[1], n_bins=n_bins) if len(dgms) > 1 else 0.0
    return torch.from_numpy(h0), torch.from_numpy(h1)


def latent_tda_cache_path(dataset_name, seed, split_name, z_features, window=20, n_bins=16):
    cache_dir = Path("models") / dataset_name / "latent_tda_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_{split_name}_N{z_features.shape[0]}_T{z_features.shape[1]}_"
        f"latent{z_features.shape[2]}_win{window}_bins{n_bins}"
    )
    return cache_dir / f"{tag}.pt"


def load_or_compute_latent_tda_features(
    dataset_name,
    seed,
    split_name,
    z_features,
    window=20,
    n_bins=16,
    recompute=False,
):
    cache_path = latent_tda_cache_path(dataset_name, seed, split_name, z_features, window=window, n_bins=n_bins)
    if cache_path.exists() and not recompute:
        print(f"Loading latent TDA cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f"Computing z + latent-window TDA for {split_name}...")
    h0, h1 = compute_latent_window_tda_features(z_features, window=window, n_bins=n_bins)
    payload = {"z": z_features.detach().cpu(), "h0": h0, "h1": h1}
    torch.save(payload, cache_path)
    print(f"Saved latent TDA cache: {cache_path}")
    return payload


def features_for_latent_tda_mode(payload, mode):
    z = payload["z"]
    if mode == "z":
        return z
    if mode == "z_latent_h0":
        return torch.cat([z, payload["h0"]], dim=-1)
    if mode == "z_latent_h1":
        return torch.cat([z, payload["h1"]], dim=-1)
    if mode == "z_latent_both":
        return torch.cat([z, payload["h0"], payload["h1"]], dim=-1)
    raise ValueError(f"Unknown latent TDA mode: {mode}")


def make_latent_tda_supervised(features, z, task="next", predict_steps_ahead=None):
    predict_steps_ahead = PREDICT_STEPS_AHEAD if predict_steps_ahead is None else predict_steps_ahead
    if predict_steps_ahead < 1 or predict_steps_ahead >= z.shape[1]:
        raise ValueError(f"PREDICT_STEPS_AHEAD must be in [1, T-1], got {predict_steps_ahead} for T={z.shape[1]}")
    x = features[:, :-predict_steps_ahead]
    z_now = z[:, :-predict_steps_ahead]
    z_future = z[:, predict_steps_ahead:]
    target = z_future if task == "next" else z_future - z_now
    return (
        x.reshape(-1, x.shape[-1]),
        target.reshape(-1, target.shape[-1]),
        z_now.reshape(-1, z_now.shape[-1]),
        z_future.reshape(-1, z_future.shape[-1]),
    )
