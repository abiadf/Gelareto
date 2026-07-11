"""Core video forecasting helpers: encoders, predictors, TDA features, and datasets."""

from pathlib import Path
import random

import cv2
import cripser
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from PIL import Image
except ImportError:  # pragma: no cover - notebook dependency fallback
    Image = None

DATASET = "default"
EXPERIMENT_SEED = "default"
TDA_MODE = "none"
LATENT_DIM = 128
REAL_TDA_SCALE = 15
REAL_TDA_BINS = 25
HORIZON = 1
PREDICTOR_EPOCHS = 10
DEVICE = "auto"
criterion = nn.MSELoss()


def configure_runtime(**kwargs):
    """Update runtime settings shared by the terminal runner and notebook helpers."""
    globals().update(kwargs)


def tensor_to_model_float(x):
    """Convert image tensors to float [0, 1] only when needed."""
    if x.dtype == torch.uint8:
        return x.float() / 255.0
    return x.float()


def get_runtime_device():
    """Resolve configured compute device with CPU fallback."""
    requested = globals().get("DEVICE", "auto")
    if isinstance(requested, torch.device):
        return requested
    requested = str(requested).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        print("Requested MPS but it is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)

class SpatialEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2), 
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) 
        )
        self.fc = nn.Linear(32 * 4 * 4, latent_dim)

    def forward(self, x):
        features = self.conv_layers(x)
        features = torch.flatten(features, start_dim=1)
        return self.fc(features)

class SpatialDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_size=(64, 64)):
        super().__init__()
        self.output_size = tuple(output_size)
        output_pixels = self.output_size[0] * self.output_size[1]
        # Direct projection from latent space back to a flattened image grid.
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_pixels),
            nn.Sigmoid()) # Keeps pixel outputs bounded between [0, 1]
        
    def forward(self, x):
        x = self.fc(x)
        # Reshape cleanly back to matching frame format (B, 1, H, W)
        return x.reshape(-1, 1, self.output_size[0], self.output_size[1])

class TopologicalPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm      = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=False)
        self.predictor = nn.Linear(hidden_dim, LATENT_DIM) 

    def forward(self, x):
        lstm_out, _ = self.lstm(x) 
        return self.predictor(lstm_out)

def diagrams_to_betti_curves(batch_diagrams, num_steps=25, min_v=0.0, max_v=1.0):
    thresholds = np.linspace(min_v, max_v, num_steps, dtype=np.float32)
    curves     = np.zeros((len(batch_diagrams), num_steps), dtype=np.float32)
    for i, diag in enumerate(batch_diagrams):
        diag = np.asarray(diag, dtype=np.float32)
        if len(diag) == 0:
            continue
        alive     = (diag[:, 0][:, None] <= thresholds) & (diag[:, 1][:, None] > thresholds)
        curves[i] = alive.sum(axis=0)
    curves = np.clip(curves / REAL_TDA_SCALE, 0.0, 1.0)
    return torch.from_numpy(curves).float()

def select_tda_features(h0_diags, h1_diags):
    betti_h0 = diagrams_to_betti_curves(h0_diags, num_steps=REAL_TDA_BINS)
    betti_h1 = diagrams_to_betti_curves(h1_diags, num_steps=REAL_TDA_BINS)
    base_mode, _ = parse_tda_control_mode(TDA_MODE)
    if base_mode == "h0":
        return betti_h0
    if base_mode == "h1":
        return betti_h1
    return torch.cat([betti_h0, betti_h1], dim=1)


def parse_tda_control_mode(mode):
    """Split modes like h0_shuffle into base TDA and control names."""
    parts = mode.split("_", 1)
    base = parts[0]
    control = parts[1] if len(parts) == 2 else "real"
    return base, control


def perturb_tda_noise(tda_features, seed=0):
    """Replace TDA with random features of matching shape and scale."""
    generator = torch.Generator(device=tda_features.device).manual_seed(seed)
    noise = torch.randn(tda_features.shape, generator=generator, device=tda_features.device)
    mean = tda_features.mean()
    std = tda_features.std().clamp_min(1e-6)
    return noise * std + mean


def perturb_tda_shuffle(tda_features, seed=0):
    """Attach real TDA vectors to the wrong time/sample positions."""
    T, B, D = tda_features.shape
    flat = tda_features.reshape(T * B, D)
    generator = torch.Generator(device=tda_features.device).manual_seed(seed)
    perm = torch.randperm(T * B, generator=generator, device=tda_features.device)
    return flat[perm].reshape(T, B, D)


def perturb_tda_shift(tda_features, shift=1):
    """Shift TDA in time while keeping each sequence/sample intact."""
    if tda_features.shape[0] <= 1:
        return tda_features.clone()
    return torch.roll(tda_features, shifts=shift, dims=0)


def apply_tda_control(tda_features, control="real", seed=0, shift=1):
    """Apply a named control perturbation to TDA features."""
    if control == "real":
        return tda_features
    if control == "zero":
        return torch.zeros_like(tda_features)
    if control == "noise":
        return perturb_tda_noise(tda_features, seed=seed)
    if control == "shuffle":
        return perturb_tda_shuffle(tda_features, seed=seed)
    if control == "shift":
        return perturb_tda_shift(tda_features, shift=shift)
    raise ValueError(f"Unknown TDA control mode: {control}")


def build_controlled_tda_features(z_seq, h0, h1, mode, seed=0, shift=1):
    """Concatenate z with real or perturbed H0/H1 features."""
    base, control = parse_tda_control_mode(mode)
    if base == "none":
        return z_seq
    if base == "h0":
        tda = h0
    elif base == "h1":
        tda = h1
    elif base == "both":
        tda = torch.cat([h0, h1], dim=-1)
    else:
        raise ValueError(f"Unknown TDA base mode: {base}")
    return torch.cat([z_seq, apply_tda_control(tda, control=control, seed=seed, shift=shift)], dim=-1)


def run_cripser_tda_on_current_frames(video_prefix):
    """Compute H0/H1 with cripser for the current frame of each clip."""
    current = video_prefix[-1, :, 0].detach().cpu()
    if current.dtype == torch.uint8:
        current_frames = current.numpy().astype(np.float32) / 255.0
    else:
        current_frames = current.numpy().astype(np.float32)
    h0_diagrams, h1_diagrams = [], []
    for frame in current_frames:
        ph = cripser.compute_ph(frame.astype(np.float32), maxdim=1)
        h0 = ph[ph[:, 0] == 0][:, 1:3]
        h1 = ph[ph[:, 0] == 1][:, 1:3]
        h0[~np.isfinite(h0)] = 1.0
        h1[~np.isfinite(h1)] = 1.0
        if h0.size == 0: h0 = np.zeros((0, 2), dtype=np.float32)
        if h1.size == 0: h1 = np.zeros((0, 2), dtype=np.float32)
        h0_diagrams.append(h0)
        h1_diagrams.append(h1)
    return h0_diagrams, h1_diagrams

def pretrain_spatial_encoder(encoder, decoder, X_train, ae_epochs=3):
    print("\n--- Phase 1: Pre-training Spatial Encoder ---")
    device = get_runtime_device()
    encoder.to(device)
    decoder.to(device)
    print(f"AE device: {device}")
    ae_optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    all_frames = X_train.reshape(-1, *X_train.shape[2:])
    frame_batch_size = globals().get("AE_FRAME_BATCH_SIZE", 256)
    max_frames_per_epoch = globals().get("AE_MAX_FRAMES_PER_EPOCH", min(len(all_frames), 8192))
    for ae_epoch in range(ae_epochs):
        encoder.train()
        decoder.train()
        generator = torch.Generator().manual_seed(1000 + ae_epoch)
        if max_frames_per_epoch is None or max_frames_per_epoch >= len(all_frames):
            frame_idx = torch.randperm(len(all_frames), generator=generator)
        else:
            frame_idx = torch.randperm(len(all_frames), generator=generator)[:max_frames_per_epoch]
        total_loss, total_seen = 0.0, 0
        for start in range(0, len(frame_idx), frame_batch_size):
            idx = frame_idx[start:start + frame_batch_size]
            frames = tensor_to_model_float(all_frames[idx]).to(device, non_blocking=True)
            ae_optimizer.zero_grad()
            latents = encoder(frames)
            reconstructions = decoder(latents)
            ae_loss = criterion(reconstructions, frames)
            ae_loss.backward()
            ae_optimizer.step()
            total_loss += ae_loss.item() * len(frames)
            total_seen += len(frames)
        print(f"AE Pretrain Epoch {ae_epoch+1}/{ae_epochs} | Reconstr MSE: {total_loss / total_seen:.4f}")

def build_real_tda_features(video_tensor, encoder, use_tda, split_name="Train"):
    device = get_runtime_device()
    encoder.to(device)
    encoder.eval()
    z_history = []
    tda_history = []
    with torch.no_grad():
        for t in range(video_tensor.shape[0]):
            frame_t = tensor_to_model_float(video_tensor[t]).to(device, non_blocking=True)
            z_t     = encoder(frame_t)
            if use_tda:
                if (t + 1) % 20 == 0 or t == 0 or t == video_tensor.shape[0] - 1:
                    print(f"  [{split_name} Frame {t+1:02d}/{video_tensor.shape[0]}] Computing TDA")
                h0_diags, h1_diags = run_cripser_tda_on_current_frames(video_tensor[0:t+1])
                B_t = select_tda_features(h0_diags, h1_diags).to(device)
                tda_history.append(B_t)
            z_history.append(z_t)
    z_seq = torch.stack(z_history, dim=0)
    if not use_tda:
        return z_seq, z_seq

    tda_seq = torch.stack(tda_history, dim=0)
    _, control = parse_tda_control_mode(TDA_MODE)
    control_seed = globals().get("EXPERIMENT_SEED", 0)
    if isinstance(control_seed, str):
        control_seed = abs(hash(control_seed)) % (2**31)
    tda_seq = apply_tda_control(tda_seq, control=control, seed=int(control_seed), shift=1)
    return torch.cat([z_seq, tda_seq], dim=2), z_seq

def train_predictor(model, encoder, X_train, predictor_epochs, use_tda, learning_rate):
    print(f"\n--- Phase 2: Training LSTM Predictor (USE_TDA={use_tda}) ---")
    device = get_runtime_device()
    encoder.to(device)
    model.to(device)
    print(f"Predictor device: {device}")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(predictor_epochs):
        model.train()
        optimizer.zero_grad()
        print(f"\n--- Epoch {epoch+1:02d}/{predictor_epochs:02d} ---")
        total_features, z_features = build_real_tda_features(X_train, encoder, use_tda, split_name="Train")
        horizon = HORIZON
        predictions = model(total_features[:-horizon])
        targets = z_features[horizon:]
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == predictor_epochs - 1:
            print(f"Epoch {epoch+1:02d}/{predictor_epochs:02d} | Training MSE Loss: {loss.item():.4f}")

def load_or_train_model(encoder, decoder, model, X_train, retrain_encoder, retrain_predictor, use_tda, learning_rate):
    device = get_runtime_device()
    model_dir   = Path("models") / DATASET
    model_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir = model_dir / "encoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)

    # The encoder is shared across TDA/no-TDA variants for the same dataset/config.
    # The predictor remains variant-specific because its input dimension changes.
    seed_tag     = globals().get("EXPERIMENT_SEED", "default")
    encoder_tag  = f"seed{seed_tag}_T{X_train.shape[0]}_B{X_train.shape[1]}_H{X_train.shape[-2]}_W{X_train.shape[-1]}_latent{LATENT_DIM}"
    predict_tag = f"pred{HORIZON}"
    model_suffix = (
        f"seed{seed_tag}_{predict_tag}_real_tda_{TDA_MODE}_realtdascale{REAL_TDA_SCALE}"
        if use_tda
        else f"seed{seed_tag}_{predict_tag}_real_tda_none"
    )
    encoder_path = encoder_dir / f"encoder_{encoder_tag}.pt"
    model_path   = model_dir / f"model_{model_suffix}.pt"

    if encoder_path.exists() and not retrain_encoder:
        print(f"Loading shared encoder: {encoder_path}")
        encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    else:
        reason = "retraining" if encoder_path.exists() else "missing; training once"
        print(f"Shared encoder {reason}: {encoder_path}")
        pretrain_spatial_encoder(encoder, decoder, X_train, ae_epochs=3)
        torch.save(encoder.state_dict(), encoder_path)
        print(f"Saved shared encoder: {encoder_path}")
    encoder.to(device)

    for param in encoder.parameters():
        param.requires_grad = False

    if model_path.exists() and not retrain_predictor:
        print(f"Loading predictor: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.to(device)
    else:
        reason = "retraining" if model_path.exists() else "missing; training once"
        print(f"Predictor {reason}: {model_path}")
        model.to(device)
        train_predictor(model, encoder, X_train, predictor_epochs=PREDICTOR_EPOCHS, use_tda=use_tda, learning_rate=learning_rate)
        torch.save(model.state_dict(), model_path)
        print(f"Saved predictor weights to {model_path}")

def test_predictor(model, encoder, X_test, use_tda):
    import matplotlib.pyplot as plt

    print(f"\nTesting on Test Set (USE_TDA = {use_tda})...")
    device = get_runtime_device()
    encoder.to(device)
    model.to(device)
    model.eval()
    total_test_features, test_z_features = build_real_tda_features(X_test, encoder, use_tda, split_name="Test")
    with torch.no_grad():
        horizon = HORIZON
        test_predictions = model(total_test_features[:-horizon])
        test_targets = test_z_features[horizon:]
        test_loss = criterion(test_predictions, test_targets)
        per_frame_mse = ((test_predictions - test_targets) ** 2).mean(dim=(1, 2))
    print(f"Final test MSE Loss (USE_TDA={use_tda}): {test_loss.item():.6f}")
    print("Per-frame MSE:", per_frame_mse.detach().cpu().numpy())
    curve_label = f"{DATASET} seed={globals().get('EXPERIMENT_SEED', '?')} mode={globals().get('TDA_MODE', 'none')}"
    plt.plot(per_frame_mse.detach().cpu().numpy(), marker='o', label=curve_label)
    plt.xlabel("prediction index")
    plt.ylabel("MSE")
    plt.legend(fontsize="x-small")
    return test_loss.item(), per_frame_mse.detach().cpu()

def prepare_clip_tensor_for_sequence_pipeline(clips, max_clips=None, seed=42):
    """Convert saved clips from (N, T, C, H, W) into normalized float tensors."""
    clips = clips.detach().cpu().float()
    if clips.ndim != 5:
        raise ValueError(f"Expected clips with shape (N, T, C, H, W), got {tuple(clips.shape)}")
    if clips.shape[2] != 1:
        raise ValueError(f"Expected single-channel clips at dim 2, got shape {tuple(clips.shape)}")
    if clips.max() > 1.0:
        clips = clips / 255.0
    if max_clips is not None and clips.shape[0] > max_clips:
        generator = torch.Generator().manual_seed(seed)
        idx = torch.randperm(clips.shape[0], generator=generator)[:max_clips]
        clips = clips[idx]
    return clips.contiguous()

def pretrain_spatial_encoder_on_clips(
    encoder,
    decoder,
    train_clips,
    ae_epochs=1,
    frame_batch_size=256,
    max_frames_per_epoch=8192,
    learning_rate=1e-3,
):
    print("\n--- Phase 1: Mini-batch pre-training Spatial Encoder ---")
    encoder.train()
    decoder.train()
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    all_frames = train_clips.reshape(-1, 1, train_clips.shape[-2], train_clips.shape[-1])
    n_frames = all_frames.shape[0]
    for epoch in range(1, ae_epochs + 1):
        generator = torch.Generator().manual_seed(1000 + epoch)
        if max_frames_per_epoch is None or max_frames_per_epoch >= n_frames:
            frame_idx = torch.randperm(n_frames, generator=generator)
        else:
            frame_idx = torch.randperm(n_frames, generator=generator)[:max_frames_per_epoch]
        total_loss = 0.0
        total_seen = 0
        for start in range(0, len(frame_idx), frame_batch_size):
            idx = frame_idx[start:start + frame_batch_size]
            frames = all_frames[idx]
            optimizer.zero_grad()
            latents = encoder(frames)
            reconstructions = decoder(latents)
            loss = criterion(reconstructions, frames)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(frames)
            total_seen += len(frames)
        print(f"AE Pretrain Epoch {epoch}/{ae_epochs} | Reconstr MSE: {total_loss / total_seen:.4f}")

def normalize_video_array(arr):
    arr = arr.astype(np.float32)
    arr -= arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr /= max_val
    return arr

def resize_video_array(video, image_size=(64, 64)):
    if video.shape[-2:] == image_size:
        return video.astype(np.float32)
    T, B = video.shape[:2]
    resized = np.empty((T, B, image_size[0], image_size[1]), dtype=np.float32)
    for t in range(T):
        for b in range(B):
            resized[t, b] = cv2.resize(video[t, b], image_size[::-1], interpolation=cv2.INTER_AREA)
    return resized

def read_tif_frame(path):
    try:
        return tiff.imread(str(path)).astype(np.float32)
    except ValueError as exc:
        if "imagecodecs" not in str(exc) and "COMPRESSION" not in str(exc):
            raise
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is not None:
            return frame.astype(np.float32)
        if Image is None:
            raise ImportError("Install Pillow or imagecodecs to read this TIFF compression")
        return np.asarray(Image.open(path), dtype=np.float32)

def load_tif_sequence(folder):
    folder = Path(folder)
    files = sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))
    if not files:
        raise FileNotFoundError(f"No TIFF frames found in {folder}")
    return np.stack([read_tif_frame(f) for f in files], axis=0)

def load_image_sequence(folder):
    folder = Path(folder)
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    files = sorted([f for pattern in patterns for f in folder.glob(pattern)])
    if not files:
        raise FileNotFoundError(f"No image frames found in {folder}")
    frames = []
    for f in files:
        frame = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            if Image is None:
                raise ValueError(f"Could not read image frame: {f}")
            frame = np.asarray(Image.open(f).convert("L"), dtype=np.float32)
        frame = frame.astype(np.float32)
        if frame.max() > 1.0:
            frame /= 255.0
        frames.append(frame)
    return np.stack(frames, axis=0)

class DavisWindowLoader:
    def __init__(self, config):
        self.root = Path(config["root"])
        self.seq_len = config.get("seq_len", 20)
        self.stride = config.get("stride", 5)
        self.image_size = config.get("image_size", (64, 64))
        self.train_ratio = config.get("train_ratio", 0.7)
        self.seed = config.get("seed", 42)
        self.max_train = config.get("max_train_windows")
        self.max_test = config.get("max_test_windows")

    def load(self):
        videos = sorted([p for p in self.root.iterdir() if p.is_dir()])
        random.Random(self.seed).shuffle(videos)
        n_train = int(self.train_ratio * len(videos))
        train_videos, test_videos = videos[:n_train], videos[n_train:]
        train_windows = self._make_windows(train_videos)[:self.max_train]
        test_windows = self._make_windows(test_videos)[:self.max_test]
        print(f"DAVIS videos: {len(train_videos)} train | {len(test_videos)} test")
        print(f"DAVIS windows: {len(train_windows)} train | {len(test_windows)} test")
        return self._load_batch(train_windows), self._load_batch(test_windows)

    def _make_windows(self, videos):
        windows = []
        for video in videos:
            n_frames = len(sorted(video.glob("*")))
            for start in range(0, max(0, n_frames - self.seq_len + 1), self.stride):
                windows.append((video, start, start + self.seq_len))
        return windows

    def _load_batch(self, windows):
        clips = []
        for video, start, end in windows:
            seq = load_image_sequence(video)
            clip = resize_video_array(seq[start:end, None, :, :], self.image_size)[:, 0]
            clips.append(clip)
        if not clips:
            raise ValueError("No DAVIS windows found for the requested seq_len/stride")
        return np.stack(clips, axis=1)

def load_davis_images(config):
    return DavisWindowLoader(config).load()

def summarize_metric_runs(results_df, group_cols, metric_cols, sort_metric=None):
    """Mean/std summary plus one n_runs column for the grouped experiment count."""
    grouped = results_df.groupby(group_cols)
    summary = grouped[metric_cols].agg(["mean", lambda x: x.std(ddof=0)])
    summary = summary.rename(columns={"<lambda_0>": "std"}, level=1)
    summary.insert(0, ("n_runs", ""), grouped.size())
    if sort_metric is None:
        sort_metric = metric_cols[0]
    return summary.sort_values((sort_metric, "mean"))

class LorenzMovingShapesDataset(torch.utils.data.Dataset):
    """Synthetic disk/ring clips with Lorenz-driven irregular motion.

    Each item is a tensor with shape (T, 1, H, W), values in [0, 1].
    Seeds are index-derived, so train/test splits stay reproducible and separate.
    """

    def __init__(
        self,
        num_clips=2000,
        clip_len=20,
        image_size=(96, 96),
        base_radius=6,
        min_balls=2,
        max_balls=6,
        seed_offset=0,
        lorenz_dt=0.015,
        radius_pulse_amp=0.25,
        radius_pulse_freq=0.15,
        thickness_min=2,
        thickness_max=2,
        overlap_strength=0.0,
        shape="ring",
        output_dtype="float32",
    ):
        if min_balls < 1 or max_balls < min_balls:
            raise ValueError("Require 1 <= min_balls <= max_balls")
        if not 0.0 <= overlap_strength <= 1.0:
            raise ValueError("Require 0 <= overlap_strength <= 1")
        if min(image_size) <= 2 * (base_radius + 4):
            raise ValueError("image_size is too small for the requested ball radius")
        if shape not in {"ring", "disk"}:
            raise ValueError("shape must be 'ring' or 'disk'")

        self.num_clips = int(num_clips)
        self.clip_len = int(clip_len)
        self.image_size = tuple(image_size)
        self.base_radius = int(base_radius)
        self.min_balls = int(min_balls)
        self.max_balls = int(max_balls)
        self.seed_offset = int(seed_offset)
        self.lorenz_dt = float(lorenz_dt)
        self.radius_pulse_amp = float(radius_pulse_amp)
        self.radius_pulse_freq = float(radius_pulse_freq)
        self.thickness_min = int(thickness_min)
        self.thickness_max = int(thickness_max)
        self.overlap_strength = float(overlap_strength)
        self.shape = str(shape)
        self.output_dtype = output_dtype

    def __len__(self):
        return self.num_clips

    def __getitem__(self, index):
        rng = np.random.RandomState(self.seed_offset + int(index) * 1009)
        num_balls = rng.randint(self.min_balls, self.max_balls + 1)
        clip = self._render_clip(
            num_balls=num_balls,
            num_frames=self.clip_len,
            image_size=self.image_size,
            base_radius=self.base_radius,
            lorenz_dt=self.lorenz_dt,
            radius_pulse_amp=self.radius_pulse_amp,
            radius_pulse_freq=self.radius_pulse_freq,
            thickness_min=self.thickness_min,
            thickness_max=self.thickness_max,
            overlap_strength=self.overlap_strength,
            shape=self.shape,
            rng=rng,
        )
        if self.output_dtype == "uint8":
            clip = np.clip(np.rint(clip * 255.0), 0, 255).astype(np.uint8)
        return torch.from_numpy(clip)

    @staticmethod
    def _lorenz_xy(num_frames, rng, dt):
        sigma, beta, rho = 10.0, 8.0 / 3.0, 28.0
        x = rng.uniform(-10.0, 10.0)
        y = rng.uniform(-10.0, 10.0)
        z = rng.uniform(10.0, 30.0)
        traj = np.empty((num_frames, 2), dtype=np.float32)

        for t in range(num_frames):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            x, y, z = x + dx, y + dy, z + dz
            traj[t] = (x, y)
        return traj

    @staticmethod
    def _scale_to_canvas(values, low, high):
        span = float(values.max() - values.min())
        if span <= 1e-6:
            return np.full_like(values, (low + high) / 2.0, dtype=np.float32)
        return low + (values - values.min()) * ((high - low) / span)

    @classmethod
    def _ball_track(cls, num_frames, image_size, base_radius, lorenz_dt, overlap_strength, rng):
        H, W = image_size
        rx = max(2, int(round(base_radius * rng.uniform(0.65, 1.35))))
        ry = max(2, int(round(base_radius * rng.uniform(0.65, 1.35))))
        pad = max(rx, ry) + 3

        traj = cls._lorenz_xy(num_frames, rng=rng, dt=lorenz_dt)
        cx = cls._scale_to_canvas(traj[:, 0], pad, W - pad - 1).astype(np.int32)
        cy = cls._scale_to_canvas(traj[:, 1], pad, H - pad - 1).astype(np.int32)
        if overlap_strength > 0:
            center_x = (W - 1) / 2.0 + rng.uniform(-0.08, 0.08) * W
            center_y = (H - 1) / 2.0 + rng.uniform(-0.08, 0.08) * H
            cx = ((1.0 - overlap_strength) * cx + overlap_strength * center_x).astype(np.int32)
            cy = ((1.0 - overlap_strength) * cy + overlap_strength * center_y).astype(np.int32)
            cx = np.clip(cx, pad, W - pad - 1)
            cy = np.clip(cy, pad, H - pad - 1)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        return cx, cy, rx, ry, phase

    @classmethod
    def _render_clip(
        cls,
        num_balls,
        num_frames,
        image_size,
        base_radius,
        lorenz_dt,
        radius_pulse_amp,
        radius_pulse_freq,
        thickness_min,
        thickness_max,
        overlap_strength,
        shape,
        rng,
    ):
        H, W = image_size
        video = np.zeros((num_frames, H, W), dtype=np.uint8)
        tracks = [
            cls._ball_track(num_frames, image_size, base_radius, lorenz_dt, overlap_strength, rng)
            for _ in range(num_balls)
        ]

        for t in range(num_frames):
            frame = np.zeros((H, W), dtype=np.uint8)
            for ball_idx, (cx, cy, rx, ry, phase) in enumerate(tracks):
                pulse = 1.0 + radius_pulse_amp * np.sin(radius_pulse_freq * t + phase)
                pulse_x = max(2, int(round(rx * pulse)))
                pulse_y = max(2, int(round(ry * (1.0 + radius_pulse_amp * np.cos(radius_pulse_freq * t + phase)))))
                if thickness_max > thickness_min:
                    thickness_phase = 0.5 + 0.5 * np.sin(radius_pulse_freq * t + phase + np.pi / 3.0)
                    thickness = int(round(thickness_min + thickness_phase * (thickness_max - thickness_min)))
                else:
                    thickness = thickness_min
                thickness = max(1, min(thickness, pulse_x, pulse_y))
                angle = np.deg2rad(t * (1.0 + 0.5 * ball_idx))
                if shape == "disk":
                    cls._draw_filled_ellipse(frame, int(cx[t]), int(cy[t]), pulse_x, pulse_y, angle)
                else:
                    cls._draw_hollow_ellipse(frame, int(cx[t]), int(cy[t]), pulse_x, pulse_y, angle, thickness=thickness)
            video[t] = frame

        return video[:, None].astype(np.float32) / 255.0

    @staticmethod
    def _draw_filled_ellipse(frame, cx, cy, rx, ry, angle):
        H, W = frame.shape
        pad = max(rx, ry) + 1
        y0, y1 = max(0, cy - pad), min(H, cy + pad + 1)
        x0, x1 = max(0, cx - pad), min(W, cx + pad + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]

        x = xx - cx
        y = yy - cy
        ca, sa = np.cos(angle), np.sin(angle)
        xr = ca * x + sa * y
        yr = -sa * x + ca * y

        filled = (xr / max(rx, 1)) ** 2 + (yr / max(ry, 1)) ** 2 <= 1.0
        frame[y0:y1, x0:x1][filled] = 255

    @staticmethod
    def _draw_hollow_ellipse(frame, cx, cy, rx, ry, angle, thickness=2):
        H, W = frame.shape
        pad = max(rx, ry) + thickness + 1
        y0, y1 = max(0, cy - pad), min(H, cy + pad + 1)
        x0, x1 = max(0, cx - pad), min(W, cx + pad + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]

        x = xx - cx
        y = yy - cy
        ca, sa = np.cos(angle), np.sin(angle)
        xr = ca * x + sa * y
        yr = -sa * x + ca * y

        outer = (xr / max(rx, 1)) ** 2 + (yr / max(ry, 1)) ** 2 <= 1.0
        inner_rx = max(rx - thickness, 1)
        inner_ry = max(ry - thickness, 1)
        inner = (xr / inner_rx) ** 2 + (yr / inner_ry) ** 2 <= 1.0
        frame[y0:y1, x0:x1][outer & ~inner] = 255

    def make_tensor(self):
        print(f"Generating {self.num_clips} synthetic Lorenz {self.shape} clips...")
        data = torch.stack([self[i] for i in range(len(self))])
        print(f"Dataset generated. Final shape: {tuple(data.shape)}")
        return data

class OrbitingShapesDataset(torch.utils.data.Dataset):
    """Synthetic disk/ring clips with periodic circular/elliptical motion."""

    def __init__(
        self,
        num_clips=512,
        clip_len=24,
        image_size=(96, 96),
        base_radius=6,
        min_shapes=2,
        max_shapes=4,
        seed_offset=0,
        orbit_radius_min=14,
        orbit_radius_max=32,
        angular_speed_min=0.18,
        angular_speed_max=0.42,
        thickness_min=2,
        thickness_max=4,
        shape="ring",
        output_dtype="float32",
    ):
        if min_shapes < 1 or max_shapes < min_shapes:
            raise ValueError("Require 1 <= min_shapes <= max_shapes")
        if shape not in {"ring", "disk"}:
            raise ValueError("shape must be 'ring' or 'disk'")
        self.num_clips = int(num_clips)
        self.clip_len = int(clip_len)
        self.image_size = tuple(image_size)
        self.base_radius = int(base_radius)
        self.min_shapes = int(min_shapes)
        self.max_shapes = int(max_shapes)
        self.seed_offset = int(seed_offset)
        self.orbit_radius_min = float(orbit_radius_min)
        self.orbit_radius_max = float(orbit_radius_max)
        self.angular_speed_min = float(angular_speed_min)
        self.angular_speed_max = float(angular_speed_max)
        self.thickness_min = int(thickness_min)
        self.thickness_max = int(thickness_max)
        self.shape = str(shape)
        self.output_dtype = output_dtype

    def __len__(self):
        return self.num_clips

    def __getitem__(self, index):
        rng = np.random.RandomState(self.seed_offset + int(index) * 1009)
        num_shapes = rng.randint(self.min_shapes, self.max_shapes + 1)
        clip = self._render_clip(num_shapes, rng)
        if self.output_dtype == "uint8":
            clip = np.clip(np.rint(clip * 255.0), 0, 255).astype(np.uint8)
        return torch.from_numpy(clip)

    def _render_clip(self, num_shapes, rng):
        H, W = self.image_size
        center_x = (W - 1) / 2.0 + rng.uniform(-0.05, 0.05) * W
        center_y = (H - 1) / 2.0 + rng.uniform(-0.05, 0.05) * H
        video = np.zeros((self.clip_len, H, W), dtype=np.uint8)
        params = []
        for idx in range(num_shapes):
            orbit_rx = rng.uniform(self.orbit_radius_min, self.orbit_radius_max)
            orbit_ry = orbit_rx * rng.uniform(0.65, 1.1)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            direction = -1.0 if rng.rand() < 0.5 else 1.0
            speed = direction * rng.uniform(self.angular_speed_min, self.angular_speed_max)
            rx = max(2, int(round(self.base_radius * rng.uniform(0.75, 1.25))))
            ry = max(2, int(round(self.base_radius * rng.uniform(0.75, 1.25))))
            params.append((orbit_rx, orbit_ry, phase, speed, rx, ry, idx))

        for t in range(self.clip_len):
            frame = np.zeros((H, W), dtype=np.uint8)
            for orbit_rx, orbit_ry, phase, speed, rx, ry, idx in params:
                theta = phase + speed * t
                cx = int(round(center_x + orbit_rx * np.cos(theta)))
                cy = int(round(center_y + orbit_ry * np.sin(theta)))
                angle = theta + 0.35 * idx
                if self.thickness_max > self.thickness_min:
                    thickness = int(round(rng.uniform(self.thickness_min, self.thickness_max)))
                else:
                    thickness = self.thickness_min
                if self.shape == "disk":
                    LorenzMovingShapesDataset._draw_filled_ellipse(frame, cx, cy, rx, ry, angle)
                else:
                    LorenzMovingShapesDataset._draw_hollow_ellipse(frame, cx, cy, rx, ry, angle, thickness=thickness)
            video[t] = frame
        return video[:, None].astype(np.float32) / 255.0

    def make_tensor(self):
        print(f"Generating {self.num_clips} synthetic orbiting {self.shape} clips...")
        data = torch.stack([self[i] for i in range(len(self))])
        print(f"Dataset generated. Final shape: {tuple(data.shape)}")
        return data


def orbiting_shapes_cache_path(config, split_name, num_clips, seed_offset):
    cache_dir = Path(config.get("cache_dir", "datasets/2D/orbiting_shapes/processed"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_size = tuple(config.get("image_size", (96, 96)))
    tag = (
        f"{split_name}_N{num_clips}_T{config.get('clip_len', 24)}_"
        f"H{image_size[0]}_W{image_size[1]}_"
        f"shape{config.get('shape', 'ring')}_"
        f"objects{config.get('min_shapes', 2)}-{config.get('max_shapes', 4)}_"
        f"r{config.get('base_radius', 6)}_"
        f"orbit{config.get('orbit_radius_min', 14)}-{config.get('orbit_radius_max', 32)}_"
        f"speed{config.get('angular_speed_min', 0.18)}-{config.get('angular_speed_max', 0.42)}_"
        f"dtype{config.get('cache_dtype', 'uint8')}_seed{seed_offset}.pt"
    )
    return cache_dir / tag.replace("/", "-")


def load_or_generate_orbiting_split(config, split_name, num_clips, seed_offset):
    cache_path = orbiting_shapes_cache_path(config, split_name, num_clips, seed_offset)
    force_rebuild = config.get("force_rebuild_cache", False)
    if cache_path.exists() and not force_rebuild:
        print(f"Loading cached orbiting-shapes {split_name}: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    data = OrbitingShapesDataset(
        num_clips=num_clips,
        clip_len=config.get("clip_len", 24),
        image_size=config.get("image_size", (96, 96)),
        base_radius=config.get("base_radius", 6),
        min_shapes=config.get("min_shapes", 2),
        max_shapes=config.get("max_shapes", 4),
        seed_offset=seed_offset,
        orbit_radius_min=config.get("orbit_radius_min", 14),
        orbit_radius_max=config.get("orbit_radius_max", 32),
        angular_speed_min=config.get("angular_speed_min", 0.18),
        angular_speed_max=config.get("angular_speed_max", 0.42),
        thickness_min=config.get("thickness_min", 2),
        thickness_max=config.get("thickness_max", 4),
        shape=config.get("shape", "ring"),
        output_dtype=config.get("cache_dtype", "uint8"),
    ).make_tensor()
    torch.save(data, cache_path)
    print(f"Saved orbiting-shapes {split_name} cache: {cache_path}")
    return data


def load_orbiting_shapes(config):
    train_n = config.get("num_train_clips", 512)
    test_n = config.get("num_test_clips", 128)
    train_seed = config.get("train_seed_offset", 0)
    test_seed = config.get("test_seed_offset", 50000)
    train = load_or_generate_orbiting_split(config, "train", train_n, train_seed)
    test = load_or_generate_orbiting_split(config, "test", test_n, test_seed)
    train = train.squeeze(2).permute(1, 0, 2, 3).numpy()
    test = test.squeeze(2).permute(1, 0, 2, 3).numpy()
    normalize = config.get("normalize", "minmax")
    if normalize == "minmax":
        if train.dtype != np.uint8:
            train = normalize_video_array(train)
            test = normalize_video_array(test)
    elif normalize in {None, "none"}:
        pass
    else:
        raise ValueError(f"Unknown orbiting_shapes normalize mode: {normalize}")
    return train, test


def lorenz_moving_shapes_cache_path(config, split_name, num_clips, seed_offset):
    cache_dir = Path(config.get("cache_dir", "datasets/2D/bouncing_rings/processed"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_size = tuple(config.get("image_size", (96, 96)))
    tag = (
        f"{split_name}_N{num_clips}_T{config.get('clip_len', 30)}_"
        f"H{image_size[0]}_W{image_size[1]}_"
        f"shape{config.get('shape', 'ring')}_"
        f"balls{config.get('min_balls', 2)}-{config.get('max_balls', 5)}_"
        f"r{config.get('base_radius', 6)}_dt{config.get('lorenz_dt', 0.015)}_"
        f"pulse{config.get('radius_pulse_amp', 0.25)}-{config.get('radius_pulse_freq', 0.15)}_"
        f"thick{config.get('thickness_min', 2)}-{config.get('thickness_max', 2)}_"
        f"overlap{config.get('overlap_strength', 0.0)}_"
        f"dtype{config.get('cache_dtype', 'uint8')}_seed{seed_offset}.pt"
    )
    return cache_dir / tag.replace("/", "-")


def load_or_generate_lorenz_shapes_split(config, split_name, num_clips, seed_offset, common):
    cache_path = lorenz_moving_shapes_cache_path(config, split_name, num_clips, seed_offset)
    force_rebuild = config.get("force_rebuild_cache", False)
    if cache_path.exists() and not force_rebuild:
        print(f"Loading cached Lorenz moving-shapes {split_name}: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    data = LorenzMovingShapesDataset(
        num_clips=num_clips,
        seed_offset=seed_offset,
        **common,
    ).make_tensor()
    torch.save(data, cache_path)
    print(f"Saved Lorenz moving-shapes {split_name} cache: {cache_path}")
    return data


def load_lorenz_moving_shapes(config):
    common = {
        "clip_len": config.get("clip_len", 30),
        "image_size": config.get("image_size", (96, 96)),
        "base_radius": config.get("base_radius", 6),
        "min_balls": config.get("min_balls", 2),
        "max_balls": config.get("max_balls", 5),
        "lorenz_dt": config.get("lorenz_dt", 0.015),
        "radius_pulse_amp": config.get("radius_pulse_amp", 0.25),
        "radius_pulse_freq": config.get("radius_pulse_freq", 0.15),
        "thickness_min": config.get("thickness_min", 2),
        "thickness_max": config.get("thickness_max", 2),
        "overlap_strength": config.get("overlap_strength", 0.0),
        "shape": config.get("shape", "ring"),
        "output_dtype": config.get("cache_dtype", "uint8"),
    }
    train_n = config.get("num_train_clips", 160)
    test_n = config.get("num_test_clips", 48)
    train_seed = config.get("train_seed_offset", 0)
    test_seed = config.get("test_seed_offset", 50000)
    train = load_or_generate_lorenz_shapes_split(config, "train", train_n, train_seed, common)
    test = load_or_generate_lorenz_shapes_split(config, "test", test_n, test_seed, common)
    train = train.squeeze(2).permute(1, 0, 2, 3).numpy()
    test = test.squeeze(2).permute(1, 0, 2, 3).numpy()
    normalize = config.get("normalize", "minmax")
    if normalize == "minmax":
        if train.dtype != np.uint8:
            train = normalize_video_array(train)
            test = normalize_video_array(test)
    elif normalize in {None, "none"}:
        pass
    else:
        raise ValueError(f"Unknown Lorenz moving-shapes normalize mode: {normalize}")
    return train, test

def load_video_dataset(config):
    if config["kind"] == "moving_mnist":
        arr = normalize_video_array(np.load(config["path"]))
        train = arr[:, config["train_slice"], :, :]
        test = arr[:, config["test_slice"], :, :]
    elif config["kind"] == "tif_folder":
        arr = normalize_video_array(load_tif_sequence(config["folder"]))
        arr = arr[:, None, :, :]
        train = arr[config["train_slice"], :, :, :]
        test = arr[config["test_slice"], :, :, :]
    elif config["kind"] == "davis_images":
        train, test = load_davis_images(config)
    elif config["kind"] == "lorenz_moving_shapes":
        train, test = load_lorenz_moving_shapes(config)
        return train, test
    elif config["kind"] == "orbiting_shapes":
        train, test = load_orbiting_shapes(config)
        return train, test
    else:
        raise ValueError(f"Unknown dataset kind: {config['kind']}")

    image_size = config.get("image_size", (64, 64))
    return resize_video_array(train, image_size), resize_video_array(test, image_size)
