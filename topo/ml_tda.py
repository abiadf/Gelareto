"""Reusable CNN + TDA helpers for the persistence notebook."""

from pathlib import Path
import random

import cv2
import cripser
import matplotlib.pyplot as plt
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
BETTI_SCALE = 15
N_STEPS = 25
PREDICT_STEPS_AHEAD = 1
EPOCHS = 10
criterion = nn.MSELoss()


def configure_runtime(**kwargs):
    """Update notebook-controlled globals used by legacy helper functions."""
    globals().update(kwargs)

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
    curves = np.clip(curves / BETTI_SCALE, 0.0, 1.0)
    return torch.from_numpy(curves).float()

def select_tda_features(h0_diags, h1_diags):
    betti_h0 = diagrams_to_betti_curves(h0_diags, num_steps=N_STEPS)
    betti_h1 = diagrams_to_betti_curves(h1_diags, num_steps=N_STEPS)
    if TDA_MODE == "h0":
        return betti_h0
    if TDA_MODE == "h1":
        return betti_h1
    return torch.cat([betti_h0, betti_h1], dim=1)

def run_cripser_tda_on_current_frames(video_prefix):
    """Compute H0/H1 with cripser for the current frame of each clip."""
    current_frames = video_prefix[-1, :, 0].detach().cpu().numpy()
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

def pretrain_spatial_encoder(encoder, decoder, X_train, epochs=3):
    print("\n--- Phase 1: Pre-training Spatial Encoder ---")
    ae_optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    all_frames   = X_train.reshape(-1, 1, 64, 64)
    for ae_epoch in range(epochs):
        encoder.train()
        decoder.train()
        ae_optimizer.zero_grad()
        latents = encoder(all_frames)
        reconstructions = decoder(latents)
        ae_loss = criterion(reconstructions, all_frames)
        ae_loss.backward()
        ae_optimizer.step()
        print(f"AE Pretrain Epoch {ae_epoch+1}/{epochs} | Reconstr MSE: {ae_loss.item():.4f}")

def build_sequence_features(video_tensor, encoder, use_tda, split_name="Train"):
    encoder.eval()
    fused_history = []
    z_history     = []
    with torch.no_grad():
        for t in range(video_tensor.shape[0]):
            frame_t = video_tensor[t]
            z_t     = encoder(frame_t)
            if use_tda:
                if (t + 1) % 20 == 0 or t == 0 or t == video_tensor.shape[0] - 1:
                    print(f"  [{split_name} Frame {t+1:02d}/{video_tensor.shape[0]}] Computing TDA")
                h0_diags, h1_diags = run_cripser_tda_on_current_frames(video_tensor[0:t+1])
                B_t = select_tda_features(h0_diags, h1_diags)
                f_t = torch.cat([z_t, B_t], dim=1)
            else:
                f_t = z_t
            fused_history.append(f_t)
            z_history.append(z_t)
    return torch.stack(fused_history, dim=0), torch.stack(z_history, dim=0)

def train_predictor(model, encoder, X_train, epochs, use_tda, learning_rate):
    print(f"\n--- Phase 2: Training LSTM Predictor (USE_TDA={use_tda}) ---")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        print(f"\n--- Epoch {epoch+1:02d}/{epochs:02d} ---")
        total_features, z_features = build_sequence_features(X_train, encoder, use_tda, split_name="Train")
        predict_steps_ahead = PREDICT_STEPS_AHEAD
        predictions = model(total_features[:-predict_steps_ahead])
        targets = z_features[predict_steps_ahead:]
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Training MSE Loss: {loss.item():.4f}")

def load_or_train_model(encoder, decoder, model, X_train, retrain_encoder, retrain_predictor, use_tda, learning_rate):
    model_dir   = Path("models") / DATASET
    model_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir = model_dir / "encoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)

    # The encoder is shared across TDA/no-TDA variants for the same dataset/config.
    # The predictor remains variant-specific because its input dimension changes.
    seed_tag     = globals().get("EXPERIMENT_SEED", "default")
    encoder_tag  = f"seed{seed_tag}_T{X_train.shape[0]}_B{X_train.shape[1]}_H{X_train.shape[-2]}_W{X_train.shape[-1]}_latent{LATENT_DIM}"
    predict_tag = f"pred{PREDICT_STEPS_AHEAD}"
    model_suffix = f"seed{seed_tag}_{predict_tag}_tda_{TDA_MODE}_bettiscale{BETTI_SCALE}" if use_tda else f"seed{seed_tag}_{predict_tag}_tda_none"
    encoder_path = encoder_dir / f"encoder_{encoder_tag}.pt"
    model_path   = model_dir / f"model_{model_suffix}.pt"

    if encoder_path.exists() and not retrain_encoder:
        print(f"Loading shared encoder: {encoder_path}")
        encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    else:
        reason = "retraining" if encoder_path.exists() else "missing; training once"
        print(f"Shared encoder {reason}: {encoder_path}")
        pretrain_spatial_encoder(encoder, decoder, X_train, epochs=3)
        torch.save(encoder.state_dict(), encoder_path)
        print(f"Saved shared encoder: {encoder_path}")

    for param in encoder.parameters():
        param.requires_grad = False

    if model_path.exists() and not retrain_predictor:
        print(f"Loading predictor: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        reason = "retraining" if model_path.exists() else "missing; training once"
        print(f"Predictor {reason}: {model_path}")
        train_predictor(model, encoder, X_train, epochs=EPOCHS, use_tda=use_tda, learning_rate=learning_rate)
        torch.save(model.state_dict(), model_path)
        print(f"Saved predictor weights to {model_path}")

def test_predictor(model, encoder, X_test, use_tda):
    print(f"\nTesting on Test Set (USE_TDA = {use_tda})...")
    model.eval()
    total_test_features, test_z_features = build_sequence_features(X_test, encoder, use_tda, split_name="Test")
    with torch.no_grad():
        predict_steps_ahead = PREDICT_STEPS_AHEAD
        test_predictions = model(total_test_features[:-predict_steps_ahead])
        test_targets = test_z_features[predict_steps_ahead:]
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
    epochs=1,
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
    for epoch in range(1, epochs + 1):
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
        print(f"AE Pretrain Epoch {epoch}/{epochs} | Reconstr MSE: {total_loss / total_seen:.4f}")

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
    else:
        raise ValueError(f"Unknown dataset kind: {config['kind']}")

    image_size = config.get("image_size", (64, 64))
    return resize_video_array(train, image_size), resize_video_array(test, image_size)
