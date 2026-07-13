"""Representation pretraining baselines for latent-TDA experiments."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from topo import ml_tda
from topo.ml_tda import SpatialEncoder
from topo.utils import tqdm_progress_bar


class VAEEncoder(nn.Module):
    """Spatial encoder returning the posterior mean as deterministic latent z."""

    def __init__(self, latent_dim=128):
        super().__init__()
        self.backbone = SpatialEncoder(latent_dim=latent_dim)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

    def encode_stats(self, x):
        h = self.backbone(x)
        return self.mu(h), self.logvar(h).clamp(-10.0, 10.0)

    def forward(self, x):
        mu, _ = self.encode_stats(x)
        return mu


class BYOLProjector(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


def _tag(seed, x_train, latent_dim, suffix):
    return f"seed{seed}_T{x_train.shape[0]}_B{x_train.shape[1]}_H{x_train.shape[-2]}_W{x_train.shape[-1]}_latent{latent_dim}_{suffix}"


def _frames(x_train):
    return x_train.reshape(-1, *x_train.shape[2:])


def _augment(frames, noise_std=0.05):
    x = ml_tda.tensor_to_model_float(frames)
    if noise_std > 0:
        x = (x + noise_std * torch.randn_like(x)).clamp(0.0, 1.0)
    # Lightweight translation augmentation; keeps shape identity.
    shifts = torch.randint(-2, 3, (2,), device=x.device)
    return torch.roll(x, shifts=(int(shifts[0]), int(shifts[1])), dims=(-2, -1))


def _cosine_loss(p, z):
    p = nn.functional.normalize(p, dim=-1)
    z = nn.functional.normalize(z.detach(), dim=-1)
    return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()


def load_or_train_vae_encoder(
    x_train,
    *,
    dataset_name,
    seed,
    latent_dim,
    ae_epochs=3,
    beta=1e-3,
    frame_batch_size=256,
    max_frames_per_epoch=8192,
    retrain=False,
):
    model_dir = Path("models") / f"{dataset_name}_vae_beta{beta:g}" / "encoders"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"encoder_{_tag(seed, x_train, latent_dim, 'vae')}.pt"
    encoder = VAEEncoder(latent_dim=latent_dim)
    decoder = ml_tda.make_spatial_decoder("mlp", latent_dim=latent_dim, output_size=x_train.shape[-2:])
    if path.exists() and not retrain:
        print(f"Loading VAE encoder: {path}")
        encoder.load_state_dict(torch.load(path, map_location="cpu"))
        encoder.to(ml_tda.get_runtime_device()).eval()
        return encoder, path

    device = ml_tda.get_runtime_device()
    encoder.to(device)
    decoder.to(device)
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    recon_loss = nn.MSELoss()
    frames = _frames(x_train)
    batch_size = min(int(frame_batch_size), len(frames))
    max_frames = min(int(max_frames_per_epoch), len(frames))
    print(f"Training VAE encoder: {path} | beta={beta}")
    for epoch in tqdm_progress_bar(range(1, int(ae_epochs) + 1), desc="VAE epochs", total=int(ae_epochs), leave=True):
        generator = torch.Generator().manual_seed(seed * 10_000 + epoch)
        idx_all = torch.randperm(len(frames), generator=generator)[:max_frames]
        total = 0.0
        total_seen = 0
        for start in tqdm_progress_bar(range(0, len(idx_all), batch_size), desc=f"VAE epoch {epoch} batches", total=(len(idx_all) + batch_size - 1) // batch_size):
            batch = ml_tda.tensor_to_model_float(frames[idx_all[start:start + batch_size]]).to(device)
            mu, logvar = encoder.encode_stats(batch)
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
            recon = decoder(z)
            rec = recon_loss(recon, batch)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = rec + float(beta) * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch.shape[0]
            total_seen += batch.shape[0]
        print(f"VAE epoch {epoch}/{ae_epochs} | loss={total / max(total_seen, 1):.4f}")
    torch.save(encoder.state_dict(), path)
    encoder.eval()
    return encoder, path


def load_or_train_byol_encoder(
    x_train,
    *,
    dataset_name,
    seed,
    latent_dim,
    byol_epochs=3,
    noise_std=0.05,
    frame_batch_size=256,
    max_frames_per_epoch=8192,
    retrain=False,
):
    model_dir = Path("models") / f"{dataset_name}_byol" / "encoders"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"encoder_{_tag(seed, x_train, latent_dim, 'byol')}.pt"
    encoder = SpatialEncoder(latent_dim=latent_dim)
    if path.exists() and not retrain:
        print(f"Loading BYOL-style encoder: {path}")
        encoder.load_state_dict(torch.load(path, map_location="cpu"))
        encoder.to(ml_tda.get_runtime_device()).eval()
        return encoder, path

    device = ml_tda.get_runtime_device()
    encoder.to(device)
    projector = BYOLProjector(latent_dim=latent_dim).to(device)
    predictor = BYOLProjector(latent_dim=latent_dim).to(device)
    optimizer = optim.AdamW(list(encoder.parameters()) + list(projector.parameters()) + list(predictor.parameters()), lr=1e-3)
    frames = _frames(x_train)
    batch_size = min(int(frame_batch_size), len(frames))
    max_frames = min(int(max_frames_per_epoch), len(frames))
    print(f"Training BYOL-style encoder: {path}")
    for epoch in tqdm_progress_bar(range(1, int(byol_epochs) + 1), desc="BYOL epochs", total=int(byol_epochs), leave=True):
        generator = torch.Generator().manual_seed(seed * 10_000 + epoch)
        idx_all = torch.randperm(len(frames), generator=generator)[:max_frames]
        total = 0.0
        total_seen = 0
        for start in tqdm_progress_bar(range(0, len(idx_all), batch_size), desc=f"BYOL epoch {epoch} batches", total=(len(idx_all) + batch_size - 1) // batch_size):
            batch = frames[idx_all[start:start + batch_size]].to(device)
            x1 = _augment(batch, noise_std=float(noise_std))
            x2 = _augment(batch, noise_std=float(noise_std))
            z1 = projector(encoder(x1))
            z2 = projector(encoder(x2))
            p1 = predictor(z1)
            p2 = predictor(z2)
            loss = 0.5 * (_cosine_loss(p1, z2) + _cosine_loss(p2, z1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch.shape[0]
            total_seen += batch.shape[0]
        print(f"BYOL epoch {epoch}/{byol_epochs} | loss={total / max(total_seen, 1):.4f}")
    torch.save(encoder.state_dict(), path)
    encoder.eval()
    return encoder, path
