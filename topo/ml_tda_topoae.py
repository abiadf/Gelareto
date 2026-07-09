"""Topology-aware autoencoder helpers for ML persistence experiments.

The regularizer here is a differentiable shape-preservation proxy: it aligns
normalized pairwise distances in image space and latent space for each training
mini-batch. This keeps the implementation lightweight while giving the encoder
pressure to preserve the geometry that latent-trajectory TDA later uses.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from topo import ml_tda
from topo.ml_tda import SpatialDecoder, SpatialEncoder


def topo_encoder_path(
    model_namespace: str,
    seed: int,
    x_train,
    latent_dim: int,
    topo_lambda: float,
) -> Path:
    encoder_dir = Path("models") / model_namespace / "topoae_encoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_T{x_train.shape[0]}_B{x_train.shape[1]}_"
        f"H{x_train.shape[-2]}_W{x_train.shape[-1]}_latent{latent_dim}_"
        f"lambda{topo_lambda:g}"
    )
    return encoder_dir / f"encoder_{tag}.pt"


def _normalized_pairwise_distances(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    distances = torch.cdist(x, x, p=2)
    scale = distances.detach().mean().clamp_min(eps)
    return distances / scale


def topology_proxy_loss(frames: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    """Match pairwise image-space and latent-space distances within a batch."""
    flat_frames = frames.flatten(start_dim=1)
    image_distances = _normalized_pairwise_distances(flat_frames).detach()
    latent_distances = _normalized_pairwise_distances(latents)
    return torch.mean((latent_distances - image_distances) ** 2)


def load_or_train_topo_encoder(
    x_train,
    *,
    dataset_name: str,
    model_namespace: str,
    seed: int,
    latent_dim: int,
    topo_lambda: float = 0.1,
    epochs: int = 3,
    frame_batch_size: int = 256,
    max_frames_per_epoch: int | None = 8192,
    pair_batch_size: int = 64,
    retrain: bool = False,
    learning_rate: float = 1e-3,
):
    """Load or train a topology-regularized spatial encoder."""
    encoder = SpatialEncoder(latent_dim=latent_dim)
    decoder = SpatialDecoder(latent_dim=latent_dim, output_size=x_train.shape[-2:])
    path = topo_encoder_path(model_namespace, seed, x_train, latent_dim, topo_lambda)

    if path.exists() and not retrain:
        print(f"Loading topo-AE encoder: {path}")
        encoder.load_state_dict(torch.load(path, map_location="cpu"))
        encoder.to(ml_tda.get_runtime_device()).eval()
        for param in encoder.parameters():
            param.requires_grad = False
        return encoder, path

    reason = "retraining" if path.exists() else "missing; training once"
    print(f"Topo-AE encoder {reason}: {path}")
    device = ml_tda.get_runtime_device()
    encoder.to(device)
    decoder.to(device)
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    recon_loss_fn = nn.MSELoss()
    all_frames = x_train.reshape(-1, *x_train.shape[2:])
    frame_batch_size = min(int(frame_batch_size), len(all_frames))
    pair_batch_size = max(2, int(pair_batch_size))
    print(
        f"Topo-AE config: dataset={dataset_name}, namespace={model_namespace}, "
        f"lambda={topo_lambda}, epochs={epochs}, device={device}"
    )

    for epoch in range(1, int(epochs) + 1):
        encoder.train()
        decoder.train()
        generator = torch.Generator().manual_seed(seed * 10_000 + epoch)
        if max_frames_per_epoch is None or max_frames_per_epoch >= len(all_frames):
            frame_idx = torch.randperm(len(all_frames), generator=generator)
        else:
            frame_idx = torch.randperm(len(all_frames), generator=generator)[:max_frames_per_epoch]

        total_recon = 0.0
        total_topo = 0.0
        total_seen = 0
        for start in range(0, len(frame_idx), frame_batch_size):
            idx = frame_idx[start:start + frame_batch_size]
            frames = ml_tda.tensor_to_model_float(all_frames[idx]).to(device, non_blocking=True)
            optimizer.zero_grad()
            latents = encoder(frames)
            reconstructions = decoder(latents)
            recon_loss = recon_loss_fn(reconstructions, frames)

            if len(frames) > pair_batch_size:
                pair_idx = torch.randperm(len(frames), device=device)[:pair_batch_size]
                topo_loss = topology_proxy_loss(frames[pair_idx], latents[pair_idx])
            else:
                topo_loss = topology_proxy_loss(frames, latents)

            loss = recon_loss + float(topo_lambda) * topo_loss
            loss.backward()
            optimizer.step()

            seen = len(frames)
            total_recon += recon_loss.item() * seen
            total_topo += topo_loss.item() * seen
            total_seen += seen

        print(
            f"Topo-AE epoch {epoch}/{epochs} | "
            f"recon_mse={total_recon / total_seen:.4f} "
            f"topo_proxy={total_topo / total_seen:.4f}"
        )

    torch.save(encoder.state_dict(), path)
    print(f"Saved topo-AE encoder: {path}")
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder, path
