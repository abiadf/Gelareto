"""Geometry-aware autoencoder helpers for ML persistence experiments.

The regularizer here is a differentiable geometry-preservation proxy: it aligns
normalized pairwise distances in image space and latent space for each training
mini-batch. This is not a persistence loss; it is the lightweight geometric
baseline used by geo_* scenarios.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from topo import ml_tda
from topo.ml_tda import SpatialDecoder, SpatialEncoder, make_spatial_decoder
from topo.utils import tqdm_progress_bar


def _decoder_suffix(decoder_type: str = "mlp") -> str:
    decoder_type = str(decoder_type).lower().strip()
    return "" if decoder_type == "mlp" else f"_dec{decoder_type}"


def geo_encoder_path(
    model_namespace: str,
    seed: int,
    x_train,
    latent_dim: int,
    geo_lambda: float,
    decoder_type: str = "mlp",
) -> Path:
    encoder_dir = Path("models") / model_namespace / "geoae_encoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_T{x_train.shape[0]}_B{x_train.shape[1]}_"
        f"H{x_train.shape[-2]}_W{x_train.shape[-1]}_latent{latent_dim}_"
        f"lambda{geo_lambda:g}{_decoder_suffix(decoder_type)}"
    )
    return encoder_dir / f"encoder_{tag}.pt"


def geo_decoder_path(
    model_namespace: str,
    seed: int,
    x_train,
    latent_dim: int,
    geo_lambda: float,
    decoder_type: str = "mlp",
) -> Path:
    decoder_dir = Path("models") / model_namespace / "geoae_decoders"
    decoder_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_T{x_train.shape[0]}_B{x_train.shape[1]}_"
        f"H{x_train.shape[-2]}_W{x_train.shape[-1]}_latent{latent_dim}_"
        f"lambda{geo_lambda:g}{_decoder_suffix(decoder_type)}"
    )
    return decoder_dir / f"decoder_{tag}.pt"


def _normalized_pairwise_distances(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    distances = torch.cdist(x, x, p=2)
    scale = distances.detach().mean().clamp_min(eps)
    return distances / scale


def geometry_proxy_loss(frames: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    """Match pairwise image-space and latent-space distances within a batch."""
    flat_frames = frames.flatten(start_dim=1)
    image_distances = _normalized_pairwise_distances(flat_frames).detach()
    latent_distances = _normalized_pairwise_distances(latents)
    return torch.mean((latent_distances - image_distances) ** 2)


def load_or_train_geo_encoder(
    x_train,
    *,
    dataset_name: str,
    model_namespace: str,
    seed: int,
    latent_dim: int,
    geo_lambda: float = 0.1,
    ae_epochs: int = 3,
    frame_batch_size: int = 256,
    max_frames_per_epoch: int | None = 8192,
    pair_batch_size: int = 64,
    retrain: bool = False,
    learning_rate: float = 1e-3,
    decoder_type: str = "mlp",
):
    """Load or train a geometry-regularized spatial encoder."""
    encoder, _, encoder_path, _ = load_or_train_geo_autoencoder(
        x_train,
        dataset_name=dataset_name,
        model_namespace=model_namespace,
        seed=seed,
        latent_dim=latent_dim,
        geo_lambda=geo_lambda,
        ae_epochs=ae_epochs,
        frame_batch_size=frame_batch_size,
        max_frames_per_epoch=max_frames_per_epoch,
        pair_batch_size=pair_batch_size,
        retrain=retrain,
        learning_rate=learning_rate,
        decoder_type=decoder_type,
    )
    return encoder, encoder_path


def load_or_train_geo_autoencoder(
    x_train,
    *,
    dataset_name: str,
    model_namespace: str,
    seed: int,
    latent_dim: int,
    geo_lambda: float = 0.1,
    ae_epochs: int = 3,
    frame_batch_size: int = 256,
    max_frames_per_epoch: int | None = 8192,
    pair_batch_size: int = 64,
    retrain: bool = False,
    learning_rate: float = 1e-3,
    decoder_type: str = "mlp",
):
    """Load or train a geometry-regularized spatial autoencoder."""
    encoder = SpatialEncoder(latent_dim=latent_dim)
    decoder = make_spatial_decoder(decoder_type, latent_dim=latent_dim, output_size=x_train.shape[-2:])
    encoder_path = geo_encoder_path(model_namespace, seed, x_train, latent_dim, geo_lambda, decoder_type)
    decoder_path = geo_decoder_path(model_namespace, seed, x_train, latent_dim, geo_lambda, decoder_type)

    if encoder_path.exists() and decoder_path.exists() and not retrain:
        print(f"Loading geo-AE encoder: {encoder_path}")
        print(f"Loading geo-AE decoder: {decoder_path}")
        encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
        device = ml_tda.get_runtime_device()
        encoder.to(device).eval()
        decoder.to(device).eval()
        for param in encoder.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False
        return encoder, decoder, encoder_path, decoder_path

    if encoder_path.exists() and not decoder_path.exists() and not retrain:
        print(f"Loading geo-AE encoder: {encoder_path}")
        print(f"Geo-AE decoder missing; training decoder only: {decoder_path}")
        encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        ml_tda.train_decoder_for_encoder(encoder, decoder, x_train, ae_epochs=ae_epochs, learning_rate=learning_rate)
        torch.save(decoder.state_dict(), decoder_path)
        device = ml_tda.get_runtime_device()
        encoder.to(device).eval()
        decoder.to(device).eval()
        for param in encoder.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False
        return encoder, decoder, encoder_path, decoder_path

    reason = "retraining" if encoder_path.exists() or decoder_path.exists() else "missing; training once"
    print(f"Geo-AE autoencoder {reason}: {encoder_path} | {decoder_path}")
    device = ml_tda.get_runtime_device()
    encoder.to(device)
    decoder.to(device)
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    recon_loss_fn = nn.MSELoss()
    all_frames = x_train.reshape(-1, *x_train.shape[2:])
    frame_batch_size = min(int(frame_batch_size), len(all_frames))
    pair_batch_size = max(2, int(pair_batch_size))
    print(
        f"Geo-AE config: dataset={dataset_name}, namespace={model_namespace}, "
        f"lambda={geo_lambda}, ae_epochs={ae_epochs}, device={device}"
    )

    epochs = range(1, int(ae_epochs) + 1)
    for epoch in tqdm_progress_bar(epochs, desc="Geo-AE epochs", total=int(ae_epochs), leave=True):
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
        batches = range(0, len(frame_idx), frame_batch_size)
        n_batches = (len(frame_idx) + frame_batch_size - 1) // frame_batch_size
        for start in tqdm_progress_bar(batches, desc=f"Geo-AE epoch {epoch} batches", total=n_batches):
            idx = frame_idx[start:start + frame_batch_size]
            frames = ml_tda.tensor_to_model_float(all_frames[idx]).to(device, non_blocking=True)
            optimizer.zero_grad()
            latents = encoder(frames)
            reconstructions = decoder(latents)
            recon_loss = recon_loss_fn(reconstructions, frames)

            if len(frames) > pair_batch_size:
                pair_idx = torch.randperm(len(frames), device=device)[:pair_batch_size]
                topo_loss = geometry_proxy_loss(frames[pair_idx], latents[pair_idx])
            else:
                topo_loss = geometry_proxy_loss(frames, latents)

            loss = recon_loss + float(geo_lambda) * topo_loss
            loss.backward()
            optimizer.step()

            seen = len(frames)
            total_recon += recon_loss.item() * seen
            total_topo += topo_loss.item() * seen
            total_seen += seen

        print(
            f"Geo-AE epoch {epoch}/{ae_epochs} | "
            f"recon_mse={total_recon / total_seen:.4f} "
            f"geo_proxy={total_topo / total_seen:.4f}"
        )

    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    print(f"Saved geo-AE encoder: {encoder_path}")
    print(f"Saved geo-AE decoder: {decoder_path}")
    encoder.eval()
    decoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False
    return encoder, decoder, encoder_path, decoder_path
