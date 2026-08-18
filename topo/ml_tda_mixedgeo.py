"""GeoAE with a fixed product of Euclidean, spherical, and hyperbolic factors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.optim as optim

from topo import ml_tda
from topo.ml_tda import SpatialEncoder, make_spatial_decoder
from topo.utils import tqdm_progress_bar


@dataclass(frozen=True)
class ManifoldFactor:
    kind: str
    dim: int


def parse_manifold_signature(signature: str, expected_dim: int | None = None) -> tuple[ManifoldFactor, ...]:
    """Parse signatures such as ``e16`` or ``h6_s6_e4``."""
    parts = str(signature).lower().strip().split("_")
    factors = []
    names = {"e": "euclidean", "s": "spherical", "h": "hyperbolic"}
    for part in parts:
        match = re.fullmatch(r"([esh])(\d+)", part)
        if match is None or int(match.group(2)) < 1:
            raise ValueError(f"Invalid manifold factor {part!r} in signature {signature!r}")
        factors.append(ManifoldFactor(names[match.group(1)], int(match.group(2))))
    total_dim = sum(factor.dim for factor in factors)
    if expected_dim is not None and total_dim != int(expected_dim):
        raise ValueError(
            f"Signature {signature!r} has intrinsic dimension {total_dim}, expected {expected_dim}"
        )
    return tuple(factors)


def _bounded_tangent(x: torch.Tensor, max_norm: float, eps: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_norm) / norm.clamp_min(eps), max=1.0)
    return x * scale


def _sphere_points(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = _bounded_tangent(x, torch.pi - 1e-4, eps)
    radius = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    direction = x / radius.clamp_min(eps)
    return torch.cat([torch.cos(radius), torch.sin(radius) * direction], dim=-1)


def _hyperboloid_points(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = _bounded_tangent(x, 8.0, eps)
    radius = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    direction = x / radius.clamp_min(eps)
    return torch.cat([torch.cosh(radius), torch.sinh(radius) * direction], dim=-1)


def product_pairwise_distances(
    tangent_latents: torch.Tensor,
    signature: str,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pairwise geodesic distance in a unit-curvature product manifold."""
    factors = parse_manifold_signature(signature, tangent_latents.shape[-1])
    # Preserve exact Euclidean cdist semantics, including zero-distance pairs.
    if all(factor.kind == "euclidean" for factor in factors):
        return torch.cdist(tangent_latents, tangent_latents)
    squared = torch.zeros(
        (*tangent_latents.shape[:-2], tangent_latents.shape[-2], tangent_latents.shape[-2]),
        device=tangent_latents.device,
        dtype=tangent_latents.dtype,
    )
    start = 0
    for factor in factors:
        x = tangent_latents[..., start:start + factor.dim]
        start += factor.dim
        if factor.kind == "euclidean":
            distance = torch.cdist(x, x)
        elif factor.kind == "spherical":
            points = _sphere_points(x, eps)
            cosine = points @ points.transpose(-1, -2)
            distance = torch.acos(cosine.clamp(-1.0 + eps, 1.0 - eps))
            distance = distance - torch.diag_embed(torch.diagonal(distance, dim1=-2, dim2=-1))
        else:
            points = _hyperboloid_points(x, eps)
            time = points[..., :1]
            space = points[..., 1:]
            lorentz = -(time @ time.transpose(-1, -2)) + space @ space.transpose(-1, -2)
            distance = torch.acosh((-lorentz).clamp_min(1.0 + eps))
            distance = distance - torch.diag_embed(torch.diagonal(distance, dim1=-2, dim2=-1))
        squared = squared + distance.square()
    # The shifted square root is zero at coincident points but has a finite
    # derivative there.  A raw sqrt(0) makes mixed-manifold training NaN when
    # an encoder initially maps two observations to the same latent point.
    distance = torch.sqrt(squared.clamp_min(0.0) + eps) - eps**0.5
    return distance - torch.diag_embed(torch.diagonal(distance, dim1=-2, dim2=-1))


def factor_pairwise_distances(
    tangent_latents: torch.Tensor,
    signature: str,
    *,
    eps: float = 1e-6,
) -> tuple[tuple[ManifoldFactor, torch.Tensor], ...]:
    """Return the pairwise geodesic-distance matrix for every product factor."""
    factors = parse_manifold_signature(signature, tangent_latents.shape[-1])
    result = []
    start = 0
    for factor in factors:
        x = tangent_latents[..., start:start + factor.dim]
        start += factor.dim
        if factor.kind == "euclidean":
            distance = torch.cdist(x, x)
        elif factor.kind == "spherical":
            points = _sphere_points(x, eps)
            cosine = points @ points.transpose(-1, -2)
            distance = torch.acos(cosine.clamp(-1.0 + eps, 1.0 - eps))
        else:
            points = _hyperboloid_points(x, eps)
            time, space = points[..., :1], points[..., 1:]
            lorentz = -(time @ time.transpose(-1, -2)) + space @ space.transpose(-1, -2)
            distance = torch.acosh((-lorentz).clamp_min(1.0 + eps))
        distance = distance - torch.diag_embed(
            torch.diagonal(distance, dim1=-2, dim2=-1)
        )
        result.append((factor, distance))
    return tuple(result)


def mixed_geometry_loss(frames: torch.Tensor, latents: torch.Tensor, signature: str, eps: float = 1e-6):
    flat_frames = frames.flatten(start_dim=1)
    input_distances = torch.cdist(flat_frames, flat_frames)
    latent_distances = product_pairwise_distances(latents, signature, eps=eps)
    input_distances = input_distances / input_distances.detach().mean().clamp_min(eps)
    latent_distances = latent_distances / latent_distances.detach().mean().clamp_min(eps)
    return torch.mean((latent_distances - input_distances.detach()).square())


def _paths(model_namespace: str, signature: str, seed: int, x_train, latent_dim: int, geo_lambda: float):
    root = Path("models") / model_namespace
    tag = (
        f"seed{seed}_T{x_train.shape[0]}_B{x_train.shape[1]}_H{x_train.shape[-2]}_"
        f"W{x_train.shape[-1]}_latent{latent_dim}_lambda{geo_lambda:g}_{signature}"
    )
    encoder_path = root / "mixedgeo_encoders" / f"encoder_{tag}.pt"
    decoder_path = root / "mixedgeo_decoders" / f"decoder_{tag}.pt"
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    decoder_path.parent.mkdir(parents=True, exist_ok=True)
    return encoder_path, decoder_path


def load_or_train_mixedgeo_encoder(x_train, **kwargs):
    encoder, _, encoder_path, _ = load_or_train_mixedgeo_autoencoder(x_train, **kwargs)
    return encoder, encoder_path


def load_or_train_mixedgeo_autoencoder(
    x_train,
    *,
    dataset_name: str,
    model_namespace: str,
    signature: str,
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
    parse_manifold_signature(signature, latent_dim)
    encoder = SpatialEncoder(latent_dim=latent_dim)
    decoder = make_spatial_decoder(decoder_type, latent_dim=latent_dim, output_size=x_train.shape[-2:])
    encoder_path, decoder_path = _paths(model_namespace, signature, seed, x_train, latent_dim, geo_lambda)
    device = ml_tda.get_runtime_device()

    if encoder_path.exists() and decoder_path.exists() and not retrain:
        print(f"Loading mixed-GeoAE ({signature}): {encoder_path}")
        encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
    else:
        print(f"Training mixed-GeoAE ({signature}): {encoder_path}")
        encoder.to(device)
        decoder.to(device)
        optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
        all_frames = x_train.reshape(-1, *x_train.shape[2:])
        batch_size = min(int(frame_batch_size), len(all_frames))
        pair_batch_size = max(2, int(pair_batch_size))
        for epoch in tqdm_progress_bar(range(1, int(ae_epochs) + 1), desc=f"mixed-GeoAE {signature}", total=int(ae_epochs), leave=True):
            encoder.train(); decoder.train()
            generator = torch.Generator().manual_seed(seed * 10_000 + epoch)
            order = torch.randperm(len(all_frames), generator=generator)
            if max_frames_per_epoch is not None:
                order = order[:int(max_frames_per_epoch)]
            total_recon = total_geo = total_seen = 0.0
            for batch_start in range(0, len(order), batch_size):
                frames = ml_tda.tensor_to_model_float(all_frames[order[batch_start:batch_start + batch_size]]).to(device)
                optimizer.zero_grad()
                z = encoder(frames)
                reconstruction = nn.functional.mse_loss(decoder(z), frames)
                if len(frames) > pair_batch_size:
                    chosen = torch.randperm(len(frames), device=device)[:pair_batch_size]
                    geometry = mixed_geometry_loss(frames[chosen], z[chosen], signature)
                else:
                    geometry = mixed_geometry_loss(frames, z, signature)
                loss = reconstruction + float(geo_lambda) * geometry
                loss.backward(); optimizer.step()
                seen = len(frames)
                total_recon += float(reconstruction.detach()) * seen
                total_geo += float(geometry.detach()) * seen
                total_seen += seen
            print(f"mixed-GeoAE {signature} epoch {epoch}/{ae_epochs} | recon_mse={total_recon/total_seen:.4f} geo={total_geo/total_seen:.4f}")
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)

    encoder.to(device).eval(); decoder.to(device).eval()
    for parameter in list(encoder.parameters()) + list(decoder.parameters()):
        parameter.requires_grad = False
    return encoder, decoder, encoder_path, decoder_path
