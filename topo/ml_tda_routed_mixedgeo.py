"""Persistence-routed product-manifold geometry autoencoder.

The global mixed-geometry loss is retained.  Input-space MST edges (exact H0
critical relations) and salient non-tree kNN edges (cycle-closing H1
candidates) are selected with stopped gradients.  Each selected relation is
then softly routed to whichever Euclidean, spherical, or hyperbolic product
factor currently represents its normalized input distance with least
distortion.  This deliberately avoids a fixed H0-to-H or H1-to-S assumption.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from topo import ml_tda
from topo.ml_tda import SpatialEncoder, make_spatial_decoder
from topo.ml_tda_mixedgeo import (
    factor_pairwise_distances,
    mixed_geometry_loss,
    parse_manifold_signature,
)
from topo.ml_tda_pwgeoae import persistence_edge_weights
from topo.utils import tqdm_progress_bar


def routed_geometry_loss(
    frames: torch.Tensor,
    latents: torch.Tensor,
    signature: str,
    *,
    knn: int = 5,
    temperature: float = 0.1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return soft-routed critical-relation loss and mean factor assignments.

    Routing targets are derived only from input distances. Factor assignments
    remain differentiable with respect to the latent representation. The
    returned assignment vector follows the factor order in ``signature``.
    """
    if temperature <= 0:
        raise ValueError(f"Routing temperature must be positive, got {temperature}")
    input_d = torch.cdist(frames.flatten(start_dim=1), frames.flatten(start_dim=1))
    input_d = input_d / input_d.detach().mean().clamp_min(eps)
    relation_weights = persistence_edge_weights(
        input_d, knn=knn, h0_weight=2.0, h1_weight=1.0
    ).detach()
    edge_mask = torch.triu(relation_weights > 0, diagonal=1)
    if not bool(edge_mask.any()):
        zero = latents.sum() * 0.0
        n_factors = len(parse_manifold_signature(signature, latents.shape[-1]))
        return zero, torch.full((n_factors,), 1.0 / n_factors, device=latents.device)

    target = input_d.detach()[edge_mask]
    errors = []
    for _, distances in factor_pairwise_distances(latents, signature, eps=eps):
        normalized = distances / distances.detach().mean().clamp_min(eps)
        errors.append((normalized[edge_mask] - target).square())
    errors = torch.stack(errors, dim=-1)
    assignments = torch.softmax(-errors / float(temperature), dim=-1)
    edge_weights = relation_weights[edge_mask]
    loss = (edge_weights[:, None] * assignments * errors).sum()
    loss = loss / edge_weights.sum().clamp_min(eps)
    return loss, assignments.detach().mean(dim=0)


def _paths(model_namespace: str, tag: str):
    root = Path("models") / model_namespace
    encoder_path = root / "routed_mixedgeo_encoders" / f"encoder_{tag}.pt"
    decoder_path = root / "routed_mixedgeo_decoders" / f"decoder_{tag}.pt"
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    decoder_path.parent.mkdir(parents=True, exist_ok=True)
    return encoder_path, decoder_path


def load_or_train_routed_mixedgeo_encoder(x_train, **kwargs):
    """Load or train the routed mixed-GeoAE and return its frozen encoder."""
    encoder, _, encoder_path, _ = load_or_train_routed_mixedgeo_autoencoder(x_train, **kwargs)
    return encoder, encoder_path


def load_or_train_routed_mixedgeo_autoencoder(
    x_train,
    *,
    dataset_name: str,
    model_namespace: str,
    signature: str,
    seed: int,
    latent_dim: int,
    geo_lambda: float = 0.1,
    route_lambda: float = 0.1,
    route_knn: int = 5,
    route_temperature: float = 0.1,
    ae_epochs: int = 3,
    frame_batch_size: int = 256,
    max_frames_per_epoch: int | None = 8192,
    pair_batch_size: int = 64,
    retrain: bool = False,
    learning_rate: float = 1e-3,
    decoder_type: str = "mlp",
):
    """Train reconstruction + global product geometry + routed geometry."""
    del dataset_name  # namespace fully identifies the checkpoint location
    parse_manifold_signature(signature, latent_dim)
    encoder = SpatialEncoder(latent_dim=latent_dim)
    decoder = make_spatial_decoder(decoder_type, latent_dim=latent_dim, output_size=x_train.shape[-2:])
    tag = (
        f"seed{seed}_T{x_train.shape[0]}_B{x_train.shape[1]}_H{x_train.shape[-2]}_"
        f"W{x_train.shape[-1]}_latent{latent_dim}_geo{geo_lambda:g}_route{route_lambda:g}_"
        f"k{route_knn}_temp{route_temperature:g}_{signature}_{decoder_type}"
    )
    encoder_path, decoder_path = _paths(model_namespace, tag)
    device = ml_tda.get_runtime_device()

    if encoder_path.exists() and decoder_path.exists() and not retrain:
        print(f"Loading routed mixed-GeoAE ({signature}): {encoder_path}")
        encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
    else:
        print(f"Training routed mixed-GeoAE ({signature}): {encoder_path}")
        encoder.to(device); decoder.to(device)
        optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
        all_frames = x_train.reshape(-1, *x_train.shape[2:])
        batch_size = min(int(frame_batch_size), len(all_frames))
        pair_batch_size = max(2, int(pair_batch_size))
        for epoch in tqdm_progress_bar(range(1, int(ae_epochs) + 1), desc=f"routed mixed-GeoAE {signature}", total=int(ae_epochs), leave=True):
            encoder.train(); decoder.train()
            generator = torch.Generator().manual_seed(seed * 10_000 + epoch)
            order = torch.randperm(len(all_frames), generator=generator)
            if max_frames_per_epoch is not None:
                order = order[:int(max_frames_per_epoch)]
            total_recon = total_geo = total_route = total_seen = 0.0
            total_assignments = torch.zeros(
                len(parse_manifold_signature(signature, latent_dim)), device=device
            )
            assignment_batches = 0
            for start in range(0, len(order), batch_size):
                frames = ml_tda.tensor_to_model_float(all_frames[order[start:start + batch_size]]).to(device)
                optimizer.zero_grad()
                z = encoder(frames)
                recon = nn.functional.mse_loss(decoder(z), frames)
                chosen = torch.arange(len(frames), device=device)
                if len(frames) > pair_batch_size:
                    chosen = torch.randperm(len(frames), device=device)[:pair_batch_size]
                geometry = mixed_geometry_loss(frames[chosen], z[chosen], signature)
                route, assignments = routed_geometry_loss(
                    frames[chosen], z[chosen], signature,
                    knn=route_knn, temperature=route_temperature,
                )
                loss = recon + float(geo_lambda) * geometry + float(route_lambda) * route
                loss.backward(); optimizer.step()
                seen = len(frames)
                total_recon += float(recon.detach()) * seen
                total_geo += float(geometry.detach()) * seen
                total_route += float(route.detach()) * seen
                total_seen += seen
                total_assignments += assignments
                assignment_batches += 1
            factor_labels = [factor.kind[0].upper() for factor in parse_manifold_signature(signature, latent_dim)]
            assignment_mean = total_assignments / max(1, assignment_batches)
            assignment_text = ",".join(
                f"{label}:{float(value):.3f}" for label, value in zip(factor_labels, assignment_mean)
            )
            print(
                f"routed mixed-GeoAE {signature} epoch {epoch}/{ae_epochs} | "
                f"recon_mse={total_recon/total_seen:.4f} geo={total_geo/total_seen:.4f} "
                f"route={total_route/total_seen:.4f} assignments={assignment_text}"
            )
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)

    encoder.to(device).eval(); decoder.to(device).eval()
    for parameter in list(encoder.parameters()) + list(decoder.parameters()):
        parameter.requires_grad = False
    return encoder, decoder, encoder_path, decoder_path
