"""Persistence-weighted geometry autoencoder helpers.

PW-GeoAE retains GeoAE's global all-pairs distance loss and adds emphasis on a
sparse graph containing local kNN edges, exact H0-critical edges (the Euclidean
minimum spanning tree), and non-tree kNN edges that close cycles. Edge selection
and weights are computed from the input with stopped gradients; gradients flow
through the corresponding latent distances.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from topo import ml_tda
from topo.ml_tda import SpatialEncoder, make_spatial_decoder
from topo.ml_tda_geoae import geometry_proxy_loss
from topo.utils import tqdm_progress_bar


def _suffix(decoder_type: str) -> str:
    value = str(decoder_type).lower().strip()
    return "" if value == "mlp" else f"_dec{value}"


def _model_tag(seed, x_train, latent_dim, pw_lambda, blend, knn, h0_weight, h1_weight, decoder_type):
    return (
        f"seed{seed}_T{x_train.shape[0]}_B{x_train.shape[1]}_"
        f"H{x_train.shape[-2]}_W{x_train.shape[-1]}_latent{latent_dim}_"
        f"lambda{pw_lambda:g}_blend{blend:g}_k{knn}_h0{h0_weight:g}_h1{h1_weight:g}"
        f"{_suffix(decoder_type)}"
    )


def _paths_max_edge(mst: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    """Maximum MST-edge length on every tree path (small batches only)."""
    n = len(mst)
    result = torch.zeros_like(distances)
    adjacency = [[] for _ in range(n)]
    for i, j in mst.nonzero(as_tuple=False).tolist():
        if i < j:
            adjacency[i].append(j)
            adjacency[j].append(i)
    for source in range(n):
        stack = [(source, -1, 0.0)]
        while stack:
            node, parent, path_max = stack.pop()
            result[source, node] = path_max
            for nxt in adjacency[node]:
                if nxt != parent:
                    edge = float(distances[node, nxt])
                    stack.append((nxt, node, max(path_max, edge)))
    return result


def persistence_edge_weights(
    distances: torch.Tensor,
    *,
    knn: int = 5,
    h0_weight: float = 2.0,
    h1_weight: float = 1.0,
) -> torch.Tensor:
    """Build symmetric local/H0/cycle edge weights from a distance matrix."""
    d = distances.detach().cpu()
    n = len(d)
    weights = torch.zeros_like(d)
    if n < 2:
        return weights.to(distances.device)

    # Local neighbourhood graph.
    k = min(max(1, int(knn)), n - 1)
    nearest = torch.topk(d + torch.eye(n) * torch.finfo(d.dtype).max, k, largest=False).indices
    rows = torch.arange(n).repeat_interleave(k)
    weights[rows, nearest.reshape(-1)] = 1.0
    weights = torch.maximum(weights, weights.T)

    # Prim's algorithm: these are precisely the H0 persistence-critical edges.
    mst = torch.zeros_like(d, dtype=torch.bool)
    selected = torch.zeros(n, dtype=torch.bool)
    selected[0] = True
    for _ in range(n - 1):
        candidates = d.clone()
        candidates[~selected, :] = float("inf")
        candidates[:, selected] = float("inf")
        flat = int(torch.argmin(candidates))
        i, j = divmod(flat, n)
        selected[j] = True
        mst[i, j] = mst[j, i] = True
    weights[mst] += float(h0_weight)

    # Non-tree local edges close graph cycles.  Their excess above the largest
    # MST edge on the induced path is a stable cycle-salience proxy.  This is
    # intentionally a sparse H1 candidate weighting, not a diagram loss.
    if h1_weight > 0:
        path_max = _paths_max_edge(mst, d)
        cycle = (weights > 0) & ~mst & ~torch.eye(n, dtype=torch.bool)
        salience = (d - path_max).clamp_min(0) * cycle
        positive = salience[salience > 0]
        if len(positive):
            salience = salience / positive.mean().clamp_min(1e-8)
            weights += float(h1_weight) * salience

    weights.fill_diagonal_(0)
    return weights.to(device=distances.device, dtype=distances.dtype)


def persistence_weighted_geometry_loss(
    frames: torch.Tensor,
    latents: torch.Tensor,
    *,
    knn: int = 5,
    h0_weight: float = 2.0,
    h1_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    flat = frames.flatten(start_dim=1)
    input_d = torch.cdist(flat, flat)
    latent_d = torch.cdist(latents, latents)
    input_d = input_d / input_d.detach().mean().clamp_min(eps)
    latent_d = latent_d / latent_d.detach().mean().clamp_min(eps)
    weights = persistence_edge_weights(
        input_d, knn=knn, h0_weight=h0_weight, h1_weight=h1_weight
    )
    return (weights * (latent_d - input_d.detach()).square()).sum() / weights.sum().clamp_min(eps)


def blended_pwgeo_loss(
    frames: torch.Tensor,
    latents: torch.Tensor,
    *,
    blend: float = 0.25,
    knn: int = 5,
    h0_weight: float = 2.0,
    h1_weight: float = 1.0,
) -> torch.Tensor:
    """Retain global GeoAE geometry while emphasizing critical sparse edges."""
    blend = float(blend)
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"PW-GeoAE blend must lie in [0, 1], got {blend}")
    global_loss = geometry_proxy_loss(frames, latents)
    if blend == 0.0:
        return global_loss
    critical_loss = persistence_weighted_geometry_loss(
        frames,
        latents,
        knn=knn,
        h0_weight=h0_weight,
        h1_weight=h1_weight,
    )
    return (1.0 - blend) * global_loss + blend * critical_loss


def _paths(model_namespace, tag):
    root = Path("models") / model_namespace
    encoder_dir = root / "pwgeoae_encoders"
    decoder_dir = root / "pwgeoae_decoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    decoder_dir.mkdir(parents=True, exist_ok=True)
    return encoder_dir / f"encoder_{tag}.pt", decoder_dir / f"decoder_{tag}.pt"


def load_or_train_pwgeo_encoder(x_train, **kwargs):
    encoder, _, encoder_path, _ = load_or_train_pwgeo_autoencoder(x_train, **kwargs)
    return encoder, encoder_path


def load_or_train_pwgeo_autoencoder(
    x_train,
    *,
    dataset_name: str,
    model_namespace: str,
    seed: int,
    latent_dim: int,
    pw_lambda: float = 0.1,
    blend: float = 0.25,
    knn: int = 5,
    h0_weight: float = 2.0,
    h1_weight: float = 1.0,
    ae_epochs: int = 3,
    frame_batch_size: int = 256,
    max_frames_per_epoch: int | None = 8192,
    pair_batch_size: int = 64,
    retrain: bool = False,
    learning_rate: float = 1e-3,
    decoder_type: str = "mlp",
):
    encoder = SpatialEncoder(latent_dim=latent_dim)
    decoder = make_spatial_decoder(decoder_type, latent_dim=latent_dim, output_size=x_train.shape[-2:])
    tag = _model_tag(seed, x_train, latent_dim, pw_lambda, blend, knn, h0_weight, h1_weight, decoder_type)
    encoder_path, decoder_path = _paths(model_namespace, tag)
    device = ml_tda.get_runtime_device()

    if encoder_path.exists() and decoder_path.exists() and not retrain:
        print(f"Loading PW-GeoAE: {encoder_path} | {decoder_path}")
        encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
    else:
        print(f"Training PW-GeoAE: {encoder_path} | {decoder_path}")
        encoder.to(device)
        decoder.to(device)
        optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
        recon_fn = nn.MSELoss()
        all_frames = x_train.reshape(-1, *x_train.shape[2:])
        batch_size = min(int(frame_batch_size), len(all_frames))
        pair_batch_size = max(2, int(pair_batch_size))
        for epoch in tqdm_progress_bar(range(1, int(ae_epochs) + 1), desc="PW-GeoAE epochs", total=int(ae_epochs), leave=True):
            encoder.train(); decoder.train()
            generator = torch.Generator().manual_seed(seed * 10_000 + epoch)
            order = torch.randperm(len(all_frames), generator=generator)
            if max_frames_per_epoch is not None:
                order = order[:int(max_frames_per_epoch)]
            total_recon = total_pw = total_seen = 0.0
            for start in range(0, len(order), batch_size):
                frames = ml_tda.tensor_to_model_float(all_frames[order[start:start + batch_size]]).to(device)
                optimizer.zero_grad()
                z = encoder(frames)
                recon = recon_fn(decoder(z), frames)
                if len(frames) > pair_batch_size:
                    chosen = torch.randperm(len(frames), device=device)[:pair_batch_size]
                    pw = blended_pwgeo_loss(
                        frames[chosen], z[chosen], blend=blend, knn=knn,
                        h0_weight=h0_weight, h1_weight=h1_weight
                    )
                else:
                    pw = blended_pwgeo_loss(
                        frames, z, blend=blend, knn=knn,
                        h0_weight=h0_weight, h1_weight=h1_weight
                    )
                loss = recon + float(pw_lambda) * pw
                loss.backward(); optimizer.step()
                seen = len(frames)
                total_recon += float(recon) * seen
                total_pw += float(pw) * seen
                total_seen += seen
            print(f"PW-GeoAE epoch {epoch}/{ae_epochs} | recon_mse={total_recon/total_seen:.4f} pw_geo={total_pw/total_seen:.4f}")
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)

    encoder.to(device).eval(); decoder.to(device).eval()
    for parameter in list(encoder.parameters()) + list(decoder.parameters()):
        parameter.requires_grad = False
    return encoder, decoder, encoder_path, decoder_path
