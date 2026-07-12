"""Persistence-regularized autoencoder helpers.

This module implements the `topo_*` autoencoder used by the terminal runner.
The loss is Rieck-style in the practical sense used by topological
autoencoders: compute persistence-critical pairs on the current minibatch,
then optimize the latent distances for those pairs with PyTorch.

The implemented signature is H0 Vietoris-Rips persistence. For a point cloud,
H0 death times are exactly the edge lengths of its minimum spanning tree, so we
can recover persistence-critical pairs without a separate differentiable TDA
backend. `signature` compares critical-pair distances directly; `wasserstein`
compares sorted H0 death-time vectors.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from topo import ml_tda
from topo.ml_tda import SpatialDecoder, SpatialEncoder, make_spatial_decoder


TOPO_AE_DISTANCES = {"signature", "wasserstein"}


def _decoder_suffix(decoder_type: str = "mlp") -> str:
    decoder_type = str(decoder_type).lower().strip()
    return "" if decoder_type == "mlp" else f"_dec{decoder_type}"


def topo_encoder_path(
    model_namespace: str,
    seed: int,
    x_train,
    latent_dim: int,
    topo_lambda: float,
    topo_distance: str,
    decoder_type: str = "mlp",
) -> Path:
    encoder_dir = Path("models") / model_namespace / "topoae_encoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_T{x_train.shape[0]}_B{x_train.shape[1]}_"
        f"H{x_train.shape[-2]}_W{x_train.shape[-1]}_latent{latent_dim}_"
        f"lambda{topo_lambda:g}_dist{topo_distance}{_decoder_suffix(decoder_type)}"
    )
    return encoder_dir / f"encoder_{tag}.pt"


def topo_decoder_path(
    model_namespace: str,
    seed: int,
    x_train,
    latent_dim: int,
    topo_lambda: float,
    topo_distance: str,
    decoder_type: str = "mlp",
) -> Path:
    decoder_dir = Path("models") / model_namespace / "topoae_decoders"
    decoder_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_T{x_train.shape[0]}_B{x_train.shape[1]}_"
        f"H{x_train.shape[-2]}_W{x_train.shape[-1]}_latent{latent_dim}_"
        f"lambda{topo_lambda:g}_dist{topo_distance}{_decoder_suffix(decoder_type)}"
    )
    return decoder_dir / f"decoder_{tag}.pt"


def _validate_topo_distance(topo_distance: str) -> str:
    topo_distance = topo_distance.lower().strip()
    if topo_distance not in TOPO_AE_DISTANCES:
        valid = ", ".join(sorted(TOPO_AE_DISTANCES))
        raise ValueError(f"Unknown topo AE distance {topo_distance!r}. Valid: {valid}")
    return topo_distance


def _normalized_distances(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    distances = torch.cdist(x, x, p=2)
    scale = distances.detach().mean().clamp_min(eps)
    return distances / scale


def _mst_edges_from_distances(distances: torch.Tensor) -> torch.Tensor:
    """Return MST edge indices for a dense distance matrix using Kruskal."""
    n = distances.shape[0]
    if n < 2:
        return torch.empty((0, 2), dtype=torch.long, device=distances.device)

    tri_i, tri_j = torch.triu_indices(n, n, offset=1, device=distances.device)
    order = torch.argsort(distances.detach()[tri_i, tri_j]).cpu().tolist()
    parent = list(range(n))
    rank = [0] * n
    edges: list[tuple[int, int]] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for edge_idx in order:
        a = int(tri_i[edge_idx].item())
        b = int(tri_j[edge_idx].item())
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            continue
        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1
        edges.append((a, b))
        if len(edges) == n - 1:
            break

    return torch.tensor(edges, dtype=torch.long, device=distances.device)


def _edge_values(distances: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if edges.numel() == 0:
        return distances.new_zeros((0,))
    return distances[edges[:, 0], edges[:, 1]]


def h0_persistence_loss(
    frames: torch.Tensor,
    latents: torch.Tensor,
    *,
    topo_distance: str = "signature",
) -> torch.Tensor:
    """Compare H0 VR persistence signatures of image-space and latent clouds."""
    topo_distance = _validate_topo_distance(topo_distance)
    if frames.shape[0] < 2:
        return latents.sum() * 0.0
    flat_frames = frames.flatten(start_dim=1)
    image_dist = _normalized_distances(flat_frames)
    latent_dist = _normalized_distances(latents)

    with torch.no_grad():
        image_edges = _mst_edges_from_distances(image_dist)
        latent_edges = _mst_edges_from_distances(latent_dist)
        image_deaths = _edge_values(image_dist, image_edges).detach()

    if topo_distance == "wasserstein":
        latent_deaths = _edge_values(latent_dist, latent_edges)
        return torch.mean((torch.sort(latent_deaths).values - torch.sort(image_deaths).values) ** 2)

    image_pair_loss = torch.mean((_edge_values(latent_dist, image_edges) - image_deaths) ** 2)
    with torch.no_grad():
        image_at_latent_edges = _edge_values(image_dist, latent_edges).detach()
    latent_pair_loss = torch.mean((_edge_values(latent_dist, latent_edges) - image_at_latent_edges) ** 2)
    return 0.5 * (image_pair_loss + latent_pair_loss)


def load_or_train_topo_encoder(
    x_train,
    *,
    dataset_name: str,
    model_namespace: str,
    seed: int,
    latent_dim: int,
    topo_lambda: float = 0.1,
    topo_distance: str = "signature",
    ae_epochs: int = 3,
    frame_batch_size: int = 256,
    max_frames_per_epoch: int | None = 8192,
    pair_batch_size: int = 64,
    retrain: bool = False,
    learning_rate: float = 1e-3,
    decoder_type: str = "mlp",
):
    """Load or train a persistence-regularized spatial encoder."""
    encoder, _, encoder_path, _ = load_or_train_topo_autoencoder(
        x_train,
        dataset_name=dataset_name,
        model_namespace=model_namespace,
        seed=seed,
        latent_dim=latent_dim,
        topo_lambda=topo_lambda,
        topo_distance=topo_distance,
        ae_epochs=ae_epochs,
        frame_batch_size=frame_batch_size,
        max_frames_per_epoch=max_frames_per_epoch,
        pair_batch_size=pair_batch_size,
        retrain=retrain,
        learning_rate=learning_rate,
        decoder_type=decoder_type,
    )
    return encoder, encoder_path


def load_or_train_topo_autoencoder(
    x_train,
    *,
    dataset_name: str,
    model_namespace: str,
    seed: int,
    latent_dim: int,
    topo_lambda: float = 0.1,
    topo_distance: str = "signature",
    ae_epochs: int = 3,
    frame_batch_size: int = 256,
    max_frames_per_epoch: int | None = 8192,
    pair_batch_size: int = 64,
    retrain: bool = False,
    learning_rate: float = 1e-3,
):
    """Load or train a persistence-regularized spatial autoencoder."""
    topo_distance = _validate_topo_distance(topo_distance)
    encoder = SpatialEncoder(latent_dim=latent_dim)
    decoder = make_spatial_decoder(decoder_type, latent_dim=latent_dim, output_size=x_train.shape[-2:])
    encoder_path = topo_encoder_path(
        model_namespace,
        seed,
        x_train,
        latent_dim,
        topo_lambda,
        topo_distance,
        decoder_type,
    )
    decoder_path = topo_decoder_path(
        model_namespace,
        seed,
        x_train,
        latent_dim,
        topo_lambda,
        topo_distance,
        decoder_type,
    )

    if encoder_path.exists() and decoder_path.exists() and not retrain:
        print(f"Loading topo-AE encoder: {encoder_path}")
        print(f"Loading topo-AE decoder: {decoder_path}")
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
        print(f"Loading topo-AE encoder: {encoder_path}")
        print(f"Topo-AE decoder missing; training decoder only: {decoder_path}")
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
    print(f"Topo-AE autoencoder {reason}: {encoder_path} | {decoder_path}")
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
        f"lambda={topo_lambda}, distance={topo_distance}, ae_epochs={ae_epochs}, device={device}"
    )

    for epoch in range(1, int(ae_epochs) + 1):
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
                topo_loss = h0_persistence_loss(
                    frames[pair_idx],
                    latents[pair_idx],
                    topo_distance=topo_distance,
                )
            else:
                topo_loss = h0_persistence_loss(frames, latents, topo_distance=topo_distance)

            loss = recon_loss + float(topo_lambda) * topo_loss
            loss.backward()
            optimizer.step()

            seen = len(frames)
            total_recon += recon_loss.item() * seen
            total_topo += topo_loss.item() * seen
            total_seen += seen

        print(
            f"Topo-AE epoch {epoch}/{ae_epochs} | "
            f"recon_mse={total_recon / total_seen:.4f} "
            f"h0_persistence={total_topo / total_seen:.4f}"
        )

    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    print(f"Saved topo-AE encoder: {encoder_path}")
    print(f"Saved topo-AE decoder: {decoder_path}")
    encoder.eval()
    decoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False
    return encoder, decoder, encoder_path, decoder_path
