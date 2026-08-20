"""Lightweight next-page forecasting baseline for Wikispeedia paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ripser import ripser
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from topo.ml_tda_wikispeedia import load_wikispeedia, validate_wikispeedia
from topo.ml_tda_retailrocket import load_retailrocket, validate_retailrocket
from topo.ml_tda_diginetica import load_diginetica, validate_diginetica
from topo.ml_tda_latent import _finite_diagram, betti_curve_from_diag
from topo.ml_tda_manifold_ph import vr_distance_matrix
from topo.ml_tda_mixedgeo import product_pairwise_distances
from topo.ml_tda_triangle import gu_triangle_curvature_samples_from_adjacency, _diagnostic_summary


class WikiPathPredictor(nn.Module):
    """Embed page IDs, summarize five visited pages with a GRU, and predict the next page."""

    def __init__(self, num_nodes: int, latent_dim: int, hidden_dim: int, topo_dim: int = 0):
        super().__init__()
        self.topo_dim = int(topo_dim)
        self.encoder = nn.Embedding(num_nodes, latent_dim)
        self.sequence_model = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.next_page = nn.Linear(hidden_dim + self.topo_dim, num_nodes)

    def forward(self, page_ids: torch.Tensor, topology: torch.Tensor | None = None) -> torch.Tensor:
        _, hidden = self.sequence_model(self.encoder(page_ids))
        features = hidden[-1]
        if self.topo_dim:
            if topology is None:
                raise ValueError("This predictor requires persistence features")
            features = torch.cat([features, topology], dim=1)
        return self.next_page(features)


def _windows(paths: list[torch.Tensor], context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    contexts, targets = [], []
    for path in paths:
        for target_index in range(context_length, len(path)):
            contexts.append(path[target_index - context_length : target_index])
            targets.append(path[target_index])
    return torch.stack(contexts), torch.stack(targets)


def _cap_examples(x: torch.Tensor, y: torch.Tensor, limit: int | None, seed: int):
    if limit is None or len(x) <= limit:
        return x, y
    indices = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))[:limit]
    return x[indices], y[indices]


@torch.no_grad()
def _evaluate(model, x, y, device: torch.device, batch_size: int, topology=None) -> dict[str, float]:
    model.eval()
    total = correct1 = correct5 = 0
    reciprocal_rank = loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    tensors = (x, y) if topology is None else (x, y, topology)
    for batch in DataLoader(TensorDataset(*tensors), batch_size=batch_size):
        xb, yb = batch[0].to(device), batch[1].to(device)
        tb = None if topology is None else batch[2].to(device)
        logits = model(xb, tb)
        loss_sum += float(criterion(logits, yb))
        correct1 += int((logits.argmax(1) == yb).sum())
        correct5 += int((logits.topk(5, dim=1).indices == yb[:, None]).any(1).sum())
        target_scores = logits.gather(1, yb[:, None])
        ranks = 1 + (logits > target_scores).sum(1)
        reciprocal_rank += float((1.0 / ranks.float()).sum())
        total += len(yb)
    return {
        "cross_entropy": loss_sum / total,
        "top1_accuracy": correct1 / total,
        "top5_accuracy": correct5 / total,
        "mrr": reciprocal_rank / total,
    }


def _load_graph_distances(root: str | Path, num_nodes: int, edge_index=None) -> torch.Tensor:
    """Load Wikispeedia distances or compute them from a smaller edge list."""
    rows = []
    path = Path(root) / "shortest-path-distance-matrix.txt"
    if not path.exists():
        if edge_index is None:
            raise FileNotFoundError(f"No graph distances or edge_index for {root}")
        edges = edge_index.cpu().numpy()
        adjacency = csr_matrix(
            (np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(num_nodes, num_nodes)
        )
        distances = shortest_path(adjacency, directed=False, unweighted=True)
        finite_max = distances[np.isfinite(distances)].max()
        distances[~np.isfinite(distances)] = finite_max + 1
        return torch.from_numpy(distances.astype(np.float32))
    with path.open(encoding="ascii") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            row = np.frombuffer(line.strip().encode("ascii"), dtype=np.uint8) - ord("0")
            row[row > 9] = 127  # '_' denotes unreachable in the directed graph.
            rows.append(row)
    distances = np.stack(rows).astype(np.float32)
    if distances.shape != (num_nodes, num_nodes):
        raise ValueError(f"Unexpected shortest-path matrix shape {distances.shape}")
    distances = np.minimum(distances, distances.T)
    finite_max = distances[distances < 127].max()
    distances[distances >= 127] = finite_max + 1
    return torch.from_numpy(distances)


def _geometry_loss(model, node_ids, graph_distances, mode: str, signature: str) -> torch.Tensor:
    z = model.encoder(node_ids)
    target = graph_distances[node_ids.cpu()][:, node_ids.cpu()].to(z.device)
    latent = torch.cdist(z, z) if mode == "graph_geo" else product_pairwise_distances(z, signature)
    upper = torch.triu_indices(len(node_ids), len(node_ids), offset=1, device=z.device)
    target_values = target[upper[0], upper[1]]
    latent_values = latent[upper[0], upper[1]]
    target_values = target_values / target_values.mean().clamp_min(1e-6)
    latent_values = latent_values / latent_values.detach().mean().clamp_min(1e-6)
    return (target_values - latent_values).square().mean()


@torch.no_grad()
def _persistence_features(model, contexts, signature: str, bins: int, batch_notice: int = 10_000):
    """Compute H0/H1 Betti-curve features from each five-page manifold point cloud."""
    model.eval()
    result = np.zeros((len(contexts), 2 * bins), dtype=np.float32)
    embeddings = model.encoder.weight.detach().cpu()
    for index, context in enumerate(contexts):
        points = embeddings[context]
        matrix = vr_distance_matrix(points, signature, vr_distance="product_manifold")
        diagrams = ripser(matrix, maxdim=1, distance_matrix=True)["dgms"]
        h0 = betti_curve_from_diag(_finite_diagram(diagrams[0]), n_bins=bins)
        h1_dgm = _finite_diagram(diagrams[1]) if len(diagrams) > 1 else np.empty((0, 2))
        h1 = betti_curve_from_diag(h1_dgm, n_bins=bins)
        result[index] = np.concatenate([h0, h1])
        if batch_notice and (index + 1) % batch_notice == 0:
            print(f"Computed persistence for {index + 1}/{len(contexts)} paths")
    return torch.from_numpy(result)


def run_graph_next_node(cfg, dataset_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train the four lightweight modes on Wikispeedia or Retailrocket paths."""
    dataset_name = cfg.dataset
    if dataset_config["kind"] == "retailrocket":
        data = load_retailrocket(dataset_config, force_rebuild=cfg.force_rebuild_data_cache)
        print("Retailrocket:", validate_retailrocket(data))
    elif dataset_config["kind"] == "diginetica":
        data = load_diginetica(dataset_config, force_rebuild=cfg.force_rebuild_data_cache)
        print("Diginetica:", validate_diginetica(data))
    else:
        data = load_wikispeedia(dataset_config, force_rebuild=cfg.force_rebuild_data_cache)
        print("Wikispeedia:", validate_wikispeedia(data))
    context_length = int(cfg.latent_tda_window or dataset_config.get("LATENT_TDA_WINDOW", 5))
    if int(cfg.horizon or dataset_config.get("HORIZON", 1)) != 1:
        raise ValueError("The graph next-item runner supports --horizon 1 only.")
    train_x, train_y = _windows(data["train_paths"], context_length)
    val_x, val_y = _windows(data["val_paths"], context_length)
    test_x, test_y = _windows(data["test_paths"], context_length)

    seeds = cfg.run_seeds or dataset_config.get("RUN_SEEDS", [0, 1, 2])
    latent_dim = int(cfg.latent_dim or dataset_config.get("LATENT_DIM", 16))
    hidden_dim = int(cfg.hidden_dim or dataset_config.get("HIDDEN_DIM", 64))
    epochs = int(cfg.predictor_epochs or dataset_config.get("PREDICTOR_EPOCHS", 8))
    lr = float(cfg.predictor_learning_rate or dataset_config.get("learning_rate", 3e-3))
    batch_size = int(dataset_config.get("BATCH_SIZE", 256))
    modes = cfg.modes or ["z"]
    valid_modes = {"z", "graph_geo", "graph_mixed", "graph_mixed_tda"}
    unknown = set(modes) - valid_modes
    if unknown:
        raise ValueError(f"Unknown graph next-item modes: {sorted(unknown)}")
    signature = str(dataset_config.get("MANIFOLD_SIGNATURE", "h8_e8"))
    geo_lambda = float(dataset_config.get("GRAPH_GEO_LAMBDA", cfg.geo_ae_lambda))
    pair_batch = int(dataset_config.get("GRAPH_GEO_PAIR_BATCH", 64))
    graph_distances = (
        _load_graph_distances(dataset_config["root"], data["num_nodes"], data["edge_index"])
        if any(mode != "z" for mode in modes) else None
    )
    device = torch.device("cpu" if cfg.device == "auto" else cfg.device)
    rows = []

    for seed in seeds:
        for mode in modes:
            torch.manual_seed(seed)
            x_seed, y_seed = _cap_examples(train_x, train_y, dataset_config.get("MAX_TRAIN_WINDOWS"), seed)
            model = WikiPathPredictor(data["num_nodes"], latent_dim, hidden_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            loader = DataLoader(
                TensorDataset(x_seed, y_seed), batch_size=batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            geometric_mode = "graph_mixed" if mode == "graph_mixed_tda" else mode
            for epoch in range(epochs):
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(xb), yb)
                    if geometric_mode in {"graph_geo", "graph_mixed"}:
                        node_ids = torch.randperm(data["num_nodes"])[:pair_batch].to(device)
                        loss = loss + geo_lambda * _geometry_loss(
                            model, node_ids, graph_distances, geometric_mode, signature
                        )
                    loss.backward()
                    optimizer.step()
                print(f"{dataset_name} mode={mode} seed={seed} epoch={epoch + 1}/{epochs}")

            train_topo = val_topo = test_topo = None
            if mode == "graph_mixed_tda":
                bins = int(dataset_config.get("TDA_BINS", 8))
                train_topo = _persistence_features(model, x_seed, signature, bins)
                val_topo = _persistence_features(model, val_x, signature, bins)
                test_topo = _persistence_features(model, test_x, signature, bins)
                fused = WikiPathPredictor(data["num_nodes"], latent_dim, hidden_dim, 2 * bins).to(device)
                fused.encoder.load_state_dict(model.encoder.state_dict())
                fused.sequence_model.load_state_dict(model.sequence_model.state_dict())
                with torch.no_grad():
                    fused.next_page.weight[:, :hidden_dim].copy_(model.next_page.weight)
                    fused.next_page.bias.copy_(model.next_page.bias)
                for parameter in list(fused.encoder.parameters()) + list(fused.sequence_model.parameters()):
                    parameter.requires_grad_(False)
                optimizer = torch.optim.Adam(fused.next_page.parameters(), lr=lr)
                fusion_loader = DataLoader(
                    TensorDataset(x_seed, y_seed, train_topo), batch_size=batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(seed + 17),
                )
                for _ in range(int(dataset_config.get("TDA_FUSION_EPOCHS", 3))):
                    for xb, yb, tb in fusion_loader:
                        optimizer.zero_grad(set_to_none=True)
                        loss = criterion(fused(xb.to(device), tb.to(device)), yb.to(device))
                        loss.backward()
                        optimizer.step()
                model = fused

            model_dir = Path(dataset_config.get("MODEL_DIR", f"models/{dataset_name}_next_node"))
            model_dir.mkdir(parents=True, exist_ok=True)
            signature_tag = f"_{signature}" if "mixed" in mode else ""
            model_path = model_dir / f"{mode}{signature_tag}_seed{seed}_ctx{context_length}_d{latent_dim}_h{hidden_dim}.pt"
            torch.save(model.state_dict(), model_path)
            val_metrics = _evaluate(model, val_x, val_y, device, batch_size, val_topo)
            test_metrics = (
                _evaluate(model, test_x, test_y, device, batch_size, test_topo)
                if not dataset_config.get("SKIP_TEST_EVALUATION", False) else {}
            )
            rows.append({
                "dataset": dataset_name, "seed": seed, "mode": mode, "horizon": 1,
                "context_length": context_length, "manifold_signature": signature if "mixed" in mode else "",
                "geo_lambda": geo_lambda if mode != "z" else 0.0,
                "n_train_windows": len(x_seed), "n_val_windows": len(val_x), "n_test_windows": len(test_x),
                **{f"val_{key}": value for key, value in val_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
                "model_path": str(model_path),
            })

    results = pd.DataFrame(rows)
    prefix = "test" if "test_cross_entropy" in results else "val"
    metrics = [f"{prefix}_cross_entropy", f"{prefix}_top1_accuracy", f"{prefix}_top5_accuracy", f"{prefix}_mrr"]
    summary = results.groupby("mode")[metrics].agg(["mean", "std"])
    print(f"\n{dataset_name} per-seed results:\n", results.to_string(index=False))
    print(f"\n{dataset_name} summary:\n", summary)
    return results, summary


def run_wikispeedia_next_node(cfg, dataset_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible wrapper for existing Wikispeedia commands."""
    return run_graph_next_node(cfg, dataset_config)


def run_wikispeedia_triangle(cfg, dataset_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Gu's triangle diagnostic directly on the fixed hyperlink graph."""
    data = load_wikispeedia(dataset_config, force_rebuild=cfg.force_rebuild_data_cache)
    edge_index = data["edge_index"].numpy()
    adjacency = csr_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(data["num_nodes"], data["num_nodes"]),
    )
    adjacency = adjacency.maximum(adjacency.T)
    seeds = cfg.run_seeds or [0, 1, 2]
    rows = []
    for seed in seeds:
        values, metadata = gu_triangle_curvature_samples_from_adjacency(
            adjacency, n_samples=cfg.triangle_samples, seed=seed
        )
        row = _diagnostic_summary(
            values, metadata, latent_dim=int(cfg.latent_dim or 16),
            flat_threshold=cfg.triangle_flat_threshold, seed=seed, knn=0,
        )
        row.update({"dataset": "wikispeedia", "scenario": "wikispeedia_triangle"})
        rows.append(row)
    results = pd.DataFrame(rows)
    metrics = ["k_mean", "negative_fraction", "flat_fraction", "positive_fraction"]
    summary = results.groupby("suggested_signature")[metrics].agg(["mean", "std"])
    print("\nWikispeedia triangle results:\n", results.to_string(index=False))
    print("\nWikispeedia triangle summary:\n", summary)
    return results, summary
