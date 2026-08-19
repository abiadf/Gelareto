"""Load Wikispeedia as a fixed graph with chronological navigation sequences.

The graph is represented by a memory-efficient directed ``edge_index`` rather
than a dense adjacency matrix. Finished human paths are resolved into sequences
of node IDs (including back-button moves) and split chronologically into
train/validation/test sets; the graph itself is shared across all three splits.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

import torch


def _data_lines(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                yield line.rstrip("\n")


def _resolve_path(encoded_path: str, node_to_id: dict[str, int]) -> list[int]:
    """Convert one path to visited node IDs, resolving ``<`` as a back click."""
    browser_stack: list[int] = []
    visited: list[int] = []
    for token in encoded_path.split(";"):
        if token == "<":
            if len(browser_stack) > 1:
                browser_stack.pop()
                visited.append(browser_stack[-1])
            continue
        node_id = node_to_id.get(token)
        if node_id is None:
            continue
        browser_stack.append(node_id)
        visited.append(node_id)
    return visited


def load_wikispeedia(config: dict, force_rebuild: bool = False) -> dict:
    """Return the fixed hyperlink graph and leakage-safe chronological paths.

    The returned mapping contains ``edge_index`` with shape ``(2, E)``, article
    names, and variable-length ``train_paths``, ``val_paths``, and ``test_paths``.
    Full graph access makes this a transductive next-node forecasting benchmark;
    only navigation sequences, never test outcomes, are split by time.
    """
    root = Path(config.get("root", "datasets/2D/wikispedia/wikispeedia_paths-and-graph"))
    cache_path = Path(config.get("cache_path", "datasets/2D/wikispedia/processed/wikispeedia.pt"))
    if cache_path.exists() and not force_rebuild:
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    required = [root / "articles.tsv", root / "links.tsv", root / "paths_finished.tsv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Wikispeedia files: {missing}")

    encoded_articles = [line.split("\t", 1)[0] for line in _data_lines(required[0])]
    node_to_id = {name: idx for idx, name in enumerate(encoded_articles)}

    edges: list[tuple[int, int]] = []
    for line in _data_lines(required[1]):
        source, target = line.split("\t")[:2]
        if source in node_to_id and target in node_to_id:
            edges.append((node_to_id[source], node_to_id[target]))

    min_path_length = int(config.get("min_path_length", 6))
    records: list[tuple[int, list[int]]] = []
    for line in _data_lines(required[2]):
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        path = _resolve_path(fields[3], node_to_id)
        if len(path) >= min_path_length:
            records.append((int(fields[1]), path))
    records.sort(key=lambda item: item[0])

    train_fraction = float(config.get("train_fraction", 0.70))
    val_fraction = float(config.get("val_fraction", 0.15))
    train_end = int(len(records) * train_fraction)
    val_end = train_end + int(len(records) * val_fraction)
    paths = [torch.tensor(path, dtype=torch.long) for _, path in records]

    payload = {
        "articles": [unquote(name) for name in encoded_articles],
        "edge_index": torch.tensor(edges, dtype=torch.long).t().contiguous(),
        "train_paths": paths[:train_end],
        "val_paths": paths[train_end:val_end],
        "test_paths": paths[val_end:],
        "split": "chronological_70_15_15",
        "min_path_length": min_path_length,
        "num_nodes": len(encoded_articles),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def validate_wikispeedia(payload: dict) -> dict[str, int | str]:
    """Validate node ranges, split sizes, and path/edge tensor shapes."""
    num_nodes = int(payload["num_nodes"])
    edge_index = payload["edge_index"]
    assert edge_index.ndim == 2 and edge_index.shape[0] == 2
    assert edge_index.numel() == 0 or (edge_index.min() >= 0 and edge_index.max() < num_nodes)
    for split in ("train_paths", "val_paths", "test_paths"):
        assert payload[split]
        assert all(path.ndim == 1 and len(path) >= payload["min_path_length"] for path in payload[split])
        assert all(path.min() >= 0 and path.max() < num_nodes for path in payload[split])
    return {
        "nodes": num_nodes,
        "directed_edges": int(edge_index.shape[1]),
        "train_paths": len(payload["train_paths"]),
        "val_paths": len(payload["val_paths"]),
        "test_paths": len(payload["test_paths"]),
        "split": payload["split"],
    }
