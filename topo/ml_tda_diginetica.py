"""Prepare Diginetica sessions for leakage-free next-item graph forecasting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch


def load_diginetica(config: dict, force_rebuild: bool = False) -> dict:
    """Build a train-only item-transition graph and chronological session splits.

    Sessions are split by start time before vocabulary selection.  The retained
    vocabulary contains only the most frequent training items, and graph edges
    are likewise learned only from training-session transitions.
    """
    root = Path(config.get("root", "datasets/2D/diginetica"))
    cache = Path(config.get("cache_path", root / "processed/diginetica.pt"))
    if cache.exists() and not force_rebuild:
        return torch.load(cache, map_location="cpu", weights_only=False)
    source = root / "train-item-views.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing Diginetica views: {source}")

    views = pd.read_csv(
        source, sep=";", usecols=["sessionId", "itemId", "timeframe", "eventdate"]
    )
    views["timestamp"] = pd.to_datetime(views["eventdate"]).astype("int64") // 10**6
    views["timestamp"] += views["timeframe"].astype("int64")
    views = views.sort_values(["sessionId", "timestamp"], kind="stable")
    grouped = []
    for session_id, group in views.groupby("sessionId", sort=False):
        items = group["itemId"].astype("int64").tolist()
        collapsed = [items[0]]
        collapsed.extend(item for prior, item in zip(items, items[1:]) if item != prior)
        grouped.append((int(group["timestamp"].iloc[0]), int(session_id), collapsed))
    grouped.sort(key=lambda row: (row[0], row[1]))

    train_end = int(len(grouped) * float(config.get("train_fraction", 0.70)))
    val_end = train_end + int(len(grouped) * float(config.get("val_fraction", 0.15)))
    raw_train, raw_val, raw_test = grouped[:train_end], grouped[train_end:val_end], grouped[val_end:]
    counts: dict[int, int] = {}
    for _, _, path in raw_train:
        for item in path:
            counts[item] = counts.get(item, 0) + 1
    max_items = int(config.get("max_items", 5_000))
    vocabulary = [item for item, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:max_items]]
    item_to_node = {item: index for index, item in enumerate(vocabulary)}
    min_length = int(config.get("min_path_length", 6))

    def encode(records):
        paths = []
        for _, _, raw_path in records:
            path = [item_to_node[item] for item in raw_path if item in item_to_node]
            if not path:
                continue
            collapsed = [path[0]]
            collapsed.extend(node for prior, node in zip(path, path[1:]) if node != prior)
            if len(collapsed) >= min_length:
                paths.append(torch.tensor(collapsed, dtype=torch.long))
        return paths

    train_paths, val_paths, test_paths = map(encode, (raw_train, raw_val, raw_test))
    edges = set()
    for path in train_paths:
        for source_node, target_node in zip(path.tolist(), path.tolist()[1:]):
            edges.add((source_node, target_node)); edges.add((target_node, source_node))
    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    payload = {
        "item_ids": vocabulary, "edge_index": edge_index, "num_nodes": len(vocabulary),
        "train_paths": train_paths, "val_paths": val_paths, "test_paths": test_paths,
        "min_path_length": min_length,
        "split": "chronological_sessions_70_15_15_train_only_vocabulary_and_graph",
        "source_sessions": len(grouped),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    return payload


def validate_diginetica(payload: dict) -> dict[str, int | str]:
    """Validate node ranges and summarize the processed Diginetica splits."""
    n, edge_index = int(payload["num_nodes"]), payload["edge_index"]
    assert edge_index.shape[0] == 2 and edge_index.min() >= 0 and edge_index.max() < n
    return {
        "items": n, "training_edges_bidirectional": int(edge_index.shape[1]),
        "train_sessions": len(payload["train_paths"]), "val_sessions": len(payload["val_paths"]),
        "test_sessions": len(payload["test_paths"]), "split": payload["split"],
    }
