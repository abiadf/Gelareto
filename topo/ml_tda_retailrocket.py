"""Prepare Retailrocket as real user sequences over a product-category tree.

Items are mapped to their latest published category assignment. User events are
ordered by time, split at 30-minute inactivity gaps, and collapsed when two
consecutive events visit the same category. Sessions are then split globally by
start time into train/validation/test sets. The category tree is stored as a
compact bidirectional edge list rather than a dense adjacency matrix.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch


def _item_categories(root: Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Extract the most recent category assignment for every item in chunks."""
    pieces = []
    for path in sorted(root.glob("item_properties_part*.csv")):
        for chunk in pd.read_csv(
            path, usecols=["timestamp", "itemid", "property", "value"],
            dtype={"property": "string", "value": "string"}, chunksize=chunksize,
        ):
            category = chunk.loc[chunk["property"] == "categoryid", ["timestamp", "itemid", "value"]]
            category = category[category["value"].str.fullmatch(r"\d+", na=False)].copy()
            if not category.empty:
                category["categoryid"] = category.pop("value").astype("int32")
                pieces.append(category)
    if not pieces:
        raise ValueError("No categoryid item properties found")
    assignments = pd.concat(pieces, ignore_index=True)
    return assignments.sort_values("timestamp").drop_duplicates("itemid", keep="last")


def load_retailrocket(config: dict, force_rebuild: bool = False) -> dict:
    """Return the category graph and chronological category-navigation sessions."""
    root = Path(config.get("root", "datasets/2D/retailrocket"))
    cache_path = Path(config.get("cache_path", root / "processed/retailrocket.pt"))
    if cache_path.exists() and not force_rebuild:
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    required = [root / "category_tree.csv", root / "events.csv"]
    if any(not path.exists() for path in required):
        raise FileNotFoundError(f"Retailrocket requires: {[str(path) for path in required]}")

    tree = pd.read_csv(required[0]).dropna(subset=["categoryid"])
    tree["categoryid"] = tree["categoryid"].astype("int32")
    categories = _item_categories(root)
    events = pd.read_csv(required[1], usecols=["timestamp", "visitorid", "event", "itemid"])
    if config.get("event_types"):
        events = events[events["event"].isin(config["event_types"])]
    events = events.merge(categories[["itemid", "categoryid"]], on="itemid", how="inner")
    events = events.sort_values(["visitorid", "timestamp"], kind="stable")

    all_categories = sorted(set(tree["categoryid"]) | set(tree["parentid"].dropna().astype(int)) | set(events["categoryid"]))
    category_to_node = {category: index for index, category in enumerate(all_categories)}
    events["node"] = events["categoryid"].map(category_to_node).astype("int32")
    gap_ms = int(config.get("session_gap_minutes", 30)) * 60 * 1000
    new_session = (
        events["visitorid"].ne(events["visitorid"].shift())
        | events["timestamp"].sub(events["timestamp"].shift()).gt(gap_ms)
    )
    events["session"] = new_session.cumsum()

    min_length = int(config.get("min_path_length", 6))
    sessions = []
    for _, group in events.groupby("session", sort=False):
        nodes = group["node"].tolist()
        collapsed = [nodes[0]]
        collapsed.extend(node for previous, node in zip(nodes, nodes[1:]) if node != previous)
        if len(collapsed) >= min_length:
            sessions.append((int(group["timestamp"].iloc[0]), torch.tensor(collapsed, dtype=torch.long)))
    sessions.sort(key=lambda record: record[0])

    edges = []
    for row in tree.dropna(subset=["parentid"]).itertuples(index=False):
        child, parent = category_to_node[int(row.categoryid)], category_to_node[int(row.parentid)]
        edges.extend(((child, parent), (parent, child)))
    # Retailrocket publishes a forest of top-level taxonomies. A non-predictive
    # virtual root joins those roots into one connected hierarchy for distances.
    virtual_root = len(all_categories)
    child_categories = set(tree["categoryid"].astype(int))
    parent_categories = set(tree["parentid"].dropna().astype(int))
    roots = sorted(parent_categories - child_categories)
    roots.extend(int(value) for value in tree.loc[tree["parentid"].isna(), "categoryid"])
    for category in sorted(set(roots)):
        node = category_to_node[category]
        edges.extend(((virtual_root, node), (node, virtual_root)))
    train_end = int(len(sessions) * float(config.get("train_fraction", 0.70)))
    val_end = train_end + int(len(sessions) * float(config.get("val_fraction", 0.15)))
    paths = [path for _, path in sessions]
    payload = {
        "category_ids": all_categories + ["__virtual_root__"],
        "edge_index": torch.tensor(edges, dtype=torch.long).t().contiguous(),
        "train_paths": paths[:train_end], "val_paths": paths[train_end:val_end],
        "test_paths": paths[val_end:], "num_nodes": len(all_categories) + 1,
        "virtual_root": virtual_root,
        "min_path_length": min_length, "split": "chronological_sessions_70_15_15",
        "session_gap_minutes": int(config.get("session_gap_minutes", 30)),
        "mapped_events": int(len(events)),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def validate_retailrocket(payload: dict) -> dict[str, int | str]:
    """Check graph ranges and report processed split sizes."""
    n = int(payload["num_nodes"])
    edge_index = payload["edge_index"]
    assert edge_index.shape[0] == 2 and edge_index.min() >= 0 and edge_index.max() < n
    for split in ("train_paths", "val_paths", "test_paths"):
        assert all(len(path) >= payload["min_path_length"] for path in payload[split])
        assert all(path.min() >= 0 and path.max() < n for path in payload[split])
    return {
        "categories": n, "tree_edges_bidirectional": int(edge_index.shape[1]),
        "mapped_events": int(payload["mapped_events"]),
        "train_sessions": len(payload["train_paths"]),
        "val_sessions": len(payload["val_paths"]),
        "test_sessions": len(payload["test_paths"]), "split": payload["split"],
    }
