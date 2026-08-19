"""CPU-friendly synthetic videos with explicitly tree-like input geometry.

Each frame represents a node by the edges on its root-to-node path. The left
half is a path-incidence code: every possible tree edge owns one fixed pixel.
Consequently, squared Euclidean distance in that panel is proportional to tree
shortest-path distance. The right half contains a faint, human-readable drawing
of the same growing route. Clips traverse different root-to-leaf paths and use
a small next-branch cue so multi-step forecasting is well posed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def _edge_indices(choices: np.ndarray, branching_factor: int = 6) -> list[int]:
    """Return heap-indexed edges along one regular-tree root-to-node path."""
    node = 0
    edges = []
    for choice in choices:
        child = int(branching_factor) * node + 1 + int(choice)
        edges.append(child - 1)
        node = child
    return edges


def _route_points(
    choices: np.ndarray, rng: np.random.Generator, branching_factor: int
) -> list[tuple[int, int]]:
    """Create a non-straight, non-crossing visual route in the right panel."""
    x, y = 48.0, 62.0
    points = [(int(round(x)), int(round(y)))]
    center = (int(branching_factor) - 1) / 2.0
    for depth, choice in enumerate(choices):
        horizontal = max(0.7, 2.8 * (0.72**depth))
        x += (int(choice) - center) * horizontal
        x += rng.normal(0.0, 0.25)
        x = float(np.clip(x, 34.0, 62.0))
        y -= 6.4
        points.append((int(round(x)), int(round(y))))
    return points


def generate_growing_tree_split(
    *,
    num_clips: int,
    clip_len: int = 64,
    depth: int = 5,
    branching_factor: int = 4,
    image_size: tuple[int, int] = (64, 64),
    seed_offset: int = 0,
    path_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Generate ``(time, clips, height, width)`` uint8 growing-tree videos."""
    height, width = map(int, image_size)
    if (height, width) != (64, 64):
        raise ValueError("growing_tree currently requires image_size=(64, 64)")
    branching_factor = int(branching_factor)
    if branching_factor < 2:
        raise ValueError("branching_factor must be at least 2")
    n_nodes = (branching_factor ** (int(depth) + 1) - 1) // (branching_factor - 1)
    n_edges = n_nodes - 1
    code_capacity = height * (width // 2) - branching_factor
    if n_edges > code_capacity:
        raise ValueError(f"depth={depth} needs {n_edges} edge codes; capacity is {code_capacity}")
    if clip_len < depth + 1:
        raise ValueError("clip_len must exceed tree depth")

    output = np.zeros((int(clip_len), int(num_clips), height, width), dtype=np.uint8)
    n_leaf_paths = branching_factor**int(depth)
    if path_ids is None and int(seed_offset) + int(num_clips) > n_leaf_paths:
        raise ValueError(f"Requested paths exceed the {n_leaf_paths} unique root-to-leaf paths")
    if path_ids is not None and len(path_ids) != int(num_clips):
        raise ValueError("path_ids must contain exactly num_clips entries")
    for clip in range(int(num_clips)):
        path_id = int(path_ids[clip]) if path_ids is not None else int(seed_offset) + clip
        rng = np.random.default_rng(path_id)
        # A clip-level temporal warp is inferable from its observed history but
        # prevents every sequence from reaching branch points simultaneously.
        time_exponent = float(rng.uniform(0.75, 1.35))
        choices = np.empty(int(depth), dtype=np.int16)
        remainder = path_id
        for position in range(int(depth) - 1, -1, -1):
            choices[position] = remainder % branching_factor
            remainder //= branching_factor
        edges = _edge_indices(choices, branching_factor)
        route = _route_points(choices, rng, branching_factor)
        for time in range(int(clip_len)):
            frame = output[time, clip]
            normalized_time = time / max(1, int(clip_len) - 1)
            progress = min(float(depth), float(depth) * normalized_time**time_exponent)
            complete = min(int(depth), int(np.floor(progress + 1e-9)))
            phase = progress - complete

            # Exact path-incidence panel. Each edge owns one pixel in columns 0:32.
            flat_code = np.zeros(height * 32, dtype=np.uint8)
            for edge in edges[:complete]:
                flat_code[edge] = 255
            if complete < depth:
                edge = edges[complete]
                flat_code[edge] = np.uint8(round(255 * phase))

                # Low-energy cue: which child will be taken at the next split.
                cue_index = height * 32 - branching_factor + int(choices[complete])
                # Deliberately subtle: informative, but no longer an easy shortcut.
                flat_code[cue_index] = 20
            frame[:, :32] = flat_code.reshape(height, 32)

            # Faint natural rendering; the incidence panel controls the metric.
            for edge_depth in range(complete):
                cv2.line(frame, route[edge_depth], route[edge_depth + 1], 48, 1, cv2.LINE_AA)
            if complete < depth and phase > 0:
                start = np.asarray(route[complete], dtype=np.float32)
                end = np.asarray(route[complete + 1], dtype=np.float32)
                partial = tuple(np.rint(start + phase * (end - start)).astype(int))
                cv2.line(frame, route[complete], partial, 48, 1, cv2.LINE_AA)
            cv2.circle(frame, route[min(complete, depth)], 1, 96, -1)
    return output


def _cache_path(config: dict, split: str, num_clips: int, seed: int) -> Path:
    cache_dir = Path(config.get("cache_dir", "datasets/2D/growing_tree/processed"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    height, width = config.get("image_size", (64, 64))
    return cache_dir / (
        f"v4_{split}_N{num_clips}_T{config.get('clip_len', 64)}_D{config.get('tree_depth', 5)}_"
        f"B{config.get('branching_factor', 4)}_"
        f"H{height}_W{width}_seed{seed}.pt"
    )


def _load_or_generate(
    config: dict, split: str, num_clips: int, seed: int, path_ids: np.ndarray
) -> np.ndarray:
    path = _cache_path(config, split, num_clips, seed)
    if path.exists() and not config.get("force_rebuild_cache", False):
        print(f"Loading cached growing-tree {split}: {path}")
        return torch.load(path, map_location="cpu").numpy()
    array = generate_growing_tree_split(
        num_clips=num_clips,
        clip_len=int(config.get("clip_len", 64)),
        depth=int(config.get("tree_depth", 5)),
        branching_factor=int(config.get("branching_factor", 4)),
        image_size=tuple(config.get("image_size", (64, 64))),
        seed_offset=seed,
        path_ids=path_ids,
    )
    torch.save(torch.from_numpy(array), path)
    print(f"Saved growing-tree {split}: {path} shape={array.shape}")
    return array


def load_growing_tree(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load disjoint, subtree-balanced train/test growing-tree clips."""
    branching = int(config.get("branching_factor", 4))
    depth = int(config.get("tree_depth", 5))
    n_paths = branching**depth
    split_rng = np.random.default_rng(int(config.get("path_split_seed", 2026)))
    shuffled_paths = split_rng.permutation(n_paths)
    n_train = int(config.get("num_train_clips", 512))
    n_test = int(config.get("num_test_clips", 128))
    if n_train + n_test > n_paths:
        raise ValueError(f"Requested {n_train + n_test} clips but only {n_paths} paths exist")
    train = _load_or_generate(
        config,
        "train",
        n_train,
        int(config.get("train_seed_offset", 0)),
        shuffled_paths[:n_train],
    )
    test = _load_or_generate(
        config,
        "test",
        n_test,
        int(config.get("test_seed_offset", config.get("num_train_clips", 512))),
        shuffled_paths[n_train:n_train + n_test],
    )
    return train, test
