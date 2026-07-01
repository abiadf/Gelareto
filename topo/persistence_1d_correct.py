"""Stream 1D chunks and return exact prefix H0 persistence after each update."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _DSU:
    """Union-find with component birth values."""

    parent: np.ndarray
    birth: np.ndarray

    @classmethod
    def from_vertex_births(cls, values: np.ndarray) -> "_DSU":
        n = len(values)
        return cls(np.arange(n, dtype=np.int64), values.astype(np.float32, copy=False))

    def root(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node


def _as_1d_float_array(values) -> np.ndarray:
    """Convert NumPy/Torch-like input to float32 1D array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")
    return values


def count_1d_streaming_work(length: int, chunk_length: int | None = None) -> dict[str, int]:
    """Count 1D cells processed by the prefix solver."""
    if length <= 0:
        return {
            "vertices": 0,
            "edges": 0,
            "sort_items": 0,
            "dsu_find_calls_approx": 0,
            "old_cells_rebuilt": 0,
            "new_boundary_edges": 0,
        }

    edges = max(0, length - 1)
    if chunk_length is None or chunk_length >= length:
        old_cells_rebuilt = length + edges
        new_boundary_edges = 0
    else:
        old_cells_rebuilt = 0
        new_boundary_edges = max(0, int(np.ceil(length / chunk_length)) - 1)

    return {
        "vertices": length,
        "edges": edges,
        "sort_items": edges,
        "dsu_find_calls_approx": 4 * edges,
        "old_cells_rebuilt": old_cells_rebuilt,
        "new_boundary_edges": new_boundary_edges,
    }


def print_1d_streaming_work(label: str, length: int, chunk_length: int | None = None) -> None:
    """Print 1D work counters."""
    counts = count_1d_streaming_work(length, chunk_length=chunk_length)
    print(f"\n{label} 1D work units")
    print(f"vertices: {counts['vertices']}")
    print(f"edges processed: {counts['edges']}")
    print(f"sort items: {counts['sort_items']}")
    print(f"DSU find calls approx: {counts['dsu_find_calls_approx']}")
    print(f"old cells rebuilt: {counts['old_cells_rebuilt']}")
    print(f"new boundary edges: {counts['new_boundary_edges']}")


def find_critical_points_1d(values) -> dict[str, np.ndarray]:
    """Return local minima and maxima indices."""
    values = _as_1d_float_array(values)
    if len(values) < 3:
        empty = np.empty(0, dtype=np.int64)
        return {"minima": empty, "maxima": empty}

    middle = values[1:-1]
    minima = np.nonzero((middle < values[:-2]) & (middle < values[2:]))[0] + 1
    maxima = np.nonzero((middle > values[:-2]) & (middle > values[2:]))[0] + 1

    boundary_minima = []
    if values[0] < values[1]:
        boundary_minima.append(0)
    if values[-1] < values[-2]:
        boundary_minima.append(len(values) - 1)
    if boundary_minima:
        minima = np.concatenate([np.asarray(boundary_minima, dtype=np.int64), minima])

    return {"minima": minima.astype(np.int64), "maxima": maxima.astype(np.int64)}


def count_1d_topological_activity(values, chunk_length: int | None = None) -> dict[str, int]:
    """Count extrema plus chunk boundaries."""
    values = _as_1d_float_array(values)
    critical = find_critical_points_1d(values)
    if chunk_length is None or chunk_length >= len(values):
        chunks = 1 if len(values) else 0
        boundary_points = 0
    else:
        chunks = int(np.ceil(len(values) / chunk_length))
        boundary_points = max(0, chunks - 1)

    return {
        "chunks": chunks,
        "minima": int(len(critical["minima"])),
        "maxima": int(len(critical["maxima"])),
        "critical_points": int(len(critical["minima"]) + len(critical["maxima"])),
        "boundary_points": boundary_points,
        "adaptive_events": int(len(critical["minima"]) + len(critical["maxima"]) + boundary_points),
    }


def print_1d_topological_activity(label: str, values, chunk_length: int | None = None) -> None:
    """Print 1D activity counters."""
    counts = count_1d_topological_activity(values, chunk_length=chunk_length)
    print(f"\n{label} 1D activity units")
    print(f"chunks: {counts['chunks']}")
    print(f"minima: {counts['minima']}")
    print(f"maxima: {counts['maxima']}")
    print(f"critical points: {counts['critical_points']}")
    print(f"boundary points: {counts['boundary_points']}")
    print(f"adaptive events approx: {counts['adaptive_events']}")


@dataclass
class PrefixStreamingPersistence1D:
    """Recycle 1D vertices/edges and query exact prefix H0."""

    value_blocks: list[np.ndarray] = field(default_factory=list)
    component_edges: list[np.ndarray] = field(default_factory=list)
    length: int = 0

    def update_chunk(self, chunk) -> None:
        """Add new values and only new/boundary edges."""
        chunk = _as_1d_float_array(chunk)
        if len(chunk) == 0:
            return

        old_length = self.length
        self.value_blocks.append(chunk)
        values = self.values()
        self.length += len(chunk)

        new_edges = self._make_new_and_boundary_edges(values, old_length, self.length)
        if len(new_edges) > 0:
            self.component_edges.append(new_edges)

    def values(self) -> np.ndarray:
        """Return streamed values as one array."""
        if not self.value_blocks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self.value_blocks).astype(np.float32, copy=False)

    def current_exact_prefix_diagram(self) -> np.ndarray:
        """Compute exact H0 for chunks seen so far."""
        values = self.values()
        if len(values) == 0:
            return np.empty((0, 2), dtype=np.float32)
        edges = (
            np.vstack(self.component_edges).astype(np.float32, copy=False)
            if self.component_edges
            else np.empty((0, 3), dtype=np.float32)
        )
        return compute_h0_pairs_from_1d_edges(edges, values)

    def stored_state_summary(self) -> dict[str, int]:
        """Count recycled 1D state."""
        return {
            "values_seen": self.length,
            "value_blocks": len(self.value_blocks),
            "edge_blocks": len(self.component_edges),
            "edges_stored": int(sum(len(block) for block in self.component_edges)),
        }

    def _make_new_and_boundary_edges(
        self, values: np.ndarray, old_length: int, new_length: int
    ) -> np.ndarray:
        """Create adjacent-edge records for the new prefix region."""
        records = []
        start_edge = max(0, old_length - 1)
        for left in range(start_edge, new_length - 1):
            right = left + 1
            edge_value = max(values[left], values[right])
            records.append((edge_value, left, right))
        if not records:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(records, dtype=np.float32)


def compute_h0_pairs_from_1d_edges(component_edges: np.ndarray, vertex_births) -> np.ndarray:
    """Compute exact 1D H0 birth-death pairs."""
    vertex_births = _as_1d_float_array(vertex_births)
    if len(vertex_births) == 0:
        return np.empty((0, 2), dtype=np.float32)

    dsu = _DSU.from_vertex_births(vertex_births)
    pairs = []
    for death, left_float, right_float in component_edges[np.argsort(component_edges[:, 0])]:
        left, right = int(left_float), int(right_float)
        left_root, right_root = dsu.root(left), dsu.root(right)
        if left_root == right_root:
            continue

        if dsu.birth[left_root] <= dsu.birth[right_root]:
            survivor, victim = left_root, right_root
        else:
            survivor, victim = right_root, left_root

        if death > dsu.birth[victim]:
            pairs.append((dsu.birth[victim], death))
        dsu.parent[victim] = survivor

    pairs.append((float(np.min(vertex_births)), np.inf))
    return np.asarray(pairs, dtype=np.float32)


def compute_1d_prefix_diagrams_by_chunks(values, chunk_length: int) -> list[np.ndarray]:
    """Return exact H0 after every streamed 1D chunk."""
    values = _as_1d_float_array(values)
    engine = PrefixStreamingPersistence1D()
    diagrams = []
    for start in range(0, len(values), chunk_length):
        engine.update_chunk(values[start : start + chunk_length])
        diagrams.append(engine.current_exact_prefix_diagram())
    return diagrams


def compute_1d_prefix_diagrams_by_full_recompute(values, chunk_length: int) -> list[np.ndarray]:
    """Return prefix H0 by rebuilding each prefix from scratch."""
    values = _as_1d_float_array(values)
    diagrams = []
    for end in range(chunk_length, len(values) + chunk_length, chunk_length):
        end = min(end, len(values))
        engine = PrefixStreamingPersistence1D()
        engine.update_chunk(values[:end])
        diagrams.append(engine.current_exact_prefix_diagram())
        if end == len(values):
            break
    return diagrams


def compute_1d_streamed_prefix_final(values, chunk_length: int) -> np.ndarray:
    """Return exact H0 for the final streamed 1D prefix."""
    diagrams = compute_1d_prefix_diagrams_by_chunks(values, chunk_length)
    return diagrams[-1] if diagrams else np.empty((0, 2), dtype=np.float32)
