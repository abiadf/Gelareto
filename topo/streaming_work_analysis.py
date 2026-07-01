"""Work-unit curves for streamed prefix-persistence benchmarks."""

from __future__ import annotations

import os
import time

import numpy as np
import psutil

from topo.persistence_1d_correct import PrefixStreamingPersistence1D
from topo.persistence_1d_correct import count_1d_topological_activity
from topo.persistence_2d_correct import IncrementalStreamingPersistence2D
from topo.persistence_2d_correct import count_adaptive_work


def prefix_chunk_sizes_1d(length: int, chunk_length: int) -> np.ndarray:
    """Return cumulative 1D prefix sizes."""
    if length <= 0:
        return np.empty(0, dtype=np.int64)
    ends = np.arange(chunk_length, length + chunk_length, chunk_length, dtype=np.int64)
    ends = np.minimum(ends, length)
    return np.unique(ends)


def prefix_row_counts_2d(n_rows: int, chunk_rows: int) -> np.ndarray:
    """Return cumulative 2D prefix row counts."""
    if n_rows <= 0:
        return np.empty(0, dtype=np.int64)
    ends = np.arange(chunk_rows, n_rows + chunk_rows, chunk_rows, dtype=np.int64)
    ends = np.minimum(ends, n_rows)
    return np.unique(ends)


def count_1d_cells(length: int) -> int:
    """Count vertices plus edges in a 1D prefix."""
    return max(length, 0) + max(length - 1, 0)


def count_2d_cells(n_rows: int, n_cols: int) -> int:
    """Count vertices plus edges plus faces in a 2D prefix."""
    vertices = max(n_rows, 0) * max(n_cols, 0)
    edges = max(n_rows, 0) * max(n_cols - 1, 0) + max(n_rows - 1, 0) * max(n_cols, 0)
    faces = max(n_rows - 1, 0) * max(n_cols - 1, 0)
    return vertices + edges + faces


def construction_work_curve_1d(length: int, chunk_length: int) -> dict[str, np.ndarray]:
    """Compare full-prefix rebuild work with recycled construction work."""
    prefixes = prefix_chunk_sizes_1d(length, chunk_length)
    full_recompute = np.array([count_1d_cells(int(n)) for n in prefixes], dtype=np.float64)
    recycled_incremental = np.diff(np.concatenate([[0.0], full_recompute]))
    saved_per_step = full_recompute - recycled_incremental
    return {
        "prefix_size": prefixes,
        "chunk_index": np.arange(1, len(prefixes) + 1, dtype=np.int64),
        "full_recompute_per_step": full_recompute,
        "recycled_incremental_per_step": recycled_incremental,
        "saved_per_step": saved_per_step,
        "full_recompute_cumulative": np.cumsum(full_recompute),
        "recycled_cumulative": np.cumsum(recycled_incremental),
        "saved_cumulative": np.cumsum(saved_per_step),
    }


def construction_work_curve_2d(n_rows: int, n_cols: int, chunk_rows: int) -> dict[str, np.ndarray]:
    """Compare full-prefix rebuild work with recycled construction work."""
    prefixes = prefix_row_counts_2d(n_rows, chunk_rows)
    full_recompute = np.array(
        [count_2d_cells(int(rows), n_cols) for rows in prefixes], dtype=np.float64
    )
    recycled_incremental = np.diff(np.concatenate([[0.0], full_recompute]))
    saved_per_step = full_recompute - recycled_incremental
    return {
        "prefix_rows": prefixes,
        "chunk_index": np.arange(1, len(prefixes) + 1, dtype=np.int64),
        "full_recompute_per_step": full_recompute,
        "recycled_incremental_per_step": recycled_incremental,
        "saved_per_step": saved_per_step,
        "full_recompute_cumulative": np.cumsum(full_recompute),
        "recycled_cumulative": np.cumsum(recycled_incremental),
        "saved_cumulative": np.cumsum(saved_per_step),
    }


def final_work_ratio(curve: dict[str, np.ndarray]) -> float:
    """Return cumulative full-recompute work divided by recycled work."""
    recycled = curve["recycled_cumulative"][-1]
    if recycled == 0:
        return np.inf
    return float(curve["full_recompute_cumulative"][-1] / recycled)


def topology_activity_summary_1d(values, chunk_length: int | None = None) -> dict[str, int]:
    """Return data-dependent 1D topology/activity counts."""
    return count_1d_topological_activity(values, chunk_length=chunk_length)


def topology_activity_summary_2d(grid, chunk_rows: int | None = None) -> dict[str, int]:
    """Return data-dependent 2D topology/activity counts."""
    if hasattr(grid, "detach"):
        grid = grid.detach().cpu().numpy()
    return count_adaptive_work(np.asarray(grid, dtype=np.float32), chunk_rows=chunk_rows)


def print_topology_activity_summary(label: str, counts: dict[str, int]) -> None:
    """Print topology/activity counts from a summary dict."""
    print(f"\n{label} topology/activity")
    for key, value in counts.items():
        print(f"{key}: {value}")


def _rss_mib() -> float:
    """Return current process RSS in MiB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def actual_prefix_metrics_1d(values, chunk_length: int) -> dict[str, np.ndarray]:
    """Run 1D streaming prefixes and record actual per-prefix metrics."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float32)
    engine = PrefixStreamingPersistence1D()

    prefix_size = []
    elapsed_sec = []
    rss_mib = []
    stored_edges = []
    h0_pairs = []
    critical_points = []

    t0 = time.perf_counter()
    for start in range(0, len(values), chunk_length):
        chunk = values[start : start + chunk_length]
        engine.update_chunk(chunk)
        h0 = engine.current_exact_prefix_diagram()
        prefix = engine.values()
        activity = topology_activity_summary_1d(prefix, chunk_length=chunk_length)
        state = engine.stored_state_summary()

        prefix_size.append(len(prefix))
        elapsed_sec.append(time.perf_counter() - t0)
        rss_mib.append(_rss_mib())
        stored_edges.append(state["edges_stored"])
        h0_pairs.append(len(h0))
        critical_points.append(activity["critical_points"])

    return {
        "prefix_size": np.asarray(prefix_size, dtype=np.int64),
        "elapsed_sec": np.asarray(elapsed_sec, dtype=np.float64),
        "rss_mib": np.asarray(rss_mib, dtype=np.float64),
        "stored_edges": np.asarray(stored_edges, dtype=np.int64),
        "h0_pairs": np.asarray(h0_pairs, dtype=np.int64),
        "critical_points": np.asarray(critical_points, dtype=np.int64),
    }


def actual_full_recompute_metrics_1d(values, chunk_length: int) -> dict[str, np.ndarray]:
    """Rebuild each 1D prefix from scratch and record actual metrics."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float32)

    prefix_size = []
    elapsed_sec = []
    rss_mib = []
    h0_pairs = []
    critical_points = []

    t0 = time.perf_counter()
    for end in prefix_chunk_sizes_1d(len(values), chunk_length):
        engine = PrefixStreamingPersistence1D()
        engine.update_chunk(values[:end])
        h0 = engine.current_exact_prefix_diagram()
        activity = topology_activity_summary_1d(values[:end], chunk_length=chunk_length)

        prefix_size.append(int(end))
        elapsed_sec.append(time.perf_counter() - t0)
        rss_mib.append(_rss_mib())
        h0_pairs.append(len(h0))
        critical_points.append(activity["critical_points"])

    return {
        "prefix_size": np.asarray(prefix_size, dtype=np.int64),
        "elapsed_sec": np.asarray(elapsed_sec, dtype=np.float64),
        "rss_mib": np.asarray(rss_mib, dtype=np.float64),
        "h0_pairs": np.asarray(h0_pairs, dtype=np.int64),
        "critical_points": np.asarray(critical_points, dtype=np.int64),
    }


def actual_prefix_metrics_2d(grid, chunk_rows: int) -> dict[str, np.ndarray]:
    """Run 2D streaming prefixes and record actual per-prefix metrics."""
    if hasattr(grid, "detach"):
        grid = grid.detach().cpu().numpy()
    grid = np.asarray(grid, dtype=np.float32)
    engine = IncrementalStreamingPersistence2D()

    prefix_rows = []
    elapsed_sec = []
    rss_mib = []
    stored_h0_edges = []
    stored_h1_edges = []
    h0_pairs = []
    h1_pairs = []
    critical_pixels = []

    t0 = time.perf_counter()
    for start in range(0, grid.shape[0], chunk_rows):
        chunk = grid[start : start + chunk_rows]
        engine.update_chunk(chunk)
        h0, h1 = engine.current_exact_prefix_diagrams()
        prefix = engine.values()
        activity = topology_activity_summary_2d(prefix, chunk_rows=chunk_rows)
        state = engine.stored_state_summary()

        prefix_rows.append(prefix.shape[0])
        elapsed_sec.append(time.perf_counter() - t0)
        rss_mib.append(_rss_mib())
        stored_h0_edges.append(sum(len(block) for block in engine.component_merge_edges))
        stored_h1_edges.append(sum(len(block) for block in engine.loop_dual_edges))
        h0_pairs.append(len(h0))
        h1_pairs.append(len(h1))
        critical_pixels.append(activity["critical_pixels"])

    return {
        "prefix_rows": np.asarray(prefix_rows, dtype=np.int64),
        "elapsed_sec": np.asarray(elapsed_sec, dtype=np.float64),
        "rss_mib": np.asarray(rss_mib, dtype=np.float64),
        "stored_h0_edges": np.asarray(stored_h0_edges, dtype=np.int64),
        "stored_h1_edges": np.asarray(stored_h1_edges, dtype=np.int64),
        "h0_pairs": np.asarray(h0_pairs, dtype=np.int64),
        "h1_pairs": np.asarray(h1_pairs, dtype=np.int64),
        "critical_pixels": np.asarray(critical_pixels, dtype=np.int64),
    }


def actual_full_recompute_metrics_2d(grid, chunk_rows: int) -> dict[str, np.ndarray]:
    """Rebuild each 2D prefix from scratch and record actual metrics."""
    if hasattr(grid, "detach"):
        grid = grid.detach().cpu().numpy()
    grid = np.asarray(grid, dtype=np.float32)

    prefix_rows = []
    elapsed_sec = []
    rss_mib = []
    h0_pairs = []
    h1_pairs = []
    critical_pixels = []

    t0 = time.perf_counter()
    for end in prefix_row_counts_2d(grid.shape[0], chunk_rows):
        engine = IncrementalStreamingPersistence2D()
        engine.update_chunk(grid[:end])
        h0, h1 = engine.current_exact_prefix_diagrams()
        activity = topology_activity_summary_2d(grid[:end], chunk_rows=chunk_rows)

        prefix_rows.append(int(end))
        elapsed_sec.append(time.perf_counter() - t0)
        rss_mib.append(_rss_mib())
        h0_pairs.append(len(h0))
        h1_pairs.append(len(h1))
        critical_pixels.append(activity["critical_pixels"])

    return {
        "prefix_rows": np.asarray(prefix_rows, dtype=np.int64),
        "elapsed_sec": np.asarray(elapsed_sec, dtype=np.float64),
        "rss_mib": np.asarray(rss_mib, dtype=np.float64),
        "h0_pairs": np.asarray(h0_pairs, dtype=np.int64),
        "h1_pairs": np.asarray(h1_pairs, dtype=np.int64),
        "critical_pixels": np.asarray(critical_pixels, dtype=np.int64),
    }


def actual_chunk_sensitivity_metrics_1d(values, chunk_counts: list[int]) -> dict[int, dict[str, np.ndarray]]:
    """Run actual 1D streaming metrics for several chunk counts."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float32)
    results = {}
    for n_chunks in chunk_counts:
        chunk_length = max(1, int(np.ceil(len(values) / n_chunks)))
        results[int(n_chunks)] = actual_prefix_metrics_1d(values, chunk_length)
    return results


def actual_chunk_sensitivity_metrics_2d(grid, chunk_counts: list[int]) -> dict[int, dict[str, np.ndarray]]:
    """Run actual 2D streaming metrics for several chunk counts."""
    if hasattr(grid, "detach"):
        grid = grid.detach().cpu().numpy()
    grid = np.asarray(grid, dtype=np.float32)
    results = {}
    for n_chunks in chunk_counts:
        chunk_rows = max(1, int(np.ceil(grid.shape[0] / n_chunks)))
        results[int(n_chunks)] = actual_prefix_metrics_2d(grid, chunk_rows)
    return results
