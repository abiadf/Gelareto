"""Run notebook-style 1D/2D persistence benchmarks from the terminal.

Use one file for both dimensions so shared setup, timing, and work summaries
stay in one place. The 1D and 2D paths are separate functions and can be run
independently with --case 1d or --case 2d.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topo.persistence_1d_correct import (
    compute_1d_prefix_diagrams_by_chunks,
    compute_1d_prefix_diagrams_by_full_recompute,
    compute_1d_streamed_prefix_final,
)
from topo.persistence_2d_correct import (
    compute_2d_prefix_diagrams_by_full_recompute,
    compute_exact_prefix_diagrams_by_rows,
    compute_streamed_by_rows,
)
from topo.streaming_work_analysis import (
    construction_work_curve_1d,
    construction_work_curve_2d,
    final_work_ratio,
    topology_activity_summary_1d,
    topology_activity_summary_2d,
)
from topo.utils import DataGenerator1D, DataGenerator2D


@dataclass
class TimedResult:
    """Store one benchmark result."""

    label: str
    elapsed_sec: float
    rss_delta_mib: float
    result: object


def _rss_mib() -> float:
    """Return current process RSS in MiB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _time_call(label: str, fn: Callable[[], object]) -> TimedResult:
    """Run a function and record elapsed time plus RSS delta."""
    rss0 = _rss_mib()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    rss1 = _rss_mib()
    return TimedResult(label, elapsed, rss1 - rss0, result)


def _to_numpy(x) -> np.ndarray:
    """Convert Torch/NumPy data to NumPy float32."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def make_1d_dataset(kind: str, n_points: int, seed: int, device: torch.device) -> np.ndarray:
    """Create the 1D signal used by the runner."""
    if kind == "constant":
        values = DataGenerator1D.make_constant_signal(n_points, device=device)
    elif kind == "smooth":
        values = DataGenerator1D.make_smooth_signal(n_points, seed=seed, device=device)
    elif kind == "multiscale":
        values = DataGenerator1D.make_multiscale_signal(n_points, seed=seed, device=device)
    elif kind == "random_walk":
        values = DataGenerator1D.make_random_walk_signal(n_points, seed=seed, device=device)
    elif kind == "peaks":
        values = DataGenerator1D.make_piecewise_peak_signal(n_points, seed=seed, device=device)
    elif kind == "legacy":
        values = DataGenerator1D.make_yvals_1d(n_points, 0, 100, seed=seed)
    else:
        raise ValueError(f"unknown 1D dataset: {kind}")
    return _to_numpy(values)


def make_2d_dataset(
    kind: str, rows: int, cols: int, seed: int, device: torch.device
) -> np.ndarray:
    """Create the 2D grid used by the runner."""
    if kind == "constant":
        grid = DataGenerator2D.generate_constant_matrix(rows, cols, device=device)
    elif kind == "hills":
        grid = DataGenerator2D.generate_smooth_hills_matrix(rows, cols, seed=seed, device=device)
    elif kind == "mountainous":
        grid = DataGenerator2D.generate_mountainous_matrix(rows, cols, seed=seed, device=device)
    elif kind == "rings":
        grid = DataGenerator2D.generate_multi_ring_matrix(rows, cols, device=device)
    elif kind == "texture":
        grid = DataGenerator2D.generate_noisy_texture_matrix(rows, cols, seed=seed, device=device)
    elif kind == "legacy":
        grid = DataGenerator2D.generate_patchy_matrix(rows, cols, device=device)
    else:
        raise ValueError(f"unknown 2D dataset: {kind}")
    return _to_numpy(grid)


def _print_timing(result: TimedResult) -> None:
    """Print one timing row."""
    print(
        f"{result.label}: {result.elapsed_sec:.4f}s, "
        f"RSS delta {result.rss_delta_mib:+.2f} MiB"
    )


def run_1d(args: argparse.Namespace) -> None:
    """Run 1D streamed-prefix persistence benchmarks."""
    device = _select_device(args.device)
    values = make_1d_dataset(args.dataset_1d, args.n_points, args.seed, device)
    chunk_length = max(1, int(np.ceil(len(values) / args.chunks)))

    print("\n1D benchmark")
    print(f"dataset: {args.dataset_1d}")
    print(f"signal length: {len(values)}")
    print(f"chunks: {args.chunks}")
    print(f"chunk length: {chunk_length}")

    full_once = _time_call(
        "full array once",
        lambda: compute_1d_streamed_prefix_final(values, len(values)),
    )
    streamed_prefixes = _time_call(
        "streamed prefixes, recycled construction",
        lambda: compute_1d_prefix_diagrams_by_chunks(values, chunk_length),
    )
    rebuilt_prefixes = _time_call(
        "prefixes rebuilt from scratch",
        lambda: compute_1d_prefix_diagrams_by_full_recompute(values, chunk_length),
    )

    for result in (full_once, streamed_prefixes, rebuilt_prefixes):
        _print_timing(result)

    final_h0 = streamed_prefixes.result[-1] if streamed_prefixes.result else np.empty((0, 2))
    activity = topology_activity_summary_1d(values, chunk_length=chunk_length)
    work_curve = construction_work_curve_1d(len(values), chunk_length)

    print(f"H0 pairs: {len(final_h0)}")
    print(f"critical points: {activity['critical_points']}")
    print(f"boundary points: {activity['boundary_points']}")
    print(f"cumulative full/recycled construction ratio: {final_work_ratio(work_curve):.2f}x")


def run_2d(args: argparse.Namespace) -> None:
    """Run 2D streamed-prefix persistence benchmarks."""
    device = _select_device(args.device)
    grid = make_2d_dataset(args.dataset_2d, args.rows, args.cols, args.seed, device)
    chunk_rows = max(1, int(np.ceil(grid.shape[0] / args.chunks)))

    print("\n2D benchmark")
    print(f"dataset: {args.dataset_2d}")
    print(f"grid shape: {grid.shape[0]} x {grid.shape[1]}")
    print(f"chunks: {args.chunks}")
    print(f"chunk rows: {chunk_rows}")

    full_once = _time_call(
        "full array once",
        lambda: compute_streamed_by_rows(grid, grid.shape[0]),
    )
    streamed_prefixes = _time_call(
        "streamed prefixes, recycled construction",
        lambda: compute_exact_prefix_diagrams_by_rows(grid, chunk_rows),
    )
    rebuilt_prefixes = _time_call(
        "prefixes rebuilt from scratch",
        lambda: compute_2d_prefix_diagrams_by_full_recompute(grid, chunk_rows),
    )

    for result in (full_once, streamed_prefixes, rebuilt_prefixes):
        _print_timing(result)

    final_h0, final_h1 = (
        streamed_prefixes.result[-1] if streamed_prefixes.result else (np.empty((0, 2)), np.empty((0, 2)))
    )
    activity = topology_activity_summary_2d(grid, chunk_rows=chunk_rows)
    work_curve = construction_work_curve_2d(grid.shape[0], grid.shape[1], chunk_rows)

    print(f"H0 pairs: {len(final_h0)}")
    print(f"H1 pairs: {len(final_h1)}")
    print(f"critical pixels: {activity['critical_pixels']}")
    print(f"frontier pixels: {activity['frontier_pixels']}")
    print(f"cumulative full/recycled construction ratio: {final_work_ratio(work_curve):.2f}x")


def _select_device(name: str) -> torch.device:
    """Select CPU/CUDA for data generation."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["1d", "2d", "both"], default="both")
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--n-points", type=int, default=100_000)
    parser.add_argument(
        "--dataset-1d",
        choices=["constant", "smooth", "multiscale", "random_walk", "peaks", "legacy"],
        default="multiscale",
    )

    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--cols", type=int, default=100)
    parser.add_argument(
        "--dataset-2d",
        choices=["constant", "hills", "mountainous", "rings", "texture", "legacy"],
        default="rings",
    )
    return parser


def main() -> None:
    """Run requested benchmark case."""
    args = build_parser().parse_args()
    if args.chunks < 1:
        raise ValueError("--chunks must be at least 1")

    if args.case in {"1d", "both"}:
        run_1d(args)
    if args.case in {"2d", "both"}:
        run_2d(args)


if __name__ == "__main__":
    main()
