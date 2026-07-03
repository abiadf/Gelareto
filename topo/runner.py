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
from datetime import datetime
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
    actual_full_recompute_metrics_1d,
    actual_full_recompute_metrics_2d,
    actual_prefix_metrics_1d,
    actual_prefix_metrics_2d,
    count_1d_cells,
    count_2d_cells,
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


@dataclass
class RunData:
    """Store generated data and chunk sizes for optional plots."""

    values_1d: np.ndarray | None = None
    chunk_length_1d: int | None = None
    grid_2d: np.ndarray | None = None
    chunk_rows_2d: int | None = None


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
    kind: str,
    rows: int,
    cols: int,
    seed: int,
    device: torch.device,
    ring_strength: float,
    patch_strength: float,
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
    elif kind == "ring_patch":
        torch.manual_seed(seed)
        patch = DataGenerator2D.generate_patchy_matrix(
            rows, cols, device=device, min_val=0.0, max_val=patch_strength, smooth_radius=12
        )
        yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
        xx = xx / max(cols - 1, 1)
        yy = yy / max(rows - 1, 1)
        ring_mask = np.zeros((rows, cols), dtype=np.float32)
        for cx, cy in [(0.33, 0.35), (0.68, 0.60)]:
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            ring_mask[np.abs(dist - 0.16) <= 0.035] = 1.0
        rings = torch.tensor(ring_mask, dtype=torch.float32, device=device)
        grid = patch + ring_strength * rings
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


def run_1d(args: argparse.Namespace) -> RunData:
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
    return RunData(values_1d=values, chunk_length_1d=chunk_length)


def run_2d(args: argparse.Namespace) -> RunData:
    """Run 2D streamed-prefix persistence benchmarks."""
    device = _select_device(args.device)
    grid = make_2d_dataset(
        args.dataset_2d,
        args.rows,
        args.cols,
        args.seed,
        device,
        args.ring_strength,
        args.patch_strength,
    )
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
    return RunData(grid_2d=grid, chunk_rows_2d=chunk_rows)


def _select_device(name: str) -> torch.device:
    """Select CPU/CUDA for data generation."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _make_output_dir(base_dir: str | Path) -> Path:
    """Create a timestamped plot output directory."""
    base = Path(base_dir)
    stamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = base / f"run_{stamp}"
    suffix = 2
    while out_dir.exists():
        out_dir = base / f"run_{stamp}_{suffix:02d}"
        suffix += 1
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _save_run_metadata(out_dir: Path, args: argparse.Namespace, run_data: RunData) -> Path:
    """Save run sizes and settings next to generated plots."""
    lines = [
        f"case: {args.case}",
        f"chunks: {args.chunks}",
        f"dataset_1d: {args.dataset_1d}",
        f"n_points_1d: {0 if run_data.values_1d is None else len(run_data.values_1d)}",
        f"chunk_length_1d: {run_data.chunk_length_1d}",
        f"dataset_2d: {args.dataset_2d}",
        f"ring_strength: {args.ring_strength}",
        f"patch_strength: {args.patch_strength}",
        (
            "shape_2d: none"
            if run_data.grid_2d is None
            else f"shape_2d: {run_data.grid_2d.shape[0]} x {run_data.grid_2d.shape[1]}"
        ),
        f"pixels_2d: {0 if run_data.grid_2d is None else run_data.grid_2d.size}",
        f"chunk_rows_2d: {run_data.chunk_rows_2d}",
    ]
    path = out_dir / "run_metadata.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_matplotlib():
    """Import matplotlib only when plots are requested."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/pechstre_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_dataset_preview(
    out_dir: Path,
    fmt: str,
    values_1d: np.ndarray | None,
    grid_2d: np.ndarray | None,
) -> Path | None:
    """Save a preview of the generated 1D/2D inputs."""
    if values_1d is None and grid_2d is None:
        return None

    plt = _load_matplotlib()
    ncols = int(values_1d is not None) + int(grid_2d is not None)
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 3.2), squeeze=False)
    ax_iter = iter(axes[0])

    if values_1d is not None:
        ax = next(ax_iter)
        plot_values = values_1d
        if len(plot_values) > 5000:
            idx = np.linspace(0, len(plot_values) - 1, 5000).astype(np.int64)
            plot_values = plot_values[idx]
            x = idx
        else:
            x = np.arange(len(plot_values))
        ax.plot(x, plot_values, linewidth=1.0)
        ax.set_title("1D Input Signal")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")

    if grid_2d is not None:
        ax = next(ax_iter)
        im = ax.imshow(grid_2d, cmap="viridis", aspect="auto")
        ax.set_title("2D Input Grid")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = out_dir / f"input_preview.{fmt}"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _chunk_counts(max_chunks: int) -> list[int]:
    """Return chunk-count scenarios up to the requested limit."""
    candidates = [1, 2, 5, 10, 20]
    return [n for n in candidates if n <= max_chunks]


def _plot_with_chunk_markers(ax, x, y, label: str, **kwargs) -> None:
    """Plot a curve with markers at streamed-prefix update points."""
    ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.4, label=label, **kwargs)


def _x_thousands(values: np.ndarray) -> np.ndarray:
    """Scale x-axis counts to thousands."""
    return np.asarray(values, dtype=np.float64) / 1_000.0


def _format_plain_number_yaxis(ax) -> None:
    """Show numeric y ticks without unnecessary trailing zeros."""
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))


def _prefix_memory_units_1d(prefix_sizes: np.ndarray) -> np.ndarray:
    """Count stored 1D vertices plus edges per prefix."""
    return np.asarray([count_1d_cells(int(n)) for n in prefix_sizes], dtype=np.float64)


def _prefix_memory_units_2d(prefix_rows: np.ndarray, n_cols: int) -> np.ndarray:
    """Count stored 2D vertices, edges, and faces per prefix."""
    return np.asarray([count_2d_cells(int(rows), n_cols) for rows in prefix_rows], dtype=np.float64)


def _save_theoretical_work_plots(
    out_dir: Path,
    fmt: str,
    values_1d: np.ndarray | None,
    chunk_length_1d: int | None,
    grid_2d: np.ndarray | None,
    chunk_rows_2d: int | None,
    max_chunks: int,
) -> Path | None:
    """Save theoretical construction-work plots."""
    if values_1d is None and grid_2d is None:
        return None

    plt = _load_matplotlib()
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    fig.suptitle("Theoretical Recycled Construction Work", fontsize=14)

    if values_1d is not None and chunk_length_1d is not None:
        curve = construction_work_curve_1d(len(values_1d), chunk_length_1d)
        x = _x_thousands(curve["prefix_size"])
        _plot_with_chunk_markers(axes[0, 0], x, curve["full_recompute_per_step"], "rebuild")
        _plot_with_chunk_markers(
            axes[0, 0], x, curve["recycled_incremental_per_step"], "recycle"
        )
        axes[0, 0].set_title("1D Marginal Prefix-Update Work")
        axes[0, 0].set_xlabel("Total Signal Length (x1e3)")
        axes[0, 0].set_ylabel("Construction Work Units")
        _format_plain_number_yaxis(axes[0, 0])
        axes[0, 0].legend()

        _plot_with_chunk_markers(axes[0, 1], x, curve["full_recompute_cumulative"], "rebuild")
        _plot_with_chunk_markers(axes[0, 1], x, curve["recycled_cumulative"], "recycle")
        axes[0, 1].set_title(f"1D Cumulative Prefix-Update Work ({len(x)} chunks)")
        axes[0, 1].set_xlabel("Total Signal Length (x1e3)")
        axes[0, 1].set_ylabel("Cumulative Construction Work Units")
        _format_plain_number_yaxis(axes[0, 1])
        axes[0, 1].legend()

        for n_chunks in _chunk_counts(max_chunks):
            chunk_len = max(1, int(np.ceil(len(values_1d) / n_chunks)))
            chunk_curve = construction_work_curve_1d(len(values_1d), chunk_len)
            _plot_with_chunk_markers(
                axes[0, 2],
                _x_thousands(chunk_curve["prefix_size"]),
                chunk_curve["full_recompute_cumulative"],
                f"rebuild, {n_chunks} chunks",
                linestyle="--",
            )
        max_sensitivity_chunks = _chunk_counts(max_chunks)[-1]
        chunk_len = max(1, int(np.ceil(len(values_1d) / max_sensitivity_chunks)))
        chunk_curve = construction_work_curve_1d(len(values_1d), chunk_len)
        _plot_with_chunk_markers(
            axes[0, 2],
            _x_thousands(chunk_curve["prefix_size"]),
            chunk_curve["recycled_cumulative"],
            f"recycled, {max_sensitivity_chunks} chunks",
            color="black",
        )
        axes[0, 2].scatter(
            _x_thousands([len(values_1d)]),
            [curve["full_recompute_per_step"][-1]],
            marker="D",
            s=45,
            facecolors="none",
            edgecolors="black",
            linewidths=1.1,
            label="single full run",
            zorder=2,
        )
        axes[0, 2].set_title("1D Cumulative Repeated-Update Work")
        axes[0, 2].set_xlabel("Total Signal Length (x1e3)")
        axes[0, 2].set_ylabel("Cumulative Construction Work Units")
        _format_plain_number_yaxis(axes[0, 2])
        axes[0, 2].legend(fontsize=7)

        memory_units = _prefix_memory_units_1d(curve["prefix_size"])
        _plot_with_chunk_markers(axes[0, 3], x, memory_units, "stored vertices + edges")
        axes[0, 3].set_title(f"1D Algorithmic Memory ({len(x)} chunks)")
        axes[0, 3].set_xlabel("Total Signal Length (x1e3)")
        axes[0, 3].set_ylabel("Stored Cell Units")
        _format_plain_number_yaxis(axes[0, 3])
        axes[0, 3].legend()
    else:
        for ax in axes[0]:
            ax.axis("off")

    if grid_2d is not None and chunk_rows_2d is not None:
        curve = construction_work_curve_2d(grid_2d.shape[0], grid_2d.shape[1], chunk_rows_2d)
        x = _x_thousands(curve["prefix_rows"] * grid_2d.shape[1])
        _plot_with_chunk_markers(axes[1, 0], x, curve["full_recompute_per_step"], "rebuild")
        _plot_with_chunk_markers(
            axes[1, 0], x, curve["recycled_incremental_per_step"], "recycle"
        )
        axes[1, 0].set_title("2D Marginal Prefix-Update Work")
        axes[1, 0].set_xlabel("Total Pixels (x1e3)")
        axes[1, 0].set_ylabel("Construction Work Units")
        _format_plain_number_yaxis(axes[1, 0])
        axes[1, 0].legend()

        _plot_with_chunk_markers(axes[1, 1], x, curve["full_recompute_cumulative"], "rebuild")
        _plot_with_chunk_markers(axes[1, 1], x, curve["recycled_cumulative"], "recycle")
        axes[1, 1].set_title(f"2D Cumulative Prefix-Update Work ({len(x)} chunks)")
        axes[1, 1].set_xlabel("Total Pixels (x1e3)")
        axes[1, 1].set_ylabel("Cumulative Construction Work Units")
        _format_plain_number_yaxis(axes[1, 1])
        axes[1, 1].legend()

        for n_chunks in _chunk_counts(max_chunks):
            rows_per_chunk = max(1, int(np.ceil(grid_2d.shape[0] / n_chunks)))
            chunk_curve = construction_work_curve_2d(
                grid_2d.shape[0], grid_2d.shape[1], rows_per_chunk
            )
            _plot_with_chunk_markers(
                axes[1, 2],
                _x_thousands(chunk_curve["prefix_rows"] * grid_2d.shape[1]),
                chunk_curve["full_recompute_cumulative"],
                f"rebuild, {n_chunks} chunks",
                linestyle="--",
            )
        max_sensitivity_chunks = _chunk_counts(max_chunks)[-1]
        rows_per_chunk = max(1, int(np.ceil(grid_2d.shape[0] / max_sensitivity_chunks)))
        chunk_curve = construction_work_curve_2d(
            grid_2d.shape[0], grid_2d.shape[1], rows_per_chunk
        )
        _plot_with_chunk_markers(
            axes[1, 2],
            _x_thousands(chunk_curve["prefix_rows"] * grid_2d.shape[1]),
            chunk_curve["recycled_cumulative"],
            f"recycled, {max_sensitivity_chunks} chunks",
            color="black",
        )
        axes[1, 2].scatter(
            _x_thousands([grid_2d.size]),
            [curve["full_recompute_per_step"][-1]],
            marker="D",
            s=45,
            facecolors="none",
            edgecolors="black",
            linewidths=1.1,
            label="single full run",
            zorder=2,
        )
        axes[1, 2].set_title("2D Cumulative Repeated-Update Work")
        axes[1, 2].set_xlabel("Total Pixels (x1e3)")
        axes[1, 2].set_ylabel("Cumulative Construction Work Units")
        _format_plain_number_yaxis(axes[1, 2])
        axes[1, 2].legend(fontsize=7)

        memory_units = _prefix_memory_units_2d(curve["prefix_rows"], grid_2d.shape[1])
        _plot_with_chunk_markers(axes[1, 3], x, memory_units, "stored vertices + edges + faces")
        axes[1, 3].set_title(f"2D Algorithmic Memory ({len(x)} chunks)")
        axes[1, 3].set_xlabel("Total Pixels (x1e3)")
        axes[1, 3].set_ylabel("Stored Cell Units")
        _format_plain_number_yaxis(axes[1, 3])
        axes[1, 3].legend()
    else:
        for ax in axes[1]:
            ax.axis("off")

    fig.tight_layout()
    path = out_dir / f"theoretical_plot.{fmt}"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_actual_prefix_metric_plots(
    out_dir: Path,
    fmt: str,
    values_1d: np.ndarray | None,
    chunk_length_1d: int | None,
    grid_2d: np.ndarray | None,
    chunk_rows_2d: int | None,
) -> Path | None:
    """Save actual streamed-vs-rebuild prefix metrics."""
    if values_1d is None and grid_2d is None:
        return None

    plt = _load_matplotlib()
    fig, axes = plt.subplots(2, 5, figsize=(22, 8.5))
    fig.suptitle("Actual Prefix Benchmark Metrics", fontsize=14)

    if values_1d is not None and chunk_length_1d is not None:
        streamed = actual_prefix_metrics_1d(values_1d, chunk_length_1d)
        rebuilt = actual_full_recompute_metrics_1d(values_1d, chunk_length_1d)
        x_s = _x_thousands(streamed["prefix_size"])
        x_r = _x_thousands(rebuilt["prefix_size"])
        n_chunks_1d = len(x_s)
        full_work_1d = _prefix_memory_units_1d(streamed["prefix_size"])
        recycled_work_1d = np.diff(np.concatenate([[0.0], full_work_1d]))

        _plot_with_chunk_markers(axes[0, 0], x_s, streamed["elapsed_sec"], "recycle")
        _plot_with_chunk_markers(axes[0, 0], x_r, rebuilt["elapsed_sec"], "rebuild")
        axes[0, 0].set_title(f"1D Time ({n_chunks_1d} chunks)")
        axes[0, 0].set_xlabel("Total Signal Length (x1e3)")
        axes[0, 0].set_ylabel("Elapsed time (s)")
        axes[0, 0].legend()

        _plot_with_chunk_markers(
            axes[0, 1], x_s, recycled_work_1d, "recycle"
        )
        _plot_with_chunk_markers(axes[0, 1], x_s, full_work_1d, "rebuild")
        axes[0, 1].set_title(f"1D Actual Marginal Work ({n_chunks_1d} chunks)")
        axes[0, 1].set_xlabel("Total Signal Length (x1e3)")
        axes[0, 1].set_ylabel("Construction Work Units")
        _format_plain_number_yaxis(axes[0, 1])
        axes[0, 1].legend()

        _plot_with_chunk_markers(
            axes[0, 2], x_s, np.maximum.accumulate(streamed["rss_mib"]), "recycle"
        )
        _plot_with_chunk_markers(
            axes[0, 2], x_r, np.maximum.accumulate(rebuilt["rss_mib"]), "rebuild"
        )
        axes[0, 2].set_title(f"1D Peak RSS So Far ({n_chunks_1d} chunks)")
        axes[0, 2].set_xlabel("Total Signal Length (x1e3)")
        axes[0, 2].set_ylabel("Peak RSS So Far (MiB)")
        axes[0, 2].legend()

        _plot_with_chunk_markers(axes[0, 3], x_s, streamed["h0_pairs"], "H0 pairs")
        axes[0, 3].set_title(f"1D H0 Pair Growth ({n_chunks_1d} chunks)")
        axes[0, 3].set_xlabel("Total Signal Length (x1e3)")
        axes[0, 3].set_ylabel("H0 Pair Count")
        axes[0, 3].legend()

        axes[0, 4].axis("off")
    else:
        for ax in axes[0]:
            ax.axis("off")

    if grid_2d is not None and chunk_rows_2d is not None:
        streamed = actual_prefix_metrics_2d(grid_2d, chunk_rows_2d)
        rebuilt = actual_full_recompute_metrics_2d(grid_2d, chunk_rows_2d)
        x_s = _x_thousands(streamed["prefix_rows"] * grid_2d.shape[1])
        x_r = _x_thousands(rebuilt["prefix_rows"] * grid_2d.shape[1])
        n_chunks_2d = len(x_s)
        full_work_2d = _prefix_memory_units_2d(streamed["prefix_rows"], grid_2d.shape[1])
        recycled_work_2d = np.diff(np.concatenate([[0.0], full_work_2d]))

        _plot_with_chunk_markers(axes[1, 0], x_s, streamed["elapsed_sec"], "recycle")
        _plot_with_chunk_markers(axes[1, 0], x_r, rebuilt["elapsed_sec"], "rebuild")
        axes[1, 0].set_title(f"2D Time ({n_chunks_2d} chunks)")
        axes[1, 0].set_xlabel("Total Pixels (x1e3)")
        axes[1, 0].set_ylabel("Elapsed time (s)")
        axes[1, 0].legend()

        _plot_with_chunk_markers(
            axes[1, 1], x_s, recycled_work_2d, "recycle"
        )
        _plot_with_chunk_markers(axes[1, 1], x_s, full_work_2d, "rebuild")
        axes[1, 1].set_title(f"2D Actual Marginal Work ({n_chunks_2d} chunks)")
        axes[1, 1].set_xlabel("Total Pixels (x1e3)")
        axes[1, 1].set_ylabel("Construction Work Units")
        _format_plain_number_yaxis(axes[1, 1])
        axes[1, 1].legend()

        _plot_with_chunk_markers(
            axes[1, 2], x_s, np.maximum.accumulate(streamed["rss_mib"]), "recycle"
        )
        _plot_with_chunk_markers(
            axes[1, 2], x_r, np.maximum.accumulate(rebuilt["rss_mib"]), "rebuild"
        )
        axes[1, 2].set_title(f"2D Peak RSS So Far ({n_chunks_2d} chunks)")
        axes[1, 2].set_xlabel("Total Pixels (x1e3)")
        axes[1, 2].set_ylabel("Peak RSS So Far (MiB)")
        axes[1, 2].legend()

        _plot_with_chunk_markers(axes[1, 3], x_s, streamed["h0_pairs"], "H0 pairs")
        axes[1, 3].set_title(f"2D H0 Pair Growth ({n_chunks_2d} chunks)")
        axes[1, 3].set_xlabel("Total Pixels (x1e3)")
        axes[1, 3].set_ylabel("H0 Pair Count")
        axes[1, 3].legend()

        _plot_with_chunk_markers(axes[1, 4], x_s, streamed["h1_pairs"], "H1 pairs")
        axes[1, 4].set_title(f"2D H1 Pair Growth ({n_chunks_2d} chunks)")
        axes[1, 4].set_xlabel("Total Pixels (x1e3)")
        axes[1, 4].set_ylabel("H1 Pair Count")
        axes[1, 4].legend()
    else:
        for ax in axes[1]:
            ax.axis("off")

    fig.tight_layout()
    path = out_dir / f"actual_plot.{fmt}"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_diagram_pairs(ax, pairs: np.ndarray, label: str) -> None:
    """Plot finite persistence birth-death pairs."""
    pairs = np.asarray(pairs, dtype=np.float32)
    if len(pairs) == 0:
        ax.scatter([], [], label=f"{label}: 0 finite")
        return

    finite = np.isfinite(pairs[:, 1])
    finite_pairs = pairs[finite]
    n_inf = int(np.count_nonzero(~finite))
    if len(finite_pairs) > 0:
        ax.scatter(
            finite_pairs[:, 0],
            finite_pairs[:, 1],
            s=7,
            alpha=0.65,
            label=f"{label}: {len(finite_pairs)} finite, {n_inf} infinite",
        )
    else:
        ax.scatter([], [], label=f"{label}: 0 finite, {n_inf} infinite")


def _finish_diagram_axis(ax) -> None:
    """Add diagonal and labels to a persistence diagram axis."""
    from matplotlib.ticker import FuncFormatter

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    if lo == hi:
        hi = lo + 1.0
    pad = 0.04 * (hi - lo)
    lo -= pad
    hi += pad
    ax.fill_between([lo, hi], [lo, hi], [lo, lo], color="0.92", zorder=0)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.legend(fontsize=8)


def _save_persistence_diagram_plot(
    out_dir: Path,
    fmt: str,
    values_1d: np.ndarray | None,
    chunk_length_1d: int | None,
    grid_2d: np.ndarray | None,
    chunk_rows_2d: int | None,
) -> Path | None:
    """Save final persistence diagrams for the streamed method."""
    if values_1d is None and grid_2d is None:
        return None

    plt = _load_matplotlib()
    ncols = int(values_1d is not None) + int(grid_2d is not None)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.6), squeeze=False)
    ax_iter = iter(axes[0])

    if values_1d is not None and chunk_length_1d is not None:
        ax = next(ax_iter)
        h0 = compute_1d_streamed_prefix_final(values_1d, chunk_length_1d)
        _plot_diagram_pairs(ax, h0, "H0")
        ax.set_title("1D Persistence Diagram")
        _finish_diagram_axis(ax)

    if grid_2d is not None and chunk_rows_2d is not None:
        ax = next(ax_iter)
        h0, h1 = compute_streamed_by_rows(grid_2d, chunk_rows_2d)
        _plot_diagram_pairs(ax, h0, "H0")
        _plot_diagram_pairs(ax, h1, "H1")
        ax.set_title("2D Persistence Diagram")
        _finish_diagram_axis(ax)

    fig.tight_layout()
    path = out_dir / f"persistence_diagram.{fmt}"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_plots(args: argparse.Namespace, run_data: RunData) -> None:
    """Save runner plots and print their paths."""
    out_dir = _make_output_dir(args.output_dir)
    paths = [
        _save_run_metadata(out_dir, args, run_data),
        _save_dataset_preview(
            out_dir, args.plot_format, run_data.values_1d, run_data.grid_2d
        ),
        _save_theoretical_work_plots(
            out_dir,
            args.plot_format,
            run_data.values_1d,
            run_data.chunk_length_1d,
            run_data.grid_2d,
            run_data.chunk_rows_2d,
            args.max_plot_chunks,
        ),
        _save_actual_prefix_metric_plots(
            out_dir,
            args.plot_format,
            run_data.values_1d,
            run_data.chunk_length_1d,
            run_data.grid_2d,
            run_data.chunk_rows_2d,
        ),
        _save_persistence_diagram_plot(
            out_dir,
            args.plot_format,
            run_data.values_1d,
            run_data.chunk_length_1d,
            run_data.grid_2d,
            run_data.chunk_rows_2d,
        ),
    ]
    print(f"\nSaved plots to: {out_dir}")
    for path in paths:
        if path is not None:
            print(f"- {path.name}")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["1d", "2d", "both"], default="both")
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--output-dir", default="results/runner_outputs")
    parser.add_argument("--plot-format", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument(
        "--max-plot-chunks",
        type=int,
        default=None,
        help="largest chunk count in sensitivity plots; defaults to --chunks",
    )

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
        choices=[
            "constant",
            "hills",
            "mountainous",
            "rings",
            "ring_patch",
            "texture",
            "legacy",
        ],
        default="rings",
    )
    parser.add_argument(
        "--ring-strength",
        type=float,
        default=1.0,
        help="ring amplitude for --dataset-2d ring_patch",
    )
    parser.add_argument(
        "--patch-strength",
        type=float,
        default=1.0,
        help="patchy background amplitude for --dataset-2d ring_patch",
    )
    return parser


def main() -> None:
    """Run requested benchmark case."""
    args = build_parser().parse_args()
    if args.chunks < 1:
        raise ValueError("--chunks must be at least 1")
    if args.max_plot_chunks is None:
        args.max_plot_chunks = args.chunks
    if args.max_plot_chunks < 1:
        raise ValueError("--max-plot-chunks must be at least 1")

    run_data = RunData()
    if args.case in {"1d", "both"}:
        data_1d = run_1d(args)
        run_data.values_1d = data_1d.values_1d
        run_data.chunk_length_1d = data_1d.chunk_length_1d
    if args.case in {"2d", "both"}:
        data_2d = run_2d(args)
        run_data.grid_2d = data_2d.grid_2d
        run_data.chunk_rows_2d = data_2d.chunk_rows_2d

    if args.save_plots:
        save_plots(args, run_data)


if __name__ == "__main__":
    main()
