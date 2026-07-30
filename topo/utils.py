"""Module for util functions, places here to avoid redundant and circular imports"""
from __future__ import annotations

import gc
import os
import psutil
import shutil, pathlib
from typing import Optional, TypeVar
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable, Iterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = TypeVar("T")

def tqdm_progress_bar(iterable: Iterable[T], *, desc: str | None = None,
                      total: int | None = None, leave: bool = False, disable: bool | None = None,) -> Iterator[T]:
    """Wrap an iterable in tqdm when available."""
    if disable is None:
        disable = os.environ.get("TOPO_DISABLE_TQDM", "").lower() in {"1", "true", "yes"}
    if tqdm is None or disable:
        yield from iterable
        return
    yield from tqdm(iterable, desc=desc, total=total, leave=leave)


class DataGenerator1D:
    """Generate 1d data"""

    @staticmethod
    def make_constant_signal(num_points: int, value: float = 0.0, device=device) -> torch.Tensor:
        """Flat signal with near-zero topology."""
        return torch.full((num_points,), float(value), dtype=torch.float32, device=device)

    @staticmethod
    def make_smooth_signal(
        num_points: int,
        n_cycles: float = 3.0,
        noise: float = 0.0,
        seed: int = 0,
        device=device,
    ) -> torch.Tensor:
        """Low-frequency signal with few extrema."""
        rng = np.random.default_rng(seed)
        t = np.linspace(0.0, 2 * np.pi * n_cycles, num_points, dtype=np.float32)
        y = np.sin(t) + 0.35 * np.sin(0.5 * t + 0.3)
        if noise > 0:
            y += noise * rng.standard_normal(num_points).astype(np.float32)
        return torch.tensor(y, dtype=torch.float32, device=device)

    @staticmethod
    def make_multiscale_signal(
        num_points: int,
        n_freqs: int = 32,
        min_freq: float = 0.5,
        max_freq: float = 40.0,
        noise: float = 0.0,
        seed: int = 0,
        device=device,
    ) -> torch.Tensor:
        """Multi-frequency signal with many extrema."""
        rng = np.random.default_rng(seed)
        t = np.linspace(0.0, 2 * np.pi, num_points, dtype=np.float32)
        y = np.zeros(num_points, dtype=np.float32)
        for _ in range(n_freqs):
            freq = rng.uniform(min_freq, max_freq)
            amp = rng.uniform(0.05, 1.0) / np.sqrt(freq)
            phase = rng.uniform(0.0, 2 * np.pi)
            y += amp * np.sin(freq * t + phase)
        if noise > 0:
            y += noise * rng.standard_normal(num_points).astype(np.float32)
        return torch.tensor(y, dtype=torch.float32, device=device)

    @staticmethod
    def make_random_walk_signal(
        num_points: int,
        step_scale: float = 1.0,
        drift: float = 0.0,
        seed: int = 0,
        device=device,
    ) -> torch.Tensor:
        """Random-walk stream with irregular extrema."""
        rng = np.random.default_rng(seed)
        steps = drift + step_scale * rng.standard_normal(num_points).astype(np.float32)
        y = np.cumsum(steps).astype(np.float32)
        return torch.tensor(y, dtype=torch.float32, device=device)

    @staticmethod
    def make_piecewise_peak_signal(
        num_points: int,
        n_peaks: int = 12,
        peak_width: float = 0.02,
        noise: float = 0.0,
        seed: int = 0,
        device=device,
    ) -> torch.Tensor:
        """Sparse peak signal with controlled local features."""
        rng = np.random.default_rng(seed)
        x = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
        y = np.zeros(num_points, dtype=np.float32)
        centers = rng.uniform(0.05, 0.95, size=n_peaks)
        amps = rng.uniform(0.5, 2.0, size=n_peaks)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n_peaks)
        for center, amp, sign in zip(centers, amps, signs):
            y += sign * amp * np.exp(-0.5 * ((x - center) / peak_width) ** 2)
        if noise > 0:
            y += noise * rng.standard_normal(num_points).astype(np.float32)
        return torch.tensor(y, dtype=torch.float32, device=device)

    @staticmethod
    def make_yvals_1d(num_points: int, start_point: int, end_point: int, n_freqs: int = 100, seed: int = 0,) -> torch.Tensor:
        """Generate a complex oscillatory signal with many extrema."""
        rng = np.random.default_rng(seed)
        t   = np.linspace(start_point, end_point, num_points, dtype=np.float32)
        y   = np.zeros_like(t)
        for _ in range(n_freqs):
            freq  = rng.uniform(0.1, 50.0)
            amp   = rng.uniform(0.1, 1.0)
            phase = rng.uniform(0, 2 * np.pi)
            y += amp * np.sin(0.5* freq * t + phase)
        return torch.tensor(y, dtype=torch.float32)


class DataGenerator2D:
    """Generate 2d data"""

    @staticmethod
    def generate_constant_matrix(
        n_rows: int,
        n_cols: int,
        value: float = 0.0,
        device=device,
        dtype=torch.float32,
    ) -> torch.Tensor:
        """Flat image with near-zero topology."""
        return torch.full((n_rows, n_cols), float(value), dtype=dtype, device=device)

    @staticmethod
    def generate_smooth_hills_matrix(
        n_rows: int,
        n_cols: int,
        n_hills: int = 4,
        min_sigma: float = 0.12,
        max_sigma: float = 0.35,
        noise: float = 0.0,
        seed: int = 0,
        device=device,
    ) -> torch.Tensor:
        """Smooth Gaussian hills with few topological features."""
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:n_rows, 0:n_cols].astype(np.float32)
        xx = xx / max(n_cols - 1, 1)
        yy = yy / max(n_rows - 1, 1)
        arr = np.zeros((n_rows, n_cols), dtype=np.float32)
        for _ in range(n_hills):
            cx, cy = rng.uniform(0.1, 0.9, size=2)
            sigma = rng.uniform(min_sigma, max_sigma)
            amp = rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0])
            arr += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        if noise > 0:
            arr += noise * rng.standard_normal((n_rows, n_cols)).astype(np.float32)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    @staticmethod
    def generate_mountainous_matrix(
        n_rows: int,
        n_cols: int,
        n_bumps: int = 80,
        min_sigma: float = 0.015,
        max_sigma: float = 0.08,
        noise: float = 0.03,
        seed: int = 0,
        device=device,
    ) -> torch.Tensor:
        """Dense Gaussian bumps with many extrema."""
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:n_rows, 0:n_cols].astype(np.float32)
        xx = xx / max(n_cols - 1, 1)
        yy = yy / max(n_rows - 1, 1)
        arr = np.zeros((n_rows, n_cols), dtype=np.float32)
        for _ in range(n_bumps):
            cx, cy = rng.uniform(0.0, 1.0, size=2)
            sigma = rng.uniform(min_sigma, max_sigma)
            amp = rng.uniform(0.1, 1.0) * rng.choice([-1.0, 1.0])
            arr += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        if noise > 0:
            arr += noise * rng.standard_normal((n_rows, n_cols)).astype(np.float32)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    @staticmethod
    def generate_multi_ring_matrix(
        n_rows: int,
        n_cols: int,
        centers: Optional[list[tuple[float, float]]] = None,
        radius: float = 0.16,
        thickness: float = 0.035,
        ring_value: float = 1.0,
        background_value: float = 0.0,
        hole_value: float = 0.1,
        device=device,
    ) -> torch.Tensor:
        """Ring image with controlled H1 loops."""
        if centers is None:
            centers = [(0.33, 0.35), (0.68, 0.60)]
        yy, xx = np.mgrid[0:n_rows, 0:n_cols].astype(np.float32)
        xx = xx / max(n_cols - 1, 1)
        yy = yy / max(n_rows - 1, 1)
        arr = np.full((n_rows, n_cols), background_value, dtype=np.float32)
        for cx, cy in centers:
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            arr[dist < radius - thickness] = hole_value
            arr[np.abs(dist - radius) <= thickness] = ring_value
        return torch.tensor(arr, dtype=torch.float32, device=device)

    @staticmethod
    def generate_noisy_texture_matrix(
        n_rows: int,
        n_cols: int,
        smooth_radius: int = 2,
        noise_weight: float = 0.35,
        seed: int = 0,
        device=device,
    ) -> torch.Tensor:
        """Patchy texture plus noise for high topological density."""
        torch.manual_seed(seed)
        base = DataGenerator2D.generate_patchy_matrix(
            n_rows,
            n_cols,
            device=device,
            min_val=0.0,
            max_val=1.0,
            smooth_radius=smooth_radius,
        )
        noise = noise_weight * torch.rand((n_rows, n_cols), device=device)
        return base + noise

    @staticmethod
    def generate_donut_matrix(n_rows: int, n_cols: int, inner_radius: float = 2.5, outer_radius: float = 4.0) -> torch.Tensor:
        """Generate a 2D matrix with a donut shape: values are 1.0 in the donut region,
        0.1 inside the inner radius, and 0.0 outside the outer radius.
        Args:
            n_rows: Number of rows.
            n_cols: Number of columns.
            inner_radius: Inner radius of the donut.
            outer_radius: Outer radius of the donut.
        Returns:
            Tensor of shape (n_rows, n_cols)."""
        arr_2d = torch.zeros(n_rows, n_cols)
        center_x, center_y = n_rows / 2, n_cols / 2

        for i in range(n_rows):
            for j in range(n_cols):
                dist = ((i - center_x) ** 2 + (j - center_y) ** 2) ** 0.5
                if inner_radius < dist < outer_radius:
                    arr_2d[i, j] = 1.0
                elif dist < inner_radius:
                    arr_2d[i, j] = 0.1
        return arr_2d

    @staticmethod
    def generate_random_matrix(n_rows: int, n_cols: int, low: float = 0.0, high: float = 1.0,
                            device=device, dtype=torch.float32) -> torch.Tensor:
        """Generate a 2D torch matrix of random values in [low, high)."""
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("n_rows and n_cols must be positive integers.")
        if high <= low:
            raise ValueError("high must be greater than low.")
        return low + (high - low) * torch.rand((n_rows, n_cols), device=device, dtype=dtype)

    @staticmethod
    def generate_patchy_matrix(
        n_rows: int,
        n_cols: int,
        device: str,
        min_val: float = 0.0,
        max_val: float = 1.0,
        smooth_radius: int = 16,) -> torch.Tensor:
        """Generate a spatially correlated random field.
        Args:
            n_rows: Height.
            n_cols: Width.
            device: Torch device.
            min_val: Minimum value in the output field.
            max_val: Maximum value in the output field.
            smooth_radius: Larger values produce larger patches.
        Returns:
            Tensor of shape (n_rows, n_cols)."""
        x = torch.rand(1, 1, n_rows, n_cols, device=device)
        k = 2 * smooth_radius + 1
        x = F.avg_pool2d(
            x,
            kernel_size=k,
            stride=1,
            padding=smooth_radius,)
        x = x[0, 0]
        # Rescale from [0, 1] to [min_val, max_val].
        x = min_val + (max_val - min_val) * x
        return x


class MemoryUtils:
    """Keeps all memory cleaning/handling in one place"""

    @staticmethod
    def clear_memory():
        """Forces garbage collection to erase backend residuals between benchmarks."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def get_vms_mib():
        return psutil.Process(os.getpid()).memory_info().vms / (1024 * 1024)

    @staticmethod
    def clear_runtime_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def clear_jupyter_state():
        import sys
        sys.modules['__main__'].__dict__.get('Out', {}).clear()

    @staticmethod
    def clear_compilation_cache():
        for p in pathlib.Path('.').rglob('__pycache__'):
            shutil.rmtree(p, ignore_errors=True)
        for p in pathlib.Path('.').rglob('*.nbi'):
            p.unlink(missing_ok=True)
        for p in pathlib.Path('.').rglob('*.nbc'):
            p.unlink(missing_ok=True)
