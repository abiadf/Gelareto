"""Module for util functions, places here to avoid redundant and circular imports"""
import gc
import os
import psutil
import shutil, pathlib
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataUtils:
    """Deals with making datasets"""

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

    @staticmethod
    def generate_torus_point_cloud(n_points: int = 2000, R_major: float = 1.0, r_minor: float = 0.3,
                                            noise: float = 0.01, rand_seed: Optional[int] = None,
                                            device = device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Generate a 2D torus-like (ring) point cloud on GPU using PyTorch.
        n_points = #points to generate
        rand_seed = random seed for reproducibility
        R_major/r_minor are the major/minor radii of the torus. Noise is added to r_minor"""

        if rand_seed is not None:
            torch.manual_seed(rand_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rand_seed)

        theta  = torch.rand(n_points, device=device, dtype=dtype) * (2 * torch.pi)
        radial = R_major + r_minor * torch.randn(n_points, device=device, dtype=dtype)
        x      = radial * torch.cos(theta)
        y      = radial * torch.sin(theta)
        pts    = torch.stack([x, y], dim=1)
        pts    = pts + noise * torch.randn_like(pts)
        return pts

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

    @staticmethod
    def get_ram_at_specific_time():
        """Returns current process RAM in MiB for a certain time. To get the RAM of a specific function,
        call this function before AND after the function is used, and take the difference in RAM usage"""
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

