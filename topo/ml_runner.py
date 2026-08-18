"""Command-line entrypoint for video topology forecasting experiments.

This runner trains/evaluates the AE, geoAE, latent-TDA, frame-TDA, and
decode-to-frame scenarios without modifying the exploratory notebook.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import resource
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    import psutil
except ImportError:  # optional profiling dependency
    psutil = None

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import topo.config as topo_config
import topo.ml_tda as ml_tda
import topo.ml_tda_aux as ml_tda_aux
import topo.ml_tda_latent as ml_tda_latent
import topo.ml_tda_pixel as ml_tda_pixel
import topo.ml_tda_geoae as ml_tda_geoae
import topo.ml_tda_pwgeoae as ml_tda_pwgeoae
import topo.ml_tda_mixedgeo as ml_tda_mixedgeo
import topo.ml_tda_routed_mixedgeo as ml_tda_routed_mixedgeo
import topo.ml_tda_manifold_ph as ml_tda_manifold_ph
import topo.ml_tda_triangle as ml_tda_triangle
import topo.ml_tda_topoae as ml_tda_topoae
import topo.ml_tda_repr as ml_tda_repr
import topo.ml_tda_vjepa as ml_tda_vjepa
import topo.ml_tda_dinov2 as ml_tda_dinov2
import topo.ml_tda_clip as ml_tda_clip
import topo.ml_tda_simvp as ml_tda_simvp
import topo.ml_tda_classification as ml_tda_classification
import topo.persistence_3d as persistence_3d
from topo.utils import tqdm_progress_bar
from topo.ml_tda import (
    SpatialDecoder,
    SpatialEncoder,
    TopologicalPredictor,
    make_predictor,
    make_spatial_decoder,
    load_or_train_model,
    load_video_dataset,
    parse_tda_control_mode,
    test_predictor,
)


RUNNER_SCENARIOS = {
    "real_tda",
    "decode_z",
    "geo_real_tda",
    "topo_real_tda",
    "latent_tda",
    "latent_stability",
    "geo_latent_stability",
    "topo_latent_stability",
    "representation_fidelity",
    "geo_representation_fidelity",
    "topo_representation_fidelity",
    "geo_latent_tda",
    "pwgeo_latent_tda",
    "mixed_geo_latent_tda",
    "routed_mixed_geo_latent_tda",
    "manifold_mixed_geo_latent_tda",
    "triangle_curvature",
    "geo_latent_spectrum",
    "topo_latent_tda",
    "vae_latent_tda",
    "byol_latent_tda",
    "vjepa_latent_tda",
    "dinov2_latent_tda",
    "dinov2_finetune_latent_tda",
    "clip_latent_tda",
    "simvp",
    "latent_classification",
    "video3d_tda",
    "aux_tda",
    "pixel_tda",
    "geo_pixel_tda",
    "topo_pixel_tda",
    "geo_decode_z",
    "topo_decode_z",
}


@dataclass
class VideoContext:
    dataset_config: dict[str, Any]
    x_train: torch.Tensor
    x_test: torch.Tensor
    latent_dim: int
    hidden_dim: int
    predictor_epochs: int
    learning_rate: float
    predictor_learning_rate: float
    real_tda_scale: int | float
    real_tda_bins: int
    horizon: int
    ae_epochs: int
    ae_frame_batch_size: int | None
    ae_max_frames_per_epoch: int | None


@dataclass
class RunConfig:
    scenario: str
    dataset: str
    device: str
    run_seeds: list[int] | None
    horizon: int | None
    latent_dim: int | None
    hidden_dim: int | None
    predictor_epochs: int | None
    learning_rate: float | None
    predictor_learning_rate: float | None
    real_tda_scale: float | None
    real_tda_bins: int | None
    ae_epochs: int | None
    ae_frame_batch_size: int | None
    ae_max_frames_per_epoch: int | None
    force_rebuild_data_cache: bool
    num_train_clips: int | None
    num_test_clips: int | None
    modes: list[str] | None
    retrain_encoder: bool
    retrain_predictor: bool
    aux_tda_lambda: float | None
    latent_tda_window: int | None
    latent_tda_bins: int | None
    latent_tda_max_train: int | None
    latent_tda_max_test: int | None
    recompute_latent_tda_features: bool
    stability_noise_levels: list[float] | None
    fidelity_windows: int
    geo_ae_lambda: float
    geo_ae_epochs: int | None
    geo_ae_pair_batch_size: int
    manifold_signatures: list[str]
    manifold_signature_policy: str
    signature_shrinkages: list[float]
    route_lambda: float
    route_knn: int
    route_temperature: float
    vr_distance: str
    triangle_knn: list[int]
    triangle_samples: int
    triangle_max_points: int
    triangle_flat_threshold: float
    pwgeo_ae_lambda: float
    pwgeo_ae_epochs: int | None
    pwgeo_ae_pair_batch_size: int
    pwgeo_blend: float | list[float]
    pwgeo_knn: int
    pwgeo_h0_weight: float
    pwgeo_h1_weight: float
    topo_ae_lambda: float
    topo_ae_epochs: int | None
    topo_ae_pair_batch_size: int
    topo_ae_distance: str
    decoder_type: str
    predictor_type: str
    vae_beta: float
    vae_epochs: int | None
    byol_epochs: int | None
    byol_noise_std: float
    vjepa_repo: str
    vjepa_batch_size: int
    vjepa_num_frames: int
    dinov2_repo: str
    dinov2_batch_size: int
    dinov2_image_size: int
    dinov2_trainable_blocks: int
    dinov2_finetune_epochs: int
    dinov2_encoder_lr: float
    dinov2_clip_batch_size: int
    clip_repo: str
    clip_batch_size: int
    clip_image_size: int
    simvp_input_frames: int
    video3d_tda_bins: int | None
    video3d_tda_scale: float | None
    video3d_tda_boundary_slices: int | None
    recompute_video3d_tda_features: bool
    pixel_tda_batch_size: int | None
    pixel_tda_fg_weight: float | None
    pixel_tda_fg_threshold: float | None
    profile_run: bool
    profile_sizes: list[int] | None
    profile_size: int | None
    profile_warmup_runs: int
    profile_repeats: int
    profile_repeat: int | None
    profile_is_warmup: bool
    hparam_file: Path | None
    output_dir: Path
    no_save: bool


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_str_list(value: str | None) -> list[str] | None:
    if value is None or value == "":
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_float_list(value: str | None) -> list[float] | None:
    if value is None or value == "":
        return None
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _select_device(name: str) -> torch.device:
    requested = name.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def _override(value, default):
    return default if value is None else value


def _make_output_dir(base_dir: Path) -> Path:
    stamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = base_dir / f"run_{stamp}"
    suffix = 2
    while out_dir.exists():
        out_dir = base_dir / f"run_{stamp}_{suffix:02d}"
        suffix += 1
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _apply_tuned_hparams(cfg: RunConfig) -> RunConfig:
    if cfg.hparam_file is None:
        return cfg
    if not cfg.hparam_file.exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {cfg.hparam_file}")
    payload = json.loads(cfg.hparam_file.read_text())
    params_by_dataset = payload.get("best_by_dataset", payload)
    params = params_by_dataset.get(cfg.dataset)
    if not params:
        print(f"No tuned hyperparameters for dataset={cfg.dataset} in {cfg.hparam_file}; using config/defaults.")
        return cfg
    updates: dict[str, Any] = {}
    if cfg.hidden_dim is None and "hidden_dim" in params:
        updates["hidden_dim"] = int(params["hidden_dim"])
    if cfg.predictor_epochs is None and "predictor_epochs" in params:
        updates["predictor_epochs"] = int(params["predictor_epochs"])
    tuned_predictor_lr = params.get("predictor_learning_rate", params.get("learning_rate"))
    if cfg.predictor_learning_rate is None and tuned_predictor_lr is not None:
        updates["predictor_learning_rate"] = float(tuned_predictor_lr)
    if not updates:
        return cfg
    tuned = replace(cfg, **updates)
    print(
        f"Using tuned hyperparameters for dataset={cfg.dataset}: "
        f"hidden_dim={tuned.hidden_dim}, predictor_epochs={tuned.predictor_epochs}, "
        f"predictor_learning_rate={tuned.predictor_learning_rate}"
    )
    return tuned


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    return str(value)


def _mse_r2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> tuple[float, float]:
    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    mse = torch.mean((pred - target) ** 2)
    var = torch.mean((target - target.mean()) ** 2)
    r2 = 1.0 - (mse / var.clamp_min(eps))
    return float(mse), float(r2)


def _ru_maxrss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


_PROCESS = psutil.Process(os.getpid()) if psutil is not None else None


def _current_cpu_rss_mb() -> float:
    if _PROCESS is not None:
        return float(_PROCESS.memory_info().rss / (1024.0 * 1024.0))
    return _ru_maxrss_mb()


class RunProfiler:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.start_time = 0.0
        self.wall_time_sec = 0.0
        self.topo_time_sec = 0.0
        self.peak_cpu_rss_mb = _ru_maxrss_mb()
        self.peak_accelerator_mem_mb = np.nan
        self.phase_time_sec = {}
        self.phase_peak_accelerator_mem_mb = {}
        self.phase_peak_cpu_rss_mb = {}
        self._active_phases = {}
        self._phase_counter = 0
        self._phase_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.peak_cpu_rss_mb = _ru_maxrss_mb()
        self._reset_accelerator_memory()
        self._previous_latent_profiler = getattr(ml_tda_latent, "ACTIVE_PROFILER", None)
        ml_tda_latent.ACTIVE_PROFILER = self
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        self._patch_stack = contextlib.ExitStack()
        self._patch_stack.enter_context(self._profiled_topology_calls())
        return self

    def __exit__(self, exc_type, exc, tb):
        self.wall_time_sec = time.perf_counter() - self.start_time
        if hasattr(self, "_patch_stack"):
            self._patch_stack.close()
        ml_tda_latent.ACTIVE_PROFILER = self._previous_latent_profiler
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample_once()
        return False

    def _reset_accelerator_memory(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            # MPS has no resettable peak counter; sampling below is best-effort.
            pass

    def _sample_loop(self):
        while not self._stop.wait(0.05):
            self._sample_once()

    def _sample_once(self):
        current_cpu_mb = _current_cpu_rss_mb()
        self.peak_cpu_rss_mb = max(self.peak_cpu_rss_mb, _ru_maxrss_mb(), current_cpu_mb)
        with self._phase_lock:
            active_phases = list(self._active_phases.values())
        for phase_name, start_cpu_mb in active_phases:
            delta_cpu_mb = max(0.0, current_cpu_mb - start_cpu_mb)
            current = self.phase_peak_cpu_rss_mb.get(phase_name, np.nan)
            if np.isnan(current):
                self.phase_peak_cpu_rss_mb[phase_name] = delta_cpu_mb
            else:
                self.phase_peak_cpu_rss_mb[phase_name] = max(current, delta_cpu_mb)
        if torch.cuda.is_available():
            cuda_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if np.isnan(self.peak_accelerator_mem_mb):
                self.peak_accelerator_mem_mb = float(cuda_mb)
            else:
                self.peak_accelerator_mem_mb = max(self.peak_accelerator_mem_mb, float(cuda_mb))
        elif hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            try:
                mps_mb = torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
            except Exception:
                return
            if np.isnan(self.peak_accelerator_mem_mb):
                self.peak_accelerator_mem_mb = float(mps_mb)
            else:
                self.peak_accelerator_mem_mb = max(self.peak_accelerator_mem_mb, float(mps_mb))

    def _current_accelerator_peak_mb(self) -> float:
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            try:
                return float(torch.mps.current_allocated_memory() / (1024.0 * 1024.0))
            except Exception:
                return np.nan
        return np.nan

    @contextlib.contextmanager
    def phase(self, name: str):
        start_cpu_mb = _current_cpu_rss_mb()
        with self._phase_lock:
            self._phase_counter += 1
            phase_id = self._phase_counter
            self._active_phases[phase_id] = (name, start_cpu_mb)
        start_allocated_mb = np.nan
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_allocated_mb = float(torch.cuda.memory_allocated() / (1024.0 * 1024.0))
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            current_cpu_mb = _current_cpu_rss_mb()
            delta_cpu_mb = max(0.0, current_cpu_mb - start_cpu_mb)
            current_cpu_peak = self.phase_peak_cpu_rss_mb.get(name, np.nan)
            if np.isnan(current_cpu_peak):
                self.phase_peak_cpu_rss_mb[name] = delta_cpu_mb
            else:
                self.phase_peak_cpu_rss_mb[name] = max(current_cpu_peak, delta_cpu_mb)
            with self._phase_lock:
                self._active_phases.pop(phase_id, None)
            peak_mb = self._current_accelerator_peak_mb()
            if torch.cuda.is_available() and not np.isnan(start_allocated_mb) and not np.isnan(peak_mb):
                peak_mb = max(0.0, peak_mb - start_allocated_mb)
            self.phase_time_sec[name] = self.phase_time_sec.get(name, 0.0) + elapsed
            if not np.isnan(peak_mb):
                current = self.phase_peak_accelerator_mem_mb.get(name, np.nan)
                if np.isnan(current):
                    self.phase_peak_accelerator_mem_mb[name] = peak_mb
                else:
                    self.phase_peak_accelerator_mem_mb[name] = max(current, peak_mb)
                if np.isnan(self.peak_accelerator_mem_mb):
                    self.peak_accelerator_mem_mb = peak_mb
                else:
                    self.peak_accelerator_mem_mb = max(self.peak_accelerator_mem_mb, peak_mb)

    @contextlib.contextmanager
    def _timed_topology(self):
        start = time.perf_counter()
        with self.phase("persistence"):
            yield
        self.topo_time_sec += time.perf_counter() - start

    @contextlib.contextmanager
    def _profiled_topology_calls(self):
        patches = []

        def patch(module, name):
            original = getattr(module, name, None)
            if original is None:
                return

            def wrapped(*args, **kwargs):
                with self._timed_topology():
                    return original(*args, **kwargs)

            setattr(module, name, wrapped)
            patches.append((module, name, original))

        patch(ml_tda_latent, "latent_window_betti_features")
        patch(ml_tda_latent, "_window_persistence_diagrams")
        patch(ml_tda, "run_cripser_tda_on_current_frames")
        patch(persistence_3d, "streaming_video_betti_features")
        try:
            yield
        finally:
            for module, name, original in reversed(patches):
                setattr(module, name, original)

    def to_row(self, cfg: RunConfig, context: VideoContext) -> dict[str, Any]:
        n_train = int(context.x_train.shape[1])
        n_test = int(context.x_test.shape[1])
        sequence_len = int(context.x_train.shape[0])
        frame_h = int(context.x_train.shape[-2])
        frame_w = int(context.x_train.shape[-1])
        processed_frames = sequence_len * (n_train + n_test)
        throughput = processed_frames / self.wall_time_sec if self.wall_time_sec > 0 else np.nan
        mode_label = ",".join(cfg.modes) if cfg.modes else np.nan
        return {
            "dataset": cfg.dataset,
            "scenario": cfg.scenario,
            "mode": mode_label,
            "profile_size": cfg.profile_size if cfg.profile_size is not None else np.nan,
            "profile_repeat": cfg.profile_repeat if cfg.profile_repeat is not None else np.nan,
            "profile_is_warmup": bool(cfg.profile_is_warmup),
            "wall_time_sec": float(self.wall_time_sec),
            "peak_cpu_rss_mb": float(self.peak_cpu_rss_mb),
            "peak_accelerator_mem_mb": float(self.peak_accelerator_mem_mb),
            "topo_time_sec": float(self.topo_time_sec),
            "topo_fraction": float(self.topo_time_sec / self.wall_time_sec) if self.wall_time_sec > 0 else np.nan,
            "encoding_time_sec": float(self.phase_time_sec.get("encoding", 0.0)),
            "encoder_train_time_sec": float(self.phase_time_sec.get("encoder_train", 0.0)),
            "persistence_time_sec": float(self.phase_time_sec.get("persistence", 0.0)),
            "fusion_time_sec": float(self.phase_time_sec.get("fusion", 0.0)),
            "predictor_time_sec": float(self.phase_time_sec.get("predictor", 0.0)),
            "encoding_peak_accelerator_mem_mb": float(self.phase_peak_accelerator_mem_mb.get("encoding", np.nan)),
            "encoder_train_peak_accelerator_mem_mb": float(self.phase_peak_accelerator_mem_mb.get("encoder_train", np.nan)),
            "persistence_peak_accelerator_mem_mb": float(self.phase_peak_accelerator_mem_mb.get("persistence", np.nan)),
            "fusion_peak_accelerator_mem_mb": float(self.phase_peak_accelerator_mem_mb.get("fusion", np.nan)),
            "predictor_peak_accelerator_mem_mb": float(self.phase_peak_accelerator_mem_mb.get("predictor", np.nan)),
            "encoding_peak_cpu_rss_mb": float(self.phase_peak_cpu_rss_mb.get("encoding", np.nan)),
            "encoder_train_peak_cpu_rss_mb": float(self.phase_peak_cpu_rss_mb.get("encoder_train", np.nan)),
            "persistence_peak_cpu_rss_mb": float(self.phase_peak_cpu_rss_mb.get("persistence", np.nan)),
            "fusion_peak_cpu_rss_mb": float(self.phase_peak_cpu_rss_mb.get("fusion", np.nan)),
            "predictor_peak_cpu_rss_mb": float(self.phase_peak_cpu_rss_mb.get("predictor", np.nan)),
            "throughput_frames_per_sec": float(throughput),
            "n_train_clips": n_train,
            "n_test_clips": n_test,
            "sequence_len": sequence_len,
            "frame_size": f"{frame_h}x{frame_w}",
            "n_processed_frames": processed_frames,
            "device": str(ml_tda.get_runtime_device()),
        }


def _save_results(
    cfg: RunConfig,
    context: VideoContext | None,
    results_df: pd.DataFrame | None,
    summary_df: pd.DataFrame | None,
    profile_df: pd.DataFrame | None = None,
) -> None:
    if cfg.no_save:
        return

    out_dir = _make_output_dir(cfg.output_dir)
    run_config = asdict(cfg)
    context_config = None if context is None else {
        "dataset_config": context.dataset_config,
        "x_train_shape": tuple(context.x_train.shape),
        "x_test_shape": tuple(context.x_test.shape),
        "latent_dim": context.latent_dim,
        "hidden_dim": context.hidden_dim,
        "predictor_epochs": context.predictor_epochs,
        "learning_rate": context.learning_rate,
        "predictor_learning_rate": context.predictor_learning_rate,
        "real_tda_scale": context.real_tda_scale,
        "real_tda_bins": context.real_tda_bins,
        "horizon": context.horizon,
        "ae_epochs": context.ae_epochs,
        "ae_frame_batch_size": context.ae_frame_batch_size,
        "ae_max_frames_per_epoch": context.ae_max_frames_per_epoch,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {"run_config": run_config, "context": context_config},
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    if results_df is not None:
        float_format = "%.8g" if "representation_fidelity" in cfg.scenario else "%.4f"
        results_df.to_csv(out_dir / "results.csv", index=False, float_format=float_format)
        gate_columns = [
            "gate_h0_mean",
            "gate_h0_std",
            "gate_h1_mean",
            "gate_h1_std",
        ]
        if all(column in results_df.columns for column in gate_columns):
            gate_results = results_df.dropna(subset=["gate_h0_mean"]).copy()
            if not gate_results.empty:
                identity_columns = [
                    column
                    for column in ("dataset", "encoder", "seed", "mode", "horizon")
                    if column in gate_results.columns
                ]
                gate_results[identity_columns + gate_columns].to_csv(
                    out_dir / "gate_results.csv",
                    index=False,
                    float_format="%.4f",
                )
                gate_summary = gate_results.groupby("mode")[gate_columns].agg(["mean", "std"])
                gate_summary.to_csv(out_dir / "gate_summary.csv", float_format="%.4f")
    if summary_df is not None:
        float_format = "%.8g" if "representation_fidelity" in cfg.scenario else "%.4f"
        summary_df.to_csv(out_dir / "summary.csv", float_format=float_format)
    if profile_df is not None:
        profile_df.to_csv(out_dir / "profile.csv", index=False, float_format="%.4f")
    print(f"\nSaved run outputs to {out_dir}")


def load_video_context(cfg: RunConfig) -> VideoContext:
    dataset_configs = topo_config.DATASET_CONFIGS
    if cfg.dataset not in dataset_configs:
        valid = ", ".join(sorted(dataset_configs))
        raise ValueError(f"Unknown dataset {cfg.dataset!r}. Valid datasets: {valid}")

    run_config = dict(dataset_configs[cfg.dataset])
    if run_config.get("kind") in {"lorenz_moving_shapes", "orbiting_shapes", "lorenz96_timeseries", "noisy_video"}:
        if cfg.num_train_clips is not None:
            run_config["num_train_clips"] = cfg.num_train_clips
        if cfg.num_test_clips is not None:
            run_config["num_test_clips"] = cfg.num_test_clips
        run_config["force_rebuild_cache"] = cfg.force_rebuild_data_cache
    elif run_config.get("kind") in {"aeon_classification", "aeon_raw_classification"}:
        if cfg.num_train_clips is not None:
            run_config["max_train_clips"] = cfg.num_train_clips
        if cfg.num_test_clips is not None:
            run_config["max_test_clips"] = cfg.num_test_clips
        run_config["force_rebuild_cache"] = cfg.force_rebuild_data_cache
    elif run_config.get("kind") == "ctc_tif_clips":
        if cfg.num_train_clips is not None:
            run_config["max_train_clips"] = cfg.num_train_clips
        if cfg.num_test_clips is not None:
            run_config["max_test_clips"] = cfg.num_test_clips
        run_config["force_rebuild_cache"] = cfg.force_rebuild_data_cache
    elif run_config.get("kind") == "synthetic_motion_classification":
        run_config["_classification_train_limit"] = cfg.num_train_clips
        run_config["_classification_test_limit"] = cfg.num_test_clips

    train_array, test_array = load_video_dataset(run_config)
    x_train = torch.from_numpy(train_array).unsqueeze(2)
    x_test = torch.from_numpy(test_array).unsqueeze(2)
    if x_train.dtype != torch.uint8:
        x_train = x_train.float()
        x_test = x_test.float()
    is_composite_classification = run_config.get("kind") == "synthetic_motion_classification"
    if not is_composite_classification and cfg.num_train_clips is not None and x_train.shape[1] > cfg.num_train_clips:
        x_train = x_train[:, : cfg.num_train_clips].contiguous()
    if not is_composite_classification and cfg.num_test_clips is not None and x_test.shape[1] > cfg.num_test_clips:
        x_test = x_test[:, : cfg.num_test_clips].contiguous()

    latent_dim = int(_override(cfg.latent_dim, run_config.get("LATENT_DIM", 128)))
    hidden_dim = int(_override(cfg.hidden_dim, run_config.get("HIDDEN_DIM", 128)))
    predictor_epochs = int(_override(cfg.predictor_epochs, run_config.get("PREDICTOR_EPOCHS", 10)))
    learning_rate = float(_override(cfg.learning_rate, run_config.get("learning_rate", 3e-4)))
    predictor_learning_rate = float(_override(cfg.predictor_learning_rate, learning_rate))
    real_tda_scale = _override(cfg.real_tda_scale, run_config.get("REAL_TDA_SCALE", 15))
    real_tda_bins = int(_override(cfg.real_tda_bins, run_config.get("REAL_TDA_BINS", 25)))
    horizon = int(
        _override(cfg.horizon, run_config.get("HORIZON", 1))
    )
    ae_frame_batch_size = _override(
        cfg.ae_frame_batch_size,
        run_config.get("AE_FRAME_BATCH_SIZE", 256),
    )
    ae_epochs = int(_override(cfg.ae_epochs, run_config.get("AE_EPOCHS", 3)))
    ae_max_frames_per_epoch = _override(
        cfg.ae_max_frames_per_epoch,
        run_config.get("AE_MAX_FRAMES_PER_EPOCH", 8192),
    )

    stats_train = x_train.float() / 255.0 if x_train.dtype == torch.uint8 else x_train.float()
    stats_test = x_test.float() / 255.0 if x_test.dtype == torch.uint8 else x_test.float()
    print(f"Dataset: {cfg.dataset} | X_train: {tuple(x_train.shape)} | X_test: {tuple(x_test.shape)}")
    print(
        "Input ranges: "
        f"train min={stats_train.min().item():.4f} max={stats_train.max().item():.4f} "
        f"mean={stats_train.mean().item():.4f} | "
        f"test min={stats_test.min().item():.4f} max={stats_test.max().item():.4f} "
        f"mean={stats_test.mean().item():.4f}"
    )

    ml_tda.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=latent_dim,
        REAL_TDA_SCALE=real_tda_scale,
        REAL_TDA_BINS=real_tda_bins,
        HORIZON=horizon,
        PREDICTOR_EPOCHS=predictor_epochs,
        PREDICTOR_TYPE=cfg.predictor_type,
        DEVICE=_select_device(cfg.device),
        AE_EPOCHS=ae_epochs,
        AE_FRAME_BATCH_SIZE=ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=ae_max_frames_per_epoch,
    )
    print(f"Using video forecasting compute device: {ml_tda.get_runtime_device()}")

    return VideoContext(
        dataset_config=run_config,
        x_train=x_train,
        x_test=x_test,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        predictor_epochs=predictor_epochs,
        learning_rate=learning_rate,
        predictor_learning_rate=predictor_learning_rate,
        real_tda_scale=real_tda_scale,
        real_tda_bins=real_tda_bins,
        horizon=horizon,
        ae_epochs=ae_epochs,
        ae_frame_batch_size=ae_frame_batch_size,
        ae_max_frames_per_epoch=ae_max_frames_per_epoch,
    )


def run_real_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get("REAL_TDA_MODES", ["none", "h0", "h1", "both"]),
    )
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(5))))
    print(
        f"Real-TDA config: seeds={seeds}, modes={modes}, "
        f"horizon={context.horizon}"
    )

    rows = []
    for seed in tqdm_progress_bar(seeds, desc="real_tda seeds", total=len(seeds), leave=True):
        print(f"\n================ real-tda seed={seed} ================")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Train or refresh the shared encoder once per seed, then all modes
        # below load that frozen checkpoint. This keeps mode comparisons fair
        # when --retrain-encoder is enabled.
        _load_or_train_baseline_autoencoder(cfg, context, seed)

        for mode in modes:
            ml_tda.configure_runtime(EXPERIMENT_SEED=seed, TDA_MODE=mode)
            base_mode, _ = parse_tda_control_mode(mode)
            use_tda = base_mode != "none"
            if base_mode == "none":
                tda_dim = 0
            elif base_mode in {"h0", "h1"}:
                tda_dim = context.real_tda_bins
            else:
                tda_dim = 2 * context.real_tda_bins
            input_dim = context.latent_dim + tda_dim

            print(f"\n-------- DATASET={cfg.dataset} seed={seed} TDA_MODE={mode} --------")
            encoder = SpatialEncoder(latent_dim=context.latent_dim)
            decoder = make_spatial_decoder(
                cfg.decoder_type,
                latent_dim=context.latent_dim,
                output_size=context.x_train.shape[-2:],
            )
            model = make_predictor(input_dim=input_dim, hidden_dim=context.hidden_dim, predictor_type=cfg.predictor_type)
            load_or_train_model(
                encoder,
                decoder,
                model,
                context.x_train,
                False,
                cfg.retrain_predictor,
                use_tda,
                context.predictor_learning_rate,
            )
            test_mse, per_frame_mse = test_predictor(model, encoder, context.x_test, use_tda)
            total_test_features, test_z_features = ml_tda.build_real_tda_features(
                context.x_test,
                encoder,
                use_tda,
                split_name="Test metrics",
            )
            device = ml_tda.get_runtime_device()
            model.to(device).eval()
            with torch.no_grad():
                pred_z = model(total_test_features[:-context.horizon].to(device)).cpu()
            target_z = test_z_features[context.horizon:].cpu()
            latent_mse, latent_r2 = _mse_r2(pred_z, target_z)
            row = {
                "dataset": cfg.dataset,
                "seed": seed,
                "mode": mode,
                "test_mse": float(latent_mse),
                "latent_r2": float(latent_r2),
            }
            rows.append(row)
            print("real-TDA summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=ml_tda_latent.latent_result_metric_columns(results_df),
        sort_metric="test_mse",
    )
    print("\nReal-TDA per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nReal-TDA mean +/- std by mode:")
    print(summary_df)
    return results_df, summary_df


def _geo_model_namespace(dataset: str, geo_lambda: float) -> str:
    return f"{dataset}_geoae_lam{geo_lambda:g}"


def _pwgeo_model_namespace(cfg: RunConfig) -> str:
    return (
        f"{cfg.dataset}_pwgeoae_lam{cfg.pwgeo_ae_lambda:g}_k{cfg.pwgeo_knn}_"
        f"blend{cfg.pwgeo_blend:g}_h0{cfg.pwgeo_h0_weight:g}_h1{cfg.pwgeo_h1_weight:g}"
    )


def _mixedgeo_model_namespace(cfg: RunConfig, signature: str) -> str:
    return f"{cfg.dataset}_mixedgeo_lam{cfg.geo_ae_lambda:g}_{signature}"


def _routed_mixedgeo_model_namespace(cfg: RunConfig, signature: str) -> str:
    """Namespace routed checkpoints separately from fixed mixed-GeoAE models."""
    return (
        f"{cfg.dataset}_routed_mixedgeo_geo{cfg.geo_ae_lambda:g}_"
        f"route{cfg.route_lambda:g}_k{cfg.route_knn}_temp{cfg.route_temperature:g}_{signature}"
    )


def _topo_model_namespace(dataset: str, topo_lambda: float, topo_distance: str) -> str:
    return f"{dataset}_topoae_lam{topo_lambda:g}_{topo_distance}"


def _autoencoder_tag(seed: int, x_train: torch.Tensor, latent_dim: int) -> str:
    return (
        f"seed{seed}_T{x_train.shape[0]}_B{x_train.shape[1]}_"
        f"H{x_train.shape[-2]}_W{x_train.shape[-1]}_latent{latent_dim}"
    )


def _decoder_suffix(decoder_type: str) -> str:
    decoder_type = str(decoder_type).lower().strip()
    return "" if decoder_type == "mlp" else f"_dec{decoder_type}"


def _baseline_autoencoder_paths(
    dataset: str,
    seed: int,
    context: VideoContext,
    decoder_type: str = "mlp",
) -> tuple[Path, Path]:
    tag = f"{_autoencoder_tag(seed, context.x_train, context.latent_dim)}{_decoder_suffix(decoder_type)}"
    model_dir = Path("models") / dataset
    encoder_dir = model_dir / "encoders"
    decoder_dir = model_dir / "decoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    decoder_dir.mkdir(parents=True, exist_ok=True)
    return encoder_dir / f"encoder_{tag}.pt", decoder_dir / f"decoder_{tag}.pt"


def _load_or_train_baseline_autoencoder(
    cfg: RunConfig,
    context: VideoContext,
    seed: int,
) -> tuple[SpatialEncoder, SpatialDecoder, Path, Path]:
    encoder = SpatialEncoder(latent_dim=context.latent_dim)
    decoder = make_spatial_decoder(cfg.decoder_type, latent_dim=context.latent_dim, output_size=context.x_train.shape[-2:])
    encoder_path, decoder_path = _baseline_autoencoder_paths(cfg.dataset, seed, context, cfg.decoder_type)
    device = ml_tda.get_runtime_device()

    if encoder_path.exists() and decoder_path.exists() and not cfg.retrain_encoder:
        print(f"Loading baseline AE encoder: {encoder_path}")
        print(f"Loading baseline AE decoder: {decoder_path}")
        encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
    elif encoder_path.exists() and not decoder_path.exists() and not cfg.retrain_encoder:
        print(f"Loading baseline AE encoder: {encoder_path}")
        print(f"Baseline AE decoder missing; training decoder only: {decoder_path}")
        encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        ml_tda.train_decoder_for_encoder(encoder, decoder, context.x_train, ae_epochs=context.ae_epochs)
        torch.save(decoder.state_dict(), decoder_path)
        print(f"Saved baseline AE decoder: {decoder_path}")
    else:
        reason = "retraining" if encoder_path.exists() or decoder_path.exists() else "missing; training once"
        print(f"Baseline AE {reason}: {encoder_path} | {decoder_path}")
        ml_tda.pretrain_spatial_encoder(encoder, decoder, context.x_train, ae_epochs=context.ae_epochs)
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)
        print(f"Saved baseline AE encoder: {encoder_path}")
        print(f"Saved baseline AE decoder: {decoder_path}")

    encoder.to(device).eval()
    decoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False
    return encoder, decoder, encoder_path, decoder_path


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ae_real_tda_predictor_path(
    model_namespace: str,
    predictor_dir: str,
    seed: int,
    mode: str,
    context: VideoContext,
    input_dim: int,
    predictor_type: str = "lstm",
) -> Path:
    model_dir = Path("models") / model_namespace / predictor_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_mode{mode}_pred{context.horizon}_"
        f"T{context.x_train.shape[0]}_B{context.x_train.shape[1]}_"
        f"latent{context.latent_dim}_input{input_dim}_{predictor_type}"
    )
    return model_dir / f"{tag}.pt"


def _decode_z_predictor_path(
    model_namespace: str,
    seed: int,
    context: VideoContext,
    decoder_type: str = "mlp",
    predictor_type: str = "lstm",
) -> Path:
    model_dir = Path("models") / model_namespace / "decode_z_predictors"
    model_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_pred{context.horizon}_"
        f"T{context.x_train.shape[0]}_B{context.x_train.shape[1]}_"
        f"latent{context.latent_dim}_{predictor_type}{_decoder_suffix(decoder_type)}"
    )
    return model_dir / f"model_{tag}.pt"


def _train_or_load_decode_z_predictor(
    cfg: RunConfig,
    context: VideoContext,
    model_namespace: str,
    seed: int,
    train_z: torch.Tensor,
) -> tuple[TopologicalPredictor, Path]:
    device = ml_tda.get_runtime_device()
    model = make_predictor(input_dim=context.latent_dim, hidden_dim=context.hidden_dim, predictor_type=cfg.predictor_type).to(device)
    model_path = _decode_z_predictor_path(
        model_namespace,
        seed,
        context,
        cfg.decoder_type,
        cfg.predictor_type,
    )

    if model_path.exists() and not cfg.retrain_predictor:
        print(f"Loading decode-z predictor: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model, model_path

    if context.horizon >= train_z.shape[0]:
        raise ValueError(
            f"horizon={context.horizon} must be smaller than sequence length "
            f"{train_z.shape[0]}"
        )

    reason = "retraining" if model_path.exists() else "missing; training once"
    print(f"Decode-z predictor {reason}: {model_path}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=context.predictor_learning_rate)
    criterion = torch.nn.MSELoss()
    x_train = train_z[:-context.horizon].to(device)
    y_train = train_z[context.horizon:].to(device)

    epochs = range(1, context.predictor_epochs + 1)
    for epoch in tqdm_progress_bar(epochs, desc=f"decode-z seed={seed}", total=context.predictor_epochs):
        model.train()
        optimizer.zero_grad()
        pred_z = model(x_train)
        loss = criterion(pred_z, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch == context.predictor_epochs:
            print(f"decode-z seed={seed} epoch {epoch}/{context.predictor_epochs} latent_mse={loss.item():.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"Saved decode-z predictor: {model_path}")
    return model, model_path


def _decode_sequence(decoder: SpatialDecoder, z_seq: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    device = ml_tda.get_runtime_device()
    decoder.to(device).eval()
    flat_z = z_seq.reshape(-1, z_seq.shape[-1])
    decoded = []
    with torch.no_grad():
        for start in range(0, len(flat_z), batch_size):
            decoded.append(decoder(flat_z[start:start + batch_size].to(device)).cpu())
    return torch.cat(decoded, dim=0).reshape(z_seq.shape[0], z_seq.shape[1], 1, *decoder.output_size)


def _evaluate_decode_z_to_future_x(
    model: TopologicalPredictor,
    decoder: SpatialDecoder,
    test_z: torch.Tensor,
    x_test: torch.Tensor,
    context: VideoContext,
) -> dict[str, float | torch.Tensor]:
    device = ml_tda.get_runtime_device()
    model.to(device).eval()
    with torch.no_grad():
        pred_z = model(test_z[:-context.horizon].to(device)).cpu()

    pred_x = _decode_sequence(decoder, pred_z)
    # This is the best image the same decoder can produce when it receives the
    # *true* future latent.  Comparing against it separates decoder error from
    # latent-forecast error.
    oracle_x = _decode_sequence(decoder, test_z[context.horizon:])
    target_x = ml_tda_pixel.target_frames(x_test[context.horizon:]).cpu()
    weighted_mse, pixel_mse, fg_mse, bg_mse = ml_tda_pixel.pixel_losses(pred_x, target_x)
    _, pixel_r2 = _mse_r2(pred_x, target_x)
    _, oracle_pixel_mse, _, _ = ml_tda_pixel.pixel_losses(oracle_x, target_x)
    forecast_to_oracle_mse = torch.mean((pred_x - oracle_x) ** 2)
    per_time_pixel_mse = ((pred_x - target_x) ** 2).mean(dim=(1, 2, 3, 4))
    return {
        "weighted_mse": float(weighted_mse),
        "pixel_mse": float(pixel_mse),
        "pixel_r2": float(pixel_r2),
        "foreground_mse": float(fg_mse),
        "background_mse": float(bg_mse),
        "oracle_pixel_mse": float(oracle_pixel_mse),
        "forecast_to_oracle_mse": float(forecast_to_oracle_mse),
        "per_time_pixel_mse": per_time_pixel_mse,
        "target_x": target_x,
        "oracle_x": oracle_x,
        "pred_x": pred_x,
    }


def _save_decode_comparison_figure(
    metrics: dict[str, float | torch.Tensor],
    *,
    dataset: str,
    encoder: str,
    seed: int,
) -> Path:
    """Save target/oracle/forecast frames for a decoded-latent run."""
    import matplotlib.pyplot as plt

    target = metrics["target_x"]
    oracle = metrics["oracle_x"]
    forecast = metrics["pred_x"]
    assert isinstance(target, torch.Tensor)
    assert isinstance(oracle, torch.Tensor)
    assert isinstance(forecast, torch.Tensor)

    # Show three well-separated times from the first test sequence.  Rows are
    # time points; columns expose where the error enters the pipeline.
    time_ids = np.linspace(0, target.shape[0] - 1, min(3, target.shape[0]), dtype=int)
    fig, axes = plt.subplots(len(time_ids), 4, figsize=(8.0, 2.0 * len(time_ids)), squeeze=False)
    for row, time_id in enumerate(time_ids):
        target_frame = target[time_id, 0, 0].numpy()
        oracle_frame = oracle[time_id, 0, 0].numpy()
        forecast_frame = forecast[time_id, 0, 0].numpy()
        panels = (
            (target_frame, "Target"),
            (oracle_frame, r"Oracle $D(z_{t+h})$"),
            (forecast_frame, r"Forecast $D(\hat z_{t+h})$"),
            (np.abs(forecast_frame - target_frame), "Absolute error"),
        )
        for col, (frame, title) in enumerate(panels):
            axes[row, col].imshow(frame, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, col].set_axis_off()
            if row == 0:
                axes[row, col].set_title(title, fontsize=9)
            if col == 0:
                axes[row, col].text(
                    -0.08, 0.5, f"t={time_id}", transform=axes[row, col].transAxes,
                    rotation=90, va="center", ha="right", fontsize=8,
                )
    fig.suptitle(f"Decoded forecast: {dataset} ({encoder}, seed {seed})", fontsize=11)
    fig.tight_layout()
    output_dir = Path("images") / "decoded_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_{encoder}_seed{seed}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _geo_feature_dim(mode: str, context: VideoContext) -> tuple[bool, int]:
    base_mode, _ = parse_tda_control_mode(mode)
    if base_mode == "none":
        return False, context.latent_dim
    if base_mode in {"h0", "h1"}:
        return True, context.latent_dim + context.real_tda_bins
    if base_mode == "both":
        return True, context.latent_dim + 2 * context.real_tda_bins
    raise ValueError(f"Unknown TDA mode: {mode}")


def _train_or_load_geo_real_tda_predictor(
    *,
    cfg: RunConfig,
    context: VideoContext,
    encoder: SpatialEncoder,
    model_namespace: str,
    seed: int,
    mode: str,
) -> tuple[TopologicalPredictor, Path]:
    use_tda, input_dim = _geo_feature_dim(mode, context)
    model = make_predictor(input_dim=input_dim, hidden_dim=context.hidden_dim, predictor_type=cfg.predictor_type)
    model_path = _ae_real_tda_predictor_path(
        model_namespace,
        "geo_real_tda_predictors",
        seed,
        mode,
        context,
        input_dim,
        cfg.predictor_type,
    )
    device = ml_tda.get_runtime_device()

    if model_path.exists() and not cfg.retrain_predictor:
        print(f"Loading geo-real-tda predictor: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return model, model_path

    reason = "retraining" if model_path.exists() else "missing; training once"
    print(f"Geo-real-TDA predictor {reason}: {model_path}")
    model.to(device)
    ml_tda.train_predictor(
        model,
        encoder,
        context.x_train,
        predictor_epochs=context.predictor_epochs,
        use_tda=use_tda,
        learning_rate=context.predictor_learning_rate,
    )
    torch.save(model.state_dict(), model_path)
    print(f"Saved geo-real-tda predictor: {model_path}")
    return model, model_path


def _load_geo_encoder_for_seed(
    cfg: RunConfig,
    context: VideoContext,
    seed: int,
) -> tuple[SpatialEncoder, Path, str]:
    geo_epochs = int(_override(cfg.geo_ae_epochs, context.dataset_config.get("GEO_AE_EPOCHS", 3)))
    model_namespace = _geo_model_namespace(cfg.dataset, cfg.geo_ae_lambda)
    encoder, encoder_path = ml_tda_geoae.load_or_train_geo_encoder(
        context.x_train,
        dataset_name=cfg.dataset,
        model_namespace=model_namespace,
        seed=seed,
        latent_dim=context.latent_dim,
        geo_lambda=cfg.geo_ae_lambda,
        ae_epochs=geo_epochs,
        frame_batch_size=context.ae_frame_batch_size or 256,
        max_frames_per_epoch=context.ae_max_frames_per_epoch,
        pair_batch_size=cfg.geo_ae_pair_batch_size,
        retrain=cfg.retrain_encoder,
        decoder_type=cfg.decoder_type,
    )
    return encoder, encoder_path, model_namespace


def _load_pwgeo_encoder_for_seed(
    cfg: RunConfig,
    context: VideoContext,
    seed: int,
) -> tuple[SpatialEncoder, Path, str]:
    epochs = int(_override(cfg.pwgeo_ae_epochs, context.dataset_config.get("GEO_AE_EPOCHS", 3)))
    model_namespace = _pwgeo_model_namespace(cfg)
    encoder, encoder_path = ml_tda_pwgeoae.load_or_train_pwgeo_encoder(
        context.x_train,
        dataset_name=cfg.dataset,
        model_namespace=model_namespace,
        seed=seed,
        latent_dim=context.latent_dim,
        pw_lambda=cfg.pwgeo_ae_lambda,
        blend=cfg.pwgeo_blend,
        knn=cfg.pwgeo_knn,
        h0_weight=cfg.pwgeo_h0_weight,
        h1_weight=cfg.pwgeo_h1_weight,
        ae_epochs=epochs,
        frame_batch_size=context.ae_frame_batch_size or 256,
        max_frames_per_epoch=context.ae_max_frames_per_epoch,
        pair_batch_size=cfg.pwgeo_ae_pair_batch_size,
        retrain=cfg.retrain_encoder,
        decoder_type=cfg.decoder_type,
    )
    return encoder, encoder_path, model_namespace


def _load_mixedgeo_encoder_for_seed(
    cfg: RunConfig,
    context: VideoContext,
    seed: int,
    signature: str,
) -> tuple[SpatialEncoder, Path, str]:
    epochs = int(_override(cfg.geo_ae_epochs, context.dataset_config.get("GEO_AE_EPOCHS", 3)))
    model_namespace = _mixedgeo_model_namespace(cfg, signature)
    encoder, encoder_path = ml_tda_mixedgeo.load_or_train_mixedgeo_encoder(
        context.x_train,
        dataset_name=cfg.dataset,
        model_namespace=model_namespace,
        signature=signature,
        seed=seed,
        latent_dim=context.latent_dim,
        geo_lambda=cfg.geo_ae_lambda,
        ae_epochs=epochs,
        frame_batch_size=context.ae_frame_batch_size or 256,
        max_frames_per_epoch=context.ae_max_frames_per_epoch,
        pair_batch_size=cfg.geo_ae_pair_batch_size,
        retrain=cfg.retrain_encoder,
        decoder_type=cfg.decoder_type,
    )
    return encoder, encoder_path, model_namespace


def _load_routed_mixedgeo_encoder_for_seed(
    cfg: RunConfig,
    context: VideoContext,
    seed: int,
    signature: str,
) -> tuple[SpatialEncoder, Path, str]:
    """Load one persistence-routed product-manifold encoder."""
    epochs = int(_override(cfg.geo_ae_epochs, context.dataset_config.get("GEO_AE_EPOCHS", 3)))
    model_namespace = _routed_mixedgeo_model_namespace(cfg, signature)
    encoder, encoder_path = ml_tda_routed_mixedgeo.load_or_train_routed_mixedgeo_encoder(
        context.x_train,
        dataset_name=cfg.dataset,
        model_namespace=model_namespace,
        signature=signature,
        seed=seed,
        latent_dim=context.latent_dim,
        geo_lambda=cfg.geo_ae_lambda,
        route_lambda=cfg.route_lambda,
        route_knn=cfg.route_knn,
        route_temperature=cfg.route_temperature,
        ae_epochs=epochs,
        frame_batch_size=context.ae_frame_batch_size or 256,
        max_frames_per_epoch=context.ae_max_frames_per_epoch,
        pair_batch_size=cfg.geo_ae_pair_batch_size,
        retrain=cfg.retrain_encoder,
        decoder_type=cfg.decoder_type,
    )
    return encoder, encoder_path, model_namespace


def _load_topo_encoder_for_seed(
    cfg: RunConfig,
    context: VideoContext,
    seed: int,
) -> tuple[SpatialEncoder, Path, str]:
    topo_epochs = int(_override(cfg.topo_ae_epochs, context.dataset_config.get("TOPO_AE_EPOCHS", 3)))
    model_namespace = _topo_model_namespace(cfg.dataset, cfg.topo_ae_lambda, cfg.topo_ae_distance)
    encoder, encoder_path = ml_tda_topoae.load_or_train_topo_encoder(
        context.x_train,
        dataset_name=cfg.dataset,
        model_namespace=model_namespace,
        seed=seed,
        latent_dim=context.latent_dim,
        topo_lambda=cfg.topo_ae_lambda,
        topo_distance=cfg.topo_ae_distance,
        ae_epochs=topo_epochs,
        frame_batch_size=context.ae_frame_batch_size or 256,
        max_frames_per_epoch=context.ae_max_frames_per_epoch,
        pair_batch_size=cfg.topo_ae_pair_batch_size,
        retrain=cfg.retrain_encoder,
        decoder_type=cfg.decoder_type,
    )
    return encoder, encoder_path, model_namespace


def _pca_spectrum_statistics(z: torch.Tensor) -> dict[str, float]:
    """Full standardized PCA spectrum diagnostics for a latent point cloud."""
    flat = z.detach().cpu().numpy().reshape(-1, z.shape[-1]).astype(np.float64)
    scale = flat.std(axis=0)
    scale[scale == 0] = 1.0
    flat = (flat - flat.mean(axis=0)) / scale
    singular_values = np.linalg.svd(flat, full_matrices=False, compute_uv=False)
    variance = singular_values**2
    variance /= variance.sum()
    cumulative = np.cumsum(variance)
    positive = variance[variance > 0]

    def k_for(threshold: float) -> int:
        return int(np.searchsorted(cumulative, threshold, side="left") + 1)

    return {
        "pc1_pc2_pct": 100.0 * float(variance[:2].sum()),
        "k50": float(k_for(0.50)),
        "k90": float(k_for(0.90)),
        "k95": float(k_for(0.95)),
        "effective_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
        "participation_ratio": float(1.0 / np.sum(variance**2)),
    }


def run_geo_latent_spectrum(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/load GeoAE and measure its test-latent PCA spectrum only."""
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(5))))
    device = _select_device(cfg.device)
    ml_tda.configure_runtime(
        DATASET=_geo_model_namespace(cfg.dataset, cfg.geo_ae_lambda),
        LATENT_DIM=context.latent_dim,
        DEVICE=device,
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
    )
    rows = []
    for seed in tqdm_progress_bar(seeds, desc="GeoAE PCA spectrum seeds", total=len(seeds), leave=True):
        _set_all_seeds(seed)
        encoder, encoder_path, _ = _load_geo_encoder_for_seed(cfg, context, seed)
        z = ml_tda_latent.encode_video_to_z(
            context.x_test,
            encoder,
            frame_batch_size=context.ae_frame_batch_size or 1024,
        )
        rows.append(
            {
                "dataset": cfg.dataset,
                "encoder": "GeoAE",
                "geo_ae_lambda": cfg.geo_ae_lambda,
                "seed": seed,
                "latent_dim": context.latent_dim,
                "n_test_points": int(z.shape[0] * z.shape[1]),
                "encoder_path": str(encoder_path),
                **_pca_spectrum_statistics(z),
            }
        )
    results = pd.DataFrame(rows)
    metrics = ["pc1_pc2_pct", "k50", "k90", "k95", "effective_rank", "participation_ratio"]
    summary = ml_tda.summarize_metric_runs(
        results,
        group_cols="geo_ae_lambda",
        metric_cols=metrics,
        sort_metric="effective_rank",
    )
    return results, summary


def run_geo_real_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get("REAL_TDA_MODES", ["none", "h0", "h1", "both"]),
    )
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(5))))
    print(
        f"Geo-real-TDA config: dataset={cfg.dataset}, seeds={seeds}, modes={modes}, "
        f"geo_ae_lambda={cfg.geo_ae_lambda}, horizon={context.horizon}"
    )

    rows = []
    for seed in tqdm_progress_bar(seeds, desc="geo_real_tda seeds", total=len(seeds), leave=True):
        print(f"\n================ geo-real-tda seed={seed} ================")
        _set_all_seeds(seed)
        encoder, encoder_path, model_namespace = _load_geo_encoder_for_seed(cfg, context, seed)

        for mode in modes:
            ml_tda.configure_runtime(EXPERIMENT_SEED=seed, TDA_MODE=mode)
            use_tda, _ = _geo_feature_dim(mode, context)
            print(f"\n-------- DATASET={cfg.dataset} seed={seed} GEO_TDA_MODE={mode} --------")
            model, model_path = _train_or_load_geo_real_tda_predictor(
                cfg=cfg,
                context=context,
                encoder=encoder,
                model_namespace=model_namespace,
                seed=seed,
                mode=mode,
            )
            test_mse, per_frame_mse = test_predictor(model, encoder, context.x_test, use_tda)
            total_test_features, test_z_features = ml_tda.build_real_tda_features(
                context.x_test,
                encoder,
                use_tda,
                split_name="Geo Test metrics",
            )
            device = ml_tda.get_runtime_device()
            model.to(device).eval()
            with torch.no_grad():
                pred_z = model(total_test_features[:-context.horizon].to(device)).cpu()
            target_z = test_z_features[context.horizon:].cpu()
            latent_mse, latent_r2 = _mse_r2(pred_z, target_z)
            row = {
                "dataset": cfg.dataset,
                "encoder": "geo_ae",
                "geo_ae_lambda": cfg.geo_ae_lambda,
                "seed": seed,
                "mode": mode,
                "test_mse": float(latent_mse),
                "latent_r2": float(latent_r2),
                "encoder_path": str(encoder_path),
                "model_path": str(model_path),
            }
            rows.append(row)
            print("geo-real-tda summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=ml_tda_latent.latent_result_metric_columns(results_df),
        sort_metric="test_mse",
    )
    print("\nGeo-real-TDA per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nGeo-real-TDA mean +/- std by mode:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def run_topo_real_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get("REAL_TDA_MODES", ["none", "h0", "h1", "both"]),
    )
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(5))))
    print(
        f"Topo-real-TDA config: dataset={cfg.dataset}, seeds={seeds}, modes={modes}, "
        f"topo_ae_lambda={cfg.topo_ae_lambda}, topo_ae_distance={cfg.topo_ae_distance}, "
        f"horizon={context.horizon}"
    )

    rows = []
    for seed in tqdm_progress_bar(seeds, desc="topo_real_tda seeds", total=len(seeds), leave=True):
        print(f"\n================ topo-real-tda seed={seed} ================")
        _set_all_seeds(seed)
        encoder, encoder_path, model_namespace = _load_topo_encoder_for_seed(cfg, context, seed)

        for mode in modes:
            ml_tda.configure_runtime(EXPERIMENT_SEED=seed, TDA_MODE=mode)
            use_tda, input_dim = _geo_feature_dim(mode, context)
            print(f"\n-------- DATASET={cfg.dataset} seed={seed} TOPO_TDA_MODE={mode} --------")
            model = make_predictor(input_dim=input_dim, hidden_dim=context.hidden_dim, predictor_type=cfg.predictor_type)
            model_path = _ae_real_tda_predictor_path(
                model_namespace,
                "topo_real_tda_predictors",
                seed,
                mode,
                context,
                input_dim,
                cfg.predictor_type,
            )
            device = ml_tda.get_runtime_device()
            if model_path.exists() and not cfg.retrain_predictor:
                print(f"Loading topo-real-tda predictor: {model_path}")
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
            else:
                reason = "retraining" if model_path.exists() else "missing; training once"
                print(f"Topo-real-TDA predictor {reason}: {model_path}")
                model.to(device)
                ml_tda.train_predictor(
                    model,
                    encoder,
                    context.x_train,
                    predictor_epochs=context.predictor_epochs,
                    use_tda=use_tda,
                    learning_rate=context.predictor_learning_rate,
                )
                torch.save(model.state_dict(), model_path)
                print(f"Saved topo-real-tda predictor: {model_path}")

            test_mse, per_frame_mse = test_predictor(model, encoder, context.x_test, use_tda)
            total_test_features, test_z_features = ml_tda.build_real_tda_features(
                context.x_test,
                encoder,
                use_tda,
                split_name="Topo Test metrics",
            )
            model.to(device).eval()
            with torch.no_grad():
                pred_z = model(total_test_features[:-context.horizon].to(device)).cpu()
            target_z = test_z_features[context.horizon:].cpu()
            latent_mse, latent_r2 = _mse_r2(pred_z, target_z)
            row = {
                "dataset": cfg.dataset,
                "encoder": "topo_ae",
                "topo_ae_lambda": cfg.topo_ae_lambda,
                "topo_ae_distance": cfg.topo_ae_distance,
                "seed": seed,
                "mode": mode,
                "test_mse": float(latent_mse),
                "latent_r2": float(latent_r2),
                "encoder_path": str(encoder_path),
                "model_path": str(model_path),
            }
            rows.append(row)
            print("topo-real-tda summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=ml_tda_latent.latent_result_metric_columns(results_df),
        sort_metric="test_mse",
    )
    print("\nTopo-real-TDA per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTopo-real-TDA mean +/- std by mode:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def run_aux_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get("AUX_TDA_MODES", ["none", "aux_h0", "aux_h1", "aux_both"]),
    )
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(3))))
    aux_lambda = float(_override(cfg.aux_tda_lambda, context.dataset_config.get("AUX_TDA_LAMBDA", 0.1)))
    aux_epochs = int(_override(cfg.predictor_epochs, context.dataset_config.get("AUX_TDA_PREDICTOR_EPOCHS", context.predictor_epochs)))
    aux_lr = float(_override(cfg.predictor_learning_rate, context.dataset_config.get("AUX_TDA_LR", context.predictor_learning_rate)))

    ml_tda.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        REAL_TDA_SCALE=context.real_tda_scale,
        REAL_TDA_BINS=context.real_tda_bins,
        HORIZON=context.horizon,
        DEVICE=_select_device(cfg.device),
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
        PREDICTOR_TYPE=cfg.predictor_type,
    )
    ml_tda_aux.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        HORIZON=context.horizon,
        REAL_TDA_BINS=context.real_tda_bins,
        REAL_TDA_SCALE=context.real_tda_scale,
        AUX_TDA_LAMBDA=aux_lambda,
        AUX_TDA_PREDICTOR_EPOCHS=aux_epochs,
        AUX_TDA_LR=aux_lr,
        AUX_TDA_RETRAIN_ENCODER=cfg.retrain_encoder,
        AUX_TDA_RETRAIN_PREDICTOR=cfg.retrain_predictor,
    )
    print(
        f"Aux TDA config: dataset={cfg.dataset}, seeds={seeds}, modes={modes}, "
        f"lambda={aux_lambda}, predictor_epochs={aux_epochs}, lr={aux_lr}, "
        f"horizon={context.horizon}"
    )
    results_df, summary_df, _ = ml_tda_aux.run_aux_tda_experiment(
        X_train=context.x_train,
        X_test=context.x_test,
        seeds=seeds,
        modes=modes,
        display_fn=None,
    )
    return results_df, summary_df


def run_latent_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get(
            "LATENT_TDA_MODES",
            ["z", "z_h0", "z_h1", "z_both"],
        ),
    )
    modes = ml_tda_latent.canonicalize_latent_tda_modes(modes)
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(5))))
    window = int(_override(cfg.latent_tda_window, context.dataset_config.get("LATENT_TDA_WINDOW", 20)))
    bins = int(_override(cfg.latent_tda_bins, context.dataset_config.get("LATENT_TDA_BINS", 16)))
    latent_epochs = int(_override(cfg.predictor_epochs, context.dataset_config.get("LATENT_TDA_PREDICTOR_EPOCHS", context.predictor_epochs)))
    latent_lr = context.predictor_learning_rate
    max_train = _override(cfg.latent_tda_max_train, context.dataset_config.get("LATENT_TDA_MAX_TRAIN"))
    max_test = _override(cfg.latent_tda_max_test, context.dataset_config.get("LATENT_TDA_MAX_TEST"))
    recompute_features = cfg.recompute_latent_tda_features or context.dataset_config.get(
        "RECOMPUTE_LATENT_TDA_FEATURES",
        False,
    )

    ml_tda.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        REAL_TDA_SCALE=context.real_tda_scale,
        REAL_TDA_BINS=context.real_tda_bins,
        HORIZON=context.horizon,
        DEVICE=_select_device(cfg.device),
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
        PREDICTOR_TYPE=cfg.predictor_type,
    )
    ml_tda_latent.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        RETRAIN_ENCODER=cfg.retrain_encoder,
        HORIZON=context.horizon,
        LATENT_TDA_WINDOW=window,
        LATENT_TDA_BINS=bins,
        LATENT_TDA_PREDICTOR_EPOCHS=latent_epochs,
        LATENT_TDA_LR=latent_lr,
        RETRAIN_LATENT_TDA_PREDICTOR=cfg.retrain_predictor,
        RECOMPUTE_LATENT_TDA_FEATURES=recompute_features,
    )
    print(
        f"Latent TDA config: dataset={cfg.dataset}, seeds={seeds}, modes={modes}, "
        f"window={window}, bins={bins}, predictor_epochs={latent_epochs}, lr={latent_lr}, "
        f"horizon={context.horizon}, "
        f"max_train={max_train}, max_test={max_test}, recompute_features={recompute_features}"
    )
    results_df, summary_df, _ = ml_tda_latent.run_latent_tda_trajectory_experiment(
        X_train=context.x_train,
        X_test=context.x_test,
        run_seeds=seeds,
        modes=modes,
        max_train=max_train,
        max_test=max_test,
        display_fn=None,
        compute_diagnostics=not cfg.profile_run,
    )
    return results_df, summary_df


def run_latent_stability(cfg: RunConfig, context: VideoContext, encoder_kind: str = "ae") -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(3))))
    window = int(_override(cfg.latent_tda_window, context.dataset_config.get("LATENT_TDA_WINDOW", 20)))
    bins = int(_override(cfg.latent_tda_bins, context.dataset_config.get("LATENT_TDA_BINS", 16)))
    max_train = _override(cfg.latent_tda_max_train, context.dataset_config.get("LATENT_TDA_MAX_TRAIN"))
    max_test = _override(cfg.latent_tda_max_test, context.dataset_config.get("LATENT_TDA_MAX_TEST"))
    noise_levels = _override(cfg.stability_noise_levels, [0.01, 0.03, 0.05, 0.10])

    ml_tda.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        REAL_TDA_SCALE=context.real_tda_scale,
        REAL_TDA_BINS=context.real_tda_bins,
        HORIZON=context.horizon,
        DEVICE=_select_device(cfg.device),
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
        PREDICTOR_TYPE=cfg.predictor_type,
    )
    ml_tda_latent.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        RETRAIN_ENCODER=cfg.retrain_encoder,
        HORIZON=context.horizon,
        LATENT_TDA_WINDOW=window,
        LATENT_TDA_BINS=bins,
        RECOMPUTE_LATENT_TDA_FEATURES=False,
    )
    encoder_loader = None
    encoder_label = "ae"
    if encoder_kind == "geo":
        encoder_label = "geo_ae"

        def encoder_loader(seed, _x_subset, _x_train_for_tag):
            encoder, _, _ = _load_geo_encoder_for_seed(cfg, context, seed)
            return encoder

    elif encoder_kind == "topo":
        encoder_label = "topo_ae"

        def encoder_loader(seed, _x_subset, _x_train_for_tag):
            encoder, _, _ = _load_topo_encoder_for_seed(cfg, context, seed)
            return encoder

    elif encoder_kind != "ae":
        raise ValueError(f"Unknown stability encoder kind: {encoder_kind}")

    print(
        f"Latent stability config: dataset={cfg.dataset}, encoder={encoder_label}, seeds={seeds}, "
        f"noise_levels={noise_levels}, window={window}, bins={bins}, "
        f"max_train={max_train}, max_test={max_test}"
    )
    return ml_tda_latent.run_latent_stability_diagnostic(
        X_train=context.x_train,
        X_test=context.x_test,
        run_seeds=seeds,
        noise_levels=noise_levels,
        max_train=max_train,
        max_test=max_test,
        encoder_loader=encoder_loader,
        encoder_label=encoder_label,
    )


def run_representation_fidelity(
    cfg: RunConfig,
    context: VideoContext,
    encoder_kind: str = "ae",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(5))))
    window = int(_override(cfg.latent_tda_window, context.dataset_config.get("LATENT_TDA_WINDOW", 20)))
    ml_tda.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        HORIZON=context.horizon,
        DEVICE=_select_device(cfg.device),
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
    )
    ml_tda_latent.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        HORIZON=context.horizon,
        RETRAIN_ENCODER=cfg.retrain_encoder,
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
    )

    if encoder_kind == "ae":
        encoder_label = "ae"

        def encoder_loader(seed):
            return ml_tda_latent.load_or_train_shared_encoder_for_latent_tda(
                seed, context.x_train, context.x_train
            )

    elif encoder_kind == "geo":
        encoder_label = "geo_ae"

        def encoder_loader(seed):
            encoder, _, _ = _load_geo_encoder_for_seed(cfg, context, seed)
            return encoder

    elif encoder_kind == "topo":
        encoder_label = "topo_ae"

        def encoder_loader(seed):
            encoder, _, _ = _load_topo_encoder_for_seed(cfg, context, seed)
            return encoder

    else:
        raise ValueError(f"Unknown fidelity encoder kind: {encoder_kind}")

    print(
        f"Representation fidelity config: dataset={cfg.dataset}, encoder={encoder_label}, "
        f"seeds={seeds}, window={window}, n_windows={cfg.fidelity_windows}"
    )
    results_df, summary_df = ml_tda_latent.run_representation_fidelity_diagnostic(
        X_test=context.x_test,
        run_seeds=seeds,
        encoder_loader=encoder_loader,
        encoder_label=encoder_label,
        window=window,
        n_windows=cfg.fidelity_windows,
    )
    if encoder_kind == "geo":
        results_df["geo_ae_lambda"] = cfg.geo_ae_lambda
        summary_df["geo_ae_lambda"] = cfg.geo_ae_lambda
    elif encoder_kind == "topo":
        results_df["topo_ae_lambda"] = cfg.topo_ae_lambda
        results_df["topo_ae_distance"] = cfg.topo_ae_distance
        summary_df["topo_ae_lambda"] = cfg.topo_ae_lambda
        summary_df["topo_ae_distance"] = cfg.topo_ae_distance
    print("\nRepresentation-fidelity per-seed summaries:")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return results_df, summary_df


def run_geo_latent_tda(
    cfg: RunConfig,
    context: VideoContext,
    *,
    encoder_kind: str = "geo",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get(
            "LATENT_TDA_MODES",
            ["z", "z_h0", "z_h1", "z_both"],
        ),
    )
    modes = ml_tda_latent.canonicalize_latent_tda_modes(modes)
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(5))))
    window = int(_override(cfg.latent_tda_window, context.dataset_config.get("LATENT_TDA_WINDOW", 20)))
    bins = int(_override(cfg.latent_tda_bins, context.dataset_config.get("LATENT_TDA_BINS", 16)))
    latent_epochs = int(_override(cfg.predictor_epochs, context.dataset_config.get("LATENT_TDA_PREDICTOR_EPOCHS", context.predictor_epochs)))
    latent_lr = context.predictor_learning_rate
    max_train = _override(cfg.latent_tda_max_train, context.dataset_config.get("LATENT_TDA_MAX_TRAIN"))
    max_test = _override(cfg.latent_tda_max_test, context.dataset_config.get("LATENT_TDA_MAX_TEST"))
    is_pwgeo = encoder_kind == "pwgeo"
    is_mixedgeo = encoder_kind in {"mixedgeo", "routed_mixedgeo", "manifold_mixedgeo"}
    is_routed = encoder_kind == "routed_mixedgeo"
    is_manifold_ph = encoder_kind == "manifold_mixedgeo"
    signature = cfg.manifold_signatures[0] if is_mixedgeo else None
    if is_manifold_ph:
        ml_tda_mixedgeo.parse_manifold_signature(signature, context.latent_dim)
        model_namespace = f"{_mixedgeo_model_namespace(cfg, signature)}_vr_{cfg.vr_distance}"
    elif is_routed:
        ml_tda_mixedgeo.parse_manifold_signature(signature, context.latent_dim)
        model_namespace = _routed_mixedgeo_model_namespace(cfg, signature)
    elif is_mixedgeo:
        ml_tda_mixedgeo.parse_manifold_signature(signature, context.latent_dim)
        model_namespace = _mixedgeo_model_namespace(cfg, signature)
    elif is_pwgeo:
        model_namespace = _pwgeo_model_namespace(cfg)
    else:
        model_namespace = _geo_model_namespace(cfg.dataset, cfg.geo_ae_lambda)
    recompute_features = cfg.recompute_latent_tda_features or context.dataset_config.get(
        "RECOMPUTE_LATENT_TDA_FEATURES",
        False,
    )

    ml_tda.configure_runtime(
        DATASET=model_namespace,
        LATENT_DIM=context.latent_dim,
        REAL_TDA_SCALE=context.real_tda_scale,
        REAL_TDA_BINS=context.real_tda_bins,
        HORIZON=context.horizon,
        DEVICE=_select_device(cfg.device),
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
        PREDICTOR_TYPE=cfg.predictor_type,
    )
    ml_tda_latent.configure_runtime(
        DATASET=model_namespace,
        # Euclidean/product VR controls retain separate caches/checkpoints but
        # must initialize predictors identically for a paired comparison.
        PREDICTOR_SEED_NAMESPACE=(
            _mixedgeo_model_namespace(cfg, signature) if is_manifold_ph else model_namespace
        ),
        DETERMINISTIC_PREDICTOR=is_manifold_ph,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        RETRAIN_ENCODER=cfg.retrain_encoder,
        HORIZON=context.horizon,
        LATENT_TDA_WINDOW=window,
        LATENT_TDA_BINS=bins,
        LATENT_TDA_PREDICTOR_EPOCHS=latent_epochs,
        LATENT_TDA_LR=latent_lr,
        RETRAIN_LATENT_TDA_PREDICTOR=cfg.retrain_predictor,
        RECOMPUTE_LATENT_TDA_FEATURES=recompute_features,
    )
    print(
        f"{'Manifold-PH-Mixed-Geo' if is_manifold_ph else ('Routed-Mixed-Geo' if is_routed else ('Mixed-Geo' if is_mixedgeo else ('PW-Geo' if is_pwgeo else 'Geo')))} latent-TDA config: dataset={cfg.dataset}, namespace={model_namespace}, "
        f"seeds={seeds}, modes={modes}, lambda={cfg.pwgeo_ae_lambda if is_pwgeo else cfg.geo_ae_lambda}, "
        f"signature={signature}, "
        f"route_lambda={cfg.route_lambda if is_routed else None}, "
        f"route_knn={cfg.route_knn if is_routed else None}, "
        f"route_temperature={cfg.route_temperature if is_routed else None}, "
        f"vr_distance={cfg.vr_distance if is_manifold_ph else None}, "
        f"window={window}, bins={bins}, predictor_epochs={latent_epochs}, lr={latent_lr}, "
        f"horizon={context.horizon}, "
        f"max_train={max_train}, max_test={max_test}, recompute_features={recompute_features}"
    )

    rows = []
    require_persistence = ml_tda_latent._latent_modes_need_persistence(modes)
    scenario_name = "manifold_mixed_geo_latent_tda" if is_manifold_ph else ("routed_mixed_geo_latent_tda" if is_routed else ("mixed_geo_latent_tda" if is_mixedgeo else ("pwgeo_latent_tda" if is_pwgeo else "geo_latent_tda")))
    for seed in tqdm_progress_bar(seeds, desc=f"{scenario_name} seeds", total=len(seeds), leave=True):
        print(f"\n================ {scenario_name} seed={seed} ================")
        _set_all_seeds(seed)
        if is_routed:
            encoder, encoder_path, _ = _load_routed_mixedgeo_encoder_for_seed(cfg, context, seed, signature)
        elif is_mixedgeo:
            encoder, encoder_path, _ = _load_mixedgeo_encoder_for_seed(cfg, context, seed, signature)
        elif is_pwgeo:
            encoder, encoder_path, _ = _load_pwgeo_encoder_for_seed(cfg, context, seed)
        else:
            encoder, encoder_path, _ = _load_geo_encoder_for_seed(cfg, context, seed)
        x_train_subset = ml_tda_latent.take_batch_subset(context.x_train, max_train, seed=seed)
        x_test_subset = ml_tda_latent.take_batch_subset(context.x_test, max_test, seed=seed + 1)
        if is_manifold_ph and require_persistence:
            train_payload = ml_tda_manifold_ph.load_or_compute_manifold_tda_features(
                seed, "train", x_train_subset, encoder, signature, cfg.vr_distance
            )
            test_payload = ml_tda_manifold_ph.load_or_compute_manifold_tda_features(
                seed, "test", x_test_subset, encoder, signature, cfg.vr_distance
            )
        else:
            train_payload = ml_tda_latent.load_or_compute_latent_tda_features(
                seed, "train", x_train_subset, encoder, require_persistence=require_persistence,
            )
            test_payload = ml_tda_latent.load_or_compute_latent_tda_features(
                seed, "test", x_test_subset, encoder, require_persistence=require_persistence,
            )
        diagnostics = (
            ml_tda_latent.latent_geometry_diagnostics(x_test_subset, test_payload["z"])
            if not cfg.profile_run
            else {}
        )

        for mode in modes:
            print(f"\n--- geo latent mode={mode} seed={seed} ---")
            train_features, test_features = ml_tda_latent.features_for_latent_tda_mode_pair(
                train_payload,
                test_payload,
                mode,
                train_control_seed=seed,
                test_control_seed=seed + 10_000,
            )
            paired_seed_namespace = (
                _mixedgeo_model_namespace(cfg, signature) if is_manifold_ph else None
            )
            model = ml_tda_latent.train_or_load_latent_tda_predictor(
                seed,
                mode,
                train_features,
                train_payload["z"],
                predictor_seed_namespace=paired_seed_namespace,
                deterministic=True if is_manifold_ph else None,
            )
            test_mse, per_frame_mse, latent_r2 = ml_tda_latent.eval_latent_tda_predictor(
                model,
                test_features,
                test_payload["z"],
            )
            row = {
                "dataset": cfg.dataset,
                "encoder": "routed_mixed_geo_ae" if is_routed else ("mixed_geo_ae" if is_mixedgeo else ("pwgeo_ae" if is_pwgeo else "geo_ae")),
                "manifold_signature": signature,
                "geo_ae_lambda": np.nan if is_pwgeo else cfg.geo_ae_lambda,
                "pwgeo_ae_lambda": cfg.pwgeo_ae_lambda if is_pwgeo else np.nan,
                "pwgeo_blend": float(cfg.pwgeo_blend) if is_pwgeo else np.nan,
                "route_lambda": cfg.route_lambda if is_routed else np.nan,
                "route_knn": cfg.route_knn if is_routed else np.nan,
                "route_temperature": cfg.route_temperature if is_routed else np.nan,
                "vr_distance": cfg.vr_distance if is_manifold_ph else "legacy_euclidean",
                "predictor_seed": ml_tda_latent.predictor_seed(
                    seed, mode, namespace=paired_seed_namespace
                ),
                "seed": seed,
                "mode": mode,
                "horizon": context.horizon,
                "test_mse": float(test_mse),
                "latent_r2": float(latent_r2),
                "encoder_path": str(encoder_path),
                **ml_tda_latent.topology_gate_statistics(model),
                **diagnostics,
            }
            rows.append(row)
            print("geo latent TDA summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols=["pwgeo_blend", "mode"] if is_pwgeo else "mode",
        metric_cols=ml_tda_latent.latent_result_metric_columns(results_df),
        sort_metric="test_mse",
    )
    print("\nGeo latent-TDA per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nGeo latent-TDA mean +/- std:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def run_pwgeo_latent_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    return run_geo_latent_tda(cfg, context, encoder_kind="pwgeo")


def run_triangle_curvature(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(3))))
    rows = []
    for knn in cfg.triangle_knn:
        for seed in seeds:
            row = ml_tda_triangle.triangle_curvature_diagnostic(
                context.x_train,
                latent_dim=context.latent_dim,
                knn=knn,
                n_samples=cfg.triangle_samples,
                max_points=cfg.triangle_max_points,
                flat_threshold=cfg.triangle_flat_threshold,
                seed=seed,
            )
            row["dataset"] = cfg.dataset
            rows.append(row)
            print("triangle-curvature:", row)
    results = pd.DataFrame(rows)
    summary = ml_tda.summarize_metric_runs(
        results,
        group_cols=["knn", "suggested_signature"],
        metric_cols=[
            "k_mean", "k_median", "negative_fraction", "flat_fraction",
            "positive_fraction", "largest_component_fraction",
        ],
    )
    print("\nTriangle-curvature per-seed results:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTriangle-curvature summary:")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    return results, summary


def _resolved_manifold_signatures(cfg: RunConfig, context: VideoContext) -> list[str]:
    """Return manual signatures or training-only triangle-shrinkage candidates."""
    if cfg.manifold_signature_policy == "manual":
        return cfg.manifold_signatures
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(3))))
    diagnostics = [
        ml_tda_triangle.triangle_curvature_diagnostic(
            context.x_train,
            latent_dim=context.latent_dim,
            knn=knn,
            n_samples=cfg.triangle_samples,
            max_points=cfg.triangle_max_points,
            flat_threshold=cfg.triangle_flat_threshold,
            seed=seed,
        )
        for knn in cfg.triangle_knn
        for seed in seeds
    ]
    negative = float(np.mean([row["negative_fraction"] for row in diagnostics]))
    flat = float(np.mean([row["flat_fraction"] for row in diagnostics]))
    positive = float(np.mean([row["positive_fraction"] for row in diagnostics]))
    diagnostic = ml_tda_triangle.suggested_signature(
        negative, flat, positive, context.latent_dim
    )
    candidates = ml_tda_triangle.softened_signature_candidates(
        diagnostic, context.latent_dim, cfg.signature_shrinkages
    )
    print(
        "Triangle-shrinkage signature policy: "
        f"fractions=(H={negative:.4f}, E={flat:.4f}, S={positive:.4f}), "
        f"diagnostic={diagnostic}, candidates={candidates}"
    )
    return candidates


def run_mixed_geo_latent_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_frames, summary_frames = [], []
    for signature in _resolved_manifold_signatures(cfg, context):
        signature_cfg = replace(cfg, manifold_signatures=[signature])
        results, summary = run_geo_latent_tda(signature_cfg, context, encoder_kind="mixedgeo")
        results["manifold_signature"] = signature
        summary["manifold_signature"] = signature
        result_frames.append(results)
        summary_frames.append(summary)
    # Preserve the named ``mode`` index; the shared output writer resets it into
    # a proper mode column when combining scenario summaries.
    return pd.concat(result_frames, ignore_index=True), pd.concat(summary_frames)


def run_routed_mixed_geo_latent_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep signatures for persistence-routed mixed-geometry encoders."""
    result_frames, summary_frames = [], []
    for signature in _resolved_manifold_signatures(cfg, context):
        signature_cfg = replace(cfg, manifold_signatures=[signature])
        results, summary = run_geo_latent_tda(signature_cfg, context, encoder_kind="routed_mixedgeo")
        results["manifold_signature"] = signature
        summary["manifold_signature"] = signature
        result_frames.append(results)
        summary_frames.append(summary)
    return pd.concat(result_frames, ignore_index=True), pd.concat(summary_frames)


def run_manifold_mixed_geo_latent_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep fixed product signatures using the requested VR distance."""
    result_frames, summary_frames = [], []
    for signature in _resolved_manifold_signatures(cfg, context):
        signature_cfg = replace(cfg, manifold_signatures=[signature])
        results, summary = run_geo_latent_tda(signature_cfg, context, encoder_kind="manifold_mixedgeo")
        results["manifold_signature"] = signature
        results["vr_distance"] = cfg.vr_distance
        summary["manifold_signature"] = signature
        summary["vr_distance"] = cfg.vr_distance
        result_frames.append(results)
        summary_frames.append(summary)
    return pd.concat(result_frames, ignore_index=True), pd.concat(summary_frames)


def run_topo_latent_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get(
            "LATENT_TDA_MODES",
            ["z", "z_h0", "z_h1", "z_both"],
        ),
    )
    modes = ml_tda_latent.canonicalize_latent_tda_modes(modes)
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(5))))
    window = int(_override(cfg.latent_tda_window, context.dataset_config.get("LATENT_TDA_WINDOW", 20)))
    bins = int(_override(cfg.latent_tda_bins, context.dataset_config.get("LATENT_TDA_BINS", 16)))
    latent_epochs = int(_override(cfg.predictor_epochs, context.dataset_config.get("LATENT_TDA_PREDICTOR_EPOCHS", context.predictor_epochs)))
    latent_lr = context.predictor_learning_rate
    max_train = _override(cfg.latent_tda_max_train, context.dataset_config.get("LATENT_TDA_MAX_TRAIN"))
    max_test = _override(cfg.latent_tda_max_test, context.dataset_config.get("LATENT_TDA_MAX_TEST"))
    model_namespace = _topo_model_namespace(cfg.dataset, cfg.topo_ae_lambda, cfg.topo_ae_distance)
    recompute_features = cfg.recompute_latent_tda_features or context.dataset_config.get(
        "RECOMPUTE_LATENT_TDA_FEATURES",
        False,
    )

    ml_tda.configure_runtime(
        DATASET=model_namespace,
        LATENT_DIM=context.latent_dim,
        REAL_TDA_SCALE=context.real_tda_scale,
        REAL_TDA_BINS=context.real_tda_bins,
        HORIZON=context.horizon,
        DEVICE=_select_device(cfg.device),
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
        PREDICTOR_TYPE=cfg.predictor_type,
    )
    ml_tda_latent.configure_runtime(
        DATASET=model_namespace,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        RETRAIN_ENCODER=cfg.retrain_encoder,
        HORIZON=context.horizon,
        LATENT_TDA_WINDOW=window,
        LATENT_TDA_BINS=bins,
        LATENT_TDA_PREDICTOR_EPOCHS=latent_epochs,
        LATENT_TDA_LR=latent_lr,
        RETRAIN_LATENT_TDA_PREDICTOR=cfg.retrain_predictor,
        RECOMPUTE_LATENT_TDA_FEATURES=recompute_features,
    )
    print(
        f"Topo latent-TDA config: dataset={cfg.dataset}, namespace={model_namespace}, "
        f"seeds={seeds}, modes={modes}, topo_ae_lambda={cfg.topo_ae_lambda}, "
        f"topo_ae_distance={cfg.topo_ae_distance}, window={window}, bins={bins}, "
        f"predictor_epochs={latent_epochs}, lr={latent_lr}, horizon={context.horizon}, "
        f"max_train={max_train}, max_test={max_test}, recompute_features={recompute_features}"
    )

    rows = []
    require_persistence = ml_tda_latent._latent_modes_need_persistence(modes)
    for seed in tqdm_progress_bar(seeds, desc="topo_latent_tda seeds", total=len(seeds), leave=True):
        print(f"\n================ topo latent TDA seed={seed} ================")
        _set_all_seeds(seed)
        encoder, encoder_path, _ = _load_topo_encoder_for_seed(cfg, context, seed)
        x_train_subset = ml_tda_latent.take_batch_subset(context.x_train, max_train, seed=seed)
        x_test_subset = ml_tda_latent.take_batch_subset(context.x_test, max_test, seed=seed + 1)
        train_payload = ml_tda_latent.load_or_compute_latent_tda_features(
            seed,
            "train",
            x_train_subset,
            encoder,
            require_persistence=require_persistence,
        )
        test_payload = ml_tda_latent.load_or_compute_latent_tda_features(
            seed,
            "test",
            x_test_subset,
            encoder,
            require_persistence=require_persistence,
        )
        diagnostics = (
            ml_tda_latent.latent_geometry_diagnostics(x_test_subset, test_payload["z"])
            if not cfg.profile_run
            else {}
        )

        for mode in modes:
            print(f"\n--- topo latent mode={mode} seed={seed} ---")
            train_features, test_features = ml_tda_latent.features_for_latent_tda_mode_pair(
                train_payload,
                test_payload,
                mode,
                train_control_seed=seed,
                test_control_seed=seed + 10_000,
            )
            model = ml_tda_latent.train_or_load_latent_tda_predictor(
                seed,
                mode,
                train_features,
                train_payload["z"],
            )
            test_mse, per_frame_mse, latent_r2 = ml_tda_latent.eval_latent_tda_predictor(
                model,
                test_features,
                test_payload["z"],
            )
            row = {
                "dataset": cfg.dataset,
                "encoder": "topo_ae",
                "topo_ae_lambda": cfg.topo_ae_lambda,
                "topo_ae_distance": cfg.topo_ae_distance,
                "seed": seed,
                "mode": mode,
                "horizon": context.horizon,
                "test_mse": float(test_mse),
                "latent_r2": float(latent_r2),
                "encoder_path": str(encoder_path),
                **ml_tda_latent.topology_gate_statistics(model),
                **diagnostics,
            }
            rows.append(row)
            print("topo latent TDA summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=ml_tda_latent.latent_result_metric_columns(results_df),
        sort_metric="test_mse",
    )
    print("\nTopo latent-TDA per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTopo latent-TDA mean +/- std:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def _run_representation_latent_tda(
    cfg: RunConfig,
    context: VideoContext,
    *,
    rep_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get(
            "LATENT_TDA_MODES",
            ["z", "z_h0", "z_h1", "z_both"],
        ),
    )
    modes = ml_tda_latent.canonicalize_latent_tda_modes(modes)
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(5))))
    window = int(_override(cfg.latent_tda_window, context.dataset_config.get("LATENT_TDA_WINDOW", 20)))
    bins = int(_override(cfg.latent_tda_bins, context.dataset_config.get("LATENT_TDA_BINS", 16)))
    latent_epochs = int(_override(cfg.predictor_epochs, context.dataset_config.get("LATENT_TDA_PREDICTOR_EPOCHS", context.predictor_epochs)))
    latent_lr = context.predictor_learning_rate
    max_train = _override(cfg.latent_tda_max_train, context.dataset_config.get("LATENT_TDA_MAX_TRAIN"))
    max_test = _override(cfg.latent_tda_max_test, context.dataset_config.get("LATENT_TDA_MAX_TEST"))
    if rep_kind == "vae":
        model_namespace = f"{cfg.dataset}_vae_beta{cfg.vae_beta:g}"
        scenario_name = "vae_latent_tda"
    elif rep_kind == "byol":
        model_namespace = f"{cfg.dataset}_byol"
        scenario_name = "byol_latent_tda"
    else:
        raise ValueError(f"Unknown representation kind: {rep_kind}")
    recompute_features = cfg.recompute_latent_tda_features or context.dataset_config.get(
        "RECOMPUTE_LATENT_TDA_FEATURES",
        False,
    )

    ml_tda.configure_runtime(
        DATASET=model_namespace,
        LATENT_DIM=context.latent_dim,
        REAL_TDA_SCALE=context.real_tda_scale,
        REAL_TDA_BINS=context.real_tda_bins,
        HORIZON=context.horizon,
        DEVICE=_select_device(cfg.device),
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
        PREDICTOR_TYPE=cfg.predictor_type,
    )
    ml_tda_latent.configure_runtime(
        DATASET=model_namespace,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        RETRAIN_ENCODER=cfg.retrain_encoder,
        HORIZON=context.horizon,
        LATENT_TDA_WINDOW=window,
        LATENT_TDA_BINS=bins,
        LATENT_TDA_PREDICTOR_EPOCHS=latent_epochs,
        LATENT_TDA_LR=latent_lr,
        RETRAIN_LATENT_TDA_PREDICTOR=cfg.retrain_predictor,
        RECOMPUTE_LATENT_TDA_FEATURES=recompute_features,
    )
    print(
        f"{scenario_name} config: dataset={cfg.dataset}, namespace={model_namespace}, "
        f"seeds={seeds}, modes={modes}, predictor_type={cfg.predictor_type}, "
        f"window={window}, bins={bins}, predictor_epochs={latent_epochs}, lr={latent_lr}"
    )

    rows = []
    for seed in tqdm_progress_bar(seeds, desc=f"{scenario_name} seeds", total=len(seeds), leave=True):
        print(f"\n================ {scenario_name} seed={seed} ================")
        _set_all_seeds(seed)
        x_train_subset = ml_tda_latent.take_batch_subset(context.x_train, max_train, seed=seed)
        x_test_subset = ml_tda_latent.take_batch_subset(context.x_test, max_test, seed=seed + 1)
        if rep_kind == "vae":
            encoder, encoder_path = ml_tda_repr.load_or_train_vae_encoder(
                x_train_subset,
                dataset_name=cfg.dataset,
                seed=seed,
                latent_dim=context.latent_dim,
                ae_epochs=int(_override(cfg.vae_epochs, context.ae_epochs)),
                beta=cfg.vae_beta,
                frame_batch_size=context.ae_frame_batch_size,
                max_frames_per_epoch=context.ae_max_frames_per_epoch,
                retrain=cfg.retrain_encoder,
            )
        else:
            encoder, encoder_path = ml_tda_repr.load_or_train_byol_encoder(
                x_train_subset,
                dataset_name=cfg.dataset,
                seed=seed,
                latent_dim=context.latent_dim,
                byol_epochs=int(_override(cfg.byol_epochs, context.ae_epochs)),
                noise_std=cfg.byol_noise_std,
                frame_batch_size=context.ae_frame_batch_size,
                max_frames_per_epoch=context.ae_max_frames_per_epoch,
                retrain=cfg.retrain_encoder,
            )
        train_payload = ml_tda_latent.load_or_compute_latent_tda_features(seed, "train", x_train_subset, encoder)
        test_payload = ml_tda_latent.load_or_compute_latent_tda_features(seed, "test", x_test_subset, encoder)
        diagnostics = ml_tda_latent.latent_geometry_diagnostics(x_test_subset, test_payload["z"])

        for mode in modes:
            print(f"\n--- {scenario_name} mode={mode} seed={seed} ---")
            train_features, test_features = ml_tda_latent.features_for_latent_tda_mode_pair(
                train_payload,
                test_payload,
                mode,
                train_control_seed=seed,
                test_control_seed=seed + 10_000,
            )
            model = ml_tda_latent.train_or_load_latent_tda_predictor(seed, mode, train_features, train_payload["z"])
            test_mse, _, latent_r2 = ml_tda_latent.eval_latent_tda_predictor(model, test_features, test_payload["z"])
            row = {
                "dataset": cfg.dataset,
                "encoder": rep_kind,
                "seed": seed,
                "mode": mode,
                "horizon": context.horizon,
                "test_mse": float(test_mse),
                "latent_r2": float(latent_r2),
                "encoder_path": str(encoder_path),
                **ml_tda_latent.topology_gate_statistics(model),
                **diagnostics,
            }
            rows.append(row)
            print(f"{scenario_name} summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=ml_tda_latent.latent_result_metric_columns(results_df),
        sort_metric="test_mse",
    )
    print(f"\n{scenario_name} mean +/- std:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def run_vae_latent_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _run_representation_latent_tda(cfg, context, rep_kind="vae")


def run_byol_latent_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _run_representation_latent_tda(cfg, context, rep_kind="byol")


def _load_or_compute_dinov2_payload(
    *,
    namespace: str,
    repo: str,
    seed: int,
    split_name: str,
    video_tensor: torch.Tensor,
    device: str,
    batch_size: int,
    image_size: int,
    window: int,
    bins: int,
    recompute: bool,
) -> dict[str, Any]:
    cache_path = ml_tda_dinov2.dinov2_feature_cache_path(
        namespace=namespace,
        repo=repo,
        seed=seed,
        split_name=split_name,
        video_tensor=video_tensor,
        image_size=image_size,
    )
    if cache_path.exists() and not recompute:
        print(f"Loading DINOv2 feature cache: {cache_path}")
        try:
            return torch.load(cache_path, map_location="cpu")
        except Exception:
            return torch.load(cache_path, map_location="cpu", weights_only=False)

    print(f"Computing DINOv2 z + latent-window TDA for {split_name}...")
    z = ml_tda_dinov2.encode_video_tensor_with_dinov2(
        video_tensor,
        repo=repo,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
    )
    h0, h1, diagrams = ml_tda_latent.latent_window_betti_features(
        z,
        window=window,
        n_bins=bins,
        return_diagrams=True,
    )
    payload = {"z": z, "h0": h0, "h1": h1, "diagrams": diagrams}
    torch.save(payload, cache_path)
    print(f"Saved DINOv2 feature cache: {cache_path}")
    return payload


def _load_or_compute_dinov2_payload_from_model(
    *,
    namespace: str,
    repo: str,
    seed: int,
    split_name: str,
    video_tensor: torch.Tensor,
    model: torch.nn.Module,
    device: str,
    batch_size: int,
    image_size: int,
    window: int,
    bins: int,
    recompute: bool,
) -> dict[str, Any]:
    cache_path = ml_tda_dinov2.dinov2_feature_cache_path(
        namespace=namespace,
        repo=repo,
        seed=seed,
        split_name=split_name,
        video_tensor=video_tensor,
        image_size=image_size,
    )
    if cache_path.exists() and not recompute:
        print(f"Loading fine-tuned DINOv2 feature cache: {cache_path}")
        try:
            return torch.load(cache_path, map_location="cpu")
        except Exception:
            return torch.load(cache_path, map_location="cpu", weights_only=False)

    print(f"Computing fine-tuned DINOv2 z + latent-window TDA for {split_name}...")
    z = ml_tda_dinov2.encode_video_tensor_with_dinov2_model(
        model,
        video_tensor,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
    )
    h0, h1, diagrams = ml_tda_latent.latent_window_betti_features(
        z,
        window=window,
        n_bins=bins,
        return_diagrams=True,
    )
    payload = {"z": z, "h0": h0, "h1": h1, "diagrams": diagrams}
    torch.save(payload, cache_path)
    print(f"Saved fine-tuned DINOv2 feature cache: {cache_path}")
    return payload


def run_dinov2_latent_tda(
    cfg: RunConfig,
    context: VideoContext,
    *,
    fine_tune: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get(
            "LATENT_TDA_MODES",
            ["z", "z_fuse_h1", "z_fuse_pi_h1", "z_fuse_perslay_h1"],
        ),
    )
    modes = ml_tda_latent.canonicalize_latent_tda_modes(modes)
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(3))))
    window = int(_override(cfg.latent_tda_window, context.dataset_config.get("LATENT_TDA_WINDOW", 20)))
    bins = int(_override(cfg.latent_tda_bins, context.dataset_config.get("LATENT_TDA_BINS", 16)))
    latent_epochs = int(_override(cfg.predictor_epochs, context.dataset_config.get("LATENT_TDA_PREDICTOR_EPOCHS", context.predictor_epochs)))
    latent_lr = context.predictor_learning_rate
    max_train = _override(cfg.latent_tda_max_train, context.dataset_config.get("LATENT_TDA_MAX_TRAIN"))
    max_test = _override(cfg.latent_tda_max_test, context.dataset_config.get("LATENT_TDA_MAX_TEST"))
    repo = str(cfg.dinov2_repo)
    scenario_name = "dinov2_finetune_latent_tda" if fine_tune else "dinov2_latent_tda"
    if fine_tune:
        lr_tag = f"{cfg.dinov2_encoder_lr:g}".replace(".", "p").replace("-", "m")
        dinov2_namespace = (
            f"{cfg.dataset}_dinov2_ft_{repo.replace('/', '__')}_"
            f"blocks{cfg.dinov2_trainable_blocks}_epochs{cfg.dinov2_finetune_epochs}_elr{lr_tag}"
        )
    else:
        dinov2_namespace = f"{cfg.dataset}_dinov2_{repo.replace('/', '__')}"
    recompute_features = cfg.recompute_latent_tda_features or context.dataset_config.get(
        "RECOMPUTE_LATENT_TDA_FEATURES",
        False,
    )
    selected_device = _select_device(cfg.device)
    dinov2_device = selected_device.type if cfg.device == "auto" else cfg.device

    print(
        f"{scenario_name} config: dataset={cfg.dataset}, namespace={dinov2_namespace}, repo={repo}, "
        f"seeds={seeds}, modes={modes}, image_size={cfg.dinov2_image_size}, "
        f"batch_size={cfg.dinov2_batch_size}, window={window}, bins={bins}, "
        f"predictor_epochs={latent_epochs}, lr={latent_lr}, horizon={context.horizon}"
    )

    rows = []
    for seed in tqdm_progress_bar(seeds, desc=f"{scenario_name} seeds", total=len(seeds), leave=True):
        print(f"\n================ {scenario_name} seed={seed} ================")
        _set_all_seeds(seed)
        x_train_subset = ml_tda_latent.take_batch_subset(context.x_train, max_train, seed=seed)
        x_test_subset = ml_tda_latent.take_batch_subset(context.x_test, max_test, seed=seed + 1)
        if fine_tune:
            dinov2_model, encoder_path = ml_tda_dinov2.load_or_finetune_dinov2_model(
                x_train_subset,
                namespace=dinov2_namespace,
                repo=repo,
                seed=seed,
                device=dinov2_device,
                batch_size=cfg.dinov2_batch_size,
                image_size=cfg.dinov2_image_size,
                trainable_blocks=cfg.dinov2_trainable_blocks,
                epochs=cfg.dinov2_finetune_epochs,
                encoder_lr=cfg.dinov2_encoder_lr,
                predictor_lr=latent_lr,
                hidden_dim=context.hidden_dim,
                horizon=context.horizon,
                clip_batch_size=cfg.dinov2_clip_batch_size,
                retrain=cfg.retrain_encoder,
            )
            train_payload = _load_or_compute_dinov2_payload_from_model(
                namespace=dinov2_namespace,
                repo=repo,
                seed=seed,
                split_name="train",
                video_tensor=x_train_subset,
                model=dinov2_model,
                device=dinov2_device,
                batch_size=cfg.dinov2_batch_size,
                image_size=cfg.dinov2_image_size,
                window=window,
                bins=bins,
                recompute=recompute_features,
            )
            test_payload = _load_or_compute_dinov2_payload_from_model(
                namespace=dinov2_namespace,
                repo=repo,
                seed=seed,
                split_name="test",
                video_tensor=x_test_subset,
                model=dinov2_model,
                device=dinov2_device,
                batch_size=cfg.dinov2_batch_size,
                image_size=cfg.dinov2_image_size,
                window=window,
                bins=bins,
                recompute=recompute_features,
            )
        else:
            encoder_path = ""
            train_payload = _load_or_compute_dinov2_payload(
                namespace=dinov2_namespace,
                repo=repo,
                seed=seed,
                split_name="train",
                video_tensor=x_train_subset,
                device=dinov2_device,
                batch_size=cfg.dinov2_batch_size,
                image_size=cfg.dinov2_image_size,
                window=window,
                bins=bins,
                recompute=recompute_features,
            )
            test_payload = _load_or_compute_dinov2_payload(
                namespace=dinov2_namespace,
                repo=repo,
                seed=seed,
                split_name="test",
                video_tensor=x_test_subset,
                device=dinov2_device,
                batch_size=cfg.dinov2_batch_size,
                image_size=cfg.dinov2_image_size,
                window=window,
                bins=bins,
                recompute=recompute_features,
            )
        dinov2_latent_dim = int(train_payload["z"].shape[-1])
        ml_tda.configure_runtime(
            DATASET=dinov2_namespace,
            LATENT_DIM=dinov2_latent_dim,
            REAL_TDA_SCALE=context.real_tda_scale,
            REAL_TDA_BINS=context.real_tda_bins,
            HORIZON=context.horizon,
            DEVICE=selected_device,
            PREDICTOR_TYPE=cfg.predictor_type,
        )
        ml_tda_latent.configure_runtime(
            DATASET=dinov2_namespace,
            LATENT_DIM=dinov2_latent_dim,
            HIDDEN_DIM=context.hidden_dim,
            RETRAIN_ENCODER=False,
            HORIZON=context.horizon,
            LATENT_TDA_WINDOW=window,
            LATENT_TDA_BINS=bins,
            LATENT_TDA_PREDICTOR_EPOCHS=latent_epochs,
            LATENT_TDA_LR=latent_lr,
            RETRAIN_LATENT_TDA_PREDICTOR=cfg.retrain_predictor,
            RECOMPUTE_LATENT_TDA_FEATURES=recompute_features,
        )
        diagnostics = ml_tda_latent.latent_geometry_diagnostics(x_test_subset, test_payload["z"])

        for mode in modes:
            print(f"\n--- {scenario_name} mode={mode} seed={seed} ---")
            train_features, test_features = ml_tda_latent.features_for_latent_tda_mode_pair(
                train_payload,
                test_payload,
                mode,
                train_control_seed=seed,
                test_control_seed=seed + 10_000,
            )
            model = ml_tda_latent.train_or_load_latent_tda_predictor(seed, mode, train_features, train_payload["z"])
            test_mse, _, latent_r2 = ml_tda_latent.eval_latent_tda_predictor(model, test_features, test_payload["z"])
            row = {
                "dataset": cfg.dataset,
                "encoder": "dinov2",
                "seed": seed,
                "mode": mode,
                "horizon": context.horizon,
                "test_mse": float(test_mse),
                "latent_r2": float(latent_r2),
                "dinov2_repo": repo,
                "dinov2_image_size": int(cfg.dinov2_image_size),
                "dinov2_trainable_blocks": int(cfg.dinov2_trainable_blocks) if fine_tune else 0,
                "encoder_path": str(encoder_path),
                "latent_dim": dinov2_latent_dim,
                **ml_tda_latent.topology_gate_statistics(model),
                **diagnostics,
            }
            rows.append(row)
            print(f"{scenario_name} summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=ml_tda_latent.latent_result_metric_columns(results_df),
        sort_metric="test_mse",
    )
    print(f"\n{scenario_name} mean +/- std:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def _load_or_compute_clip_payload(
    *,
    namespace: str,
    repo: str,
    seed: int,
    split_name: str,
    video_tensor: torch.Tensor,
    device: str,
    batch_size: int,
    image_size: int,
    window: int,
    bins: int,
    recompute: bool,
) -> dict[str, Any]:
    cache_path = ml_tda_clip.clip_feature_cache_path(
        namespace=namespace,
        repo=repo,
        seed=seed,
        split_name=split_name,
        video_tensor=video_tensor,
        image_size=image_size,
    )
    if cache_path.exists() and not recompute:
        print(f"Loading CLIP feature cache: {cache_path}")
        try:
            return torch.load(cache_path, map_location="cpu")
        except Exception:
            return torch.load(cache_path, map_location="cpu", weights_only=False)

    print(f"Computing CLIP z + latent-window TDA for {split_name}...")
    z = ml_tda_clip.encode_video_tensor_with_clip(
        video_tensor,
        repo=repo,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
    )
    h0, h1, diagrams = ml_tda_latent.latent_window_betti_features(
        z,
        window=window,
        n_bins=bins,
        return_diagrams=True,
    )
    payload = {"z": z, "h0": h0, "h1": h1, "diagrams": diagrams}
    torch.save(payload, cache_path)
    print(f"Saved CLIP feature cache: {cache_path}")
    return payload


def run_clip_latent_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get(
            "LATENT_TDA_MODES",
            ["z", "z_fuse_h1", "z_fuse_pi_h1", "z_fuse_perslay_h1"],
        ),
    )
    modes = ml_tda_latent.canonicalize_latent_tda_modes(modes)
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(3))))
    window = int(_override(cfg.latent_tda_window, context.dataset_config.get("LATENT_TDA_WINDOW", 20)))
    bins = int(_override(cfg.latent_tda_bins, context.dataset_config.get("LATENT_TDA_BINS", 16)))
    latent_epochs = int(_override(cfg.predictor_epochs, context.dataset_config.get("LATENT_TDA_PREDICTOR_EPOCHS", context.predictor_epochs)))
    latent_lr = context.predictor_learning_rate
    max_train = _override(cfg.latent_tda_max_train, context.dataset_config.get("LATENT_TDA_MAX_TRAIN"))
    max_test = _override(cfg.latent_tda_max_test, context.dataset_config.get("LATENT_TDA_MAX_TEST"))
    repo = str(cfg.clip_repo)
    clip_namespace = f"{cfg.dataset}_clip_{repo.replace('/', '__')}"
    recompute_features = cfg.recompute_latent_tda_features or context.dataset_config.get(
        "RECOMPUTE_LATENT_TDA_FEATURES",
        False,
    )
    selected_device = _select_device(cfg.device)
    clip_device = selected_device.type if cfg.device == "auto" else cfg.device

    print(
        f"clip_latent_tda config: dataset={cfg.dataset}, namespace={clip_namespace}, repo={repo}, "
        f"seeds={seeds}, modes={modes}, image_size={cfg.clip_image_size}, "
        f"batch_size={cfg.clip_batch_size}, window={window}, bins={bins}, "
        f"predictor_epochs={latent_epochs}, lr={latent_lr}, horizon={context.horizon}"
    )

    rows = []
    for seed in tqdm_progress_bar(seeds, desc="clip_latent_tda seeds", total=len(seeds), leave=True):
        print(f"\n================ clip_latent_tda seed={seed} ================")
        _set_all_seeds(seed)
        x_train_subset = ml_tda_latent.take_batch_subset(context.x_train, max_train, seed=seed)
        x_test_subset = ml_tda_latent.take_batch_subset(context.x_test, max_test, seed=seed + 1)
        train_payload = _load_or_compute_clip_payload(
            namespace=clip_namespace,
            repo=repo,
            seed=seed,
            split_name="train",
            video_tensor=x_train_subset,
            device=clip_device,
            batch_size=cfg.clip_batch_size,
            image_size=cfg.clip_image_size,
            window=window,
            bins=bins,
            recompute=recompute_features,
        )
        test_payload = _load_or_compute_clip_payload(
            namespace=clip_namespace,
            repo=repo,
            seed=seed,
            split_name="test",
            video_tensor=x_test_subset,
            device=clip_device,
            batch_size=cfg.clip_batch_size,
            image_size=cfg.clip_image_size,
            window=window,
            bins=bins,
            recompute=recompute_features,
        )
        clip_latent_dim = int(train_payload["z"].shape[-1])
        ml_tda.configure_runtime(
            DATASET=clip_namespace,
            LATENT_DIM=clip_latent_dim,
            REAL_TDA_SCALE=context.real_tda_scale,
            REAL_TDA_BINS=context.real_tda_bins,
            HORIZON=context.horizon,
            DEVICE=selected_device,
            PREDICTOR_TYPE=cfg.predictor_type,
        )
        ml_tda_latent.configure_runtime(
            DATASET=clip_namespace,
            LATENT_DIM=clip_latent_dim,
            HIDDEN_DIM=context.hidden_dim,
            RETRAIN_ENCODER=False,
            HORIZON=context.horizon,
            LATENT_TDA_WINDOW=window,
            LATENT_TDA_BINS=bins,
            LATENT_TDA_PREDICTOR_EPOCHS=latent_epochs,
            LATENT_TDA_LR=latent_lr,
            RETRAIN_LATENT_TDA_PREDICTOR=cfg.retrain_predictor,
            RECOMPUTE_LATENT_TDA_FEATURES=recompute_features,
        )
        diagnostics = ml_tda_latent.latent_geometry_diagnostics(x_test_subset, test_payload["z"])

        for mode in modes:
            print(f"\n--- clip_latent_tda mode={mode} seed={seed} ---")
            train_features, test_features = ml_tda_latent.features_for_latent_tda_mode_pair(
                train_payload,
                test_payload,
                mode,
                train_control_seed=seed,
                test_control_seed=seed + 10_000,
            )
            model = ml_tda_latent.train_or_load_latent_tda_predictor(seed, mode, train_features, train_payload["z"])
            test_mse, _, latent_r2 = ml_tda_latent.eval_latent_tda_predictor(model, test_features, test_payload["z"])
            row = {
                "dataset": cfg.dataset,
                "encoder": "clip",
                "seed": seed,
                "mode": mode,
                "horizon": context.horizon,
                "test_mse": float(test_mse),
                "latent_r2": float(latent_r2),
                "clip_repo": repo,
                "clip_image_size": int(cfg.clip_image_size),
                "latent_dim": clip_latent_dim,
                **ml_tda_latent.topology_gate_statistics(model),
                **diagnostics,
            }
            rows.append(row)
            print("clip_latent_tda summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=ml_tda_latent.latent_result_metric_columns(results_df),
        sort_metric="test_mse",
    )
    print("\nclip_latent_tda mean +/- std:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def _load_or_compute_vjepa_payload(
    *,
    namespace: str,
    repo: str,
    seed: int,
    split_name: str,
    video_tensor: torch.Tensor,
    device: str,
    batch_size: int,
    num_frames: int,
    window: int,
    bins: int,
    recompute: bool,
) -> dict[str, Any]:
    cache_path = ml_tda_vjepa.vjepa_feature_cache_path(
        namespace=namespace,
        repo=repo,
        seed=seed,
        split_name=split_name,
        video_tensor=video_tensor,
        num_frames=num_frames,
    )
    if cache_path.exists() and not recompute:
        print(f"Loading V-JEPA feature cache: {cache_path}")
        try:
            return torch.load(cache_path, map_location="cpu")
        except Exception:
            return torch.load(cache_path, map_location="cpu", weights_only=False)

    print(f"Computing V-JEPA z + latent-window TDA for {split_name}...")
    z = ml_tda_vjepa.encode_video_tensor_with_vjepa(
        video_tensor,
        repo=repo,
        device=device,
        batch_size=batch_size,
        num_frames=num_frames,
    )
    h0, h1, diagrams = ml_tda_latent.latent_window_betti_features(
        z,
        window=window,
        n_bins=bins,
        return_diagrams=True,
    )
    payload = {"z": z, "h0": h0, "h1": h1, "diagrams": diagrams}
    torch.save(payload, cache_path)
    print(f"Saved V-JEPA feature cache: {cache_path}")
    return payload


def run_vjepa_latent_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get(
            "LATENT_TDA_MODES",
            ["z", "z_temporal_stats", "z_temporal_stats_h1"],
        ),
    )
    modes = ml_tda_latent.canonicalize_latent_tda_modes(modes)
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(3))))
    window = int(_override(cfg.latent_tda_window, context.dataset_config.get("LATENT_TDA_WINDOW", 20)))
    bins = int(_override(cfg.latent_tda_bins, context.dataset_config.get("LATENT_TDA_BINS", 16)))
    latent_epochs = int(_override(cfg.predictor_epochs, context.dataset_config.get("LATENT_TDA_PREDICTOR_EPOCHS", context.predictor_epochs)))
    latent_lr = context.predictor_learning_rate
    max_train = _override(cfg.latent_tda_max_train, context.dataset_config.get("LATENT_TDA_MAX_TRAIN"))
    max_test = _override(cfg.latent_tda_max_test, context.dataset_config.get("LATENT_TDA_MAX_TEST"))
    repo = str(cfg.vjepa_repo)
    vjepa_namespace = f"{cfg.dataset}_vjepa_{repo.replace('/', '__')}"
    recompute_features = cfg.recompute_latent_tda_features or context.dataset_config.get(
        "RECOMPUTE_LATENT_TDA_FEATURES",
        False,
    )
    selected_device = _select_device(cfg.device)
    vjepa_device = selected_device.type if cfg.device == "auto" else cfg.device

    print(
        f"vjepa_latent_tda config: dataset={cfg.dataset}, namespace={vjepa_namespace}, repo={repo}, "
        f"seeds={seeds}, modes={modes}, vjepa_num_frames={cfg.vjepa_num_frames}, "
        f"vjepa_batch_size={cfg.vjepa_batch_size}, window={window}, bins={bins}, "
        f"predictor_epochs={latent_epochs}, lr={latent_lr}, horizon={context.horizon}"
    )

    rows = []
    for seed in tqdm_progress_bar(seeds, desc="vjepa_latent_tda seeds", total=len(seeds), leave=True):
        print(f"\n================ vjepa_latent_tda seed={seed} ================")
        _set_all_seeds(seed)
        x_train_subset = ml_tda_latent.take_batch_subset(context.x_train, max_train, seed=seed)
        x_test_subset = ml_tda_latent.take_batch_subset(context.x_test, max_test, seed=seed + 1)
        train_payload = _load_or_compute_vjepa_payload(
            namespace=vjepa_namespace,
            repo=repo,
            seed=seed,
            split_name="train",
            video_tensor=x_train_subset,
            device=vjepa_device,
            batch_size=cfg.vjepa_batch_size,
            num_frames=cfg.vjepa_num_frames,
            window=window,
            bins=bins,
            recompute=recompute_features,
        )
        test_payload = _load_or_compute_vjepa_payload(
            namespace=vjepa_namespace,
            repo=repo,
            seed=seed,
            split_name="test",
            video_tensor=x_test_subset,
            device=vjepa_device,
            batch_size=cfg.vjepa_batch_size,
            num_frames=cfg.vjepa_num_frames,
            window=window,
            bins=bins,
            recompute=recompute_features,
        )
        vjepa_latent_dim = int(train_payload["z"].shape[-1])
        ml_tda.configure_runtime(
            DATASET=vjepa_namespace,
            LATENT_DIM=vjepa_latent_dim,
            REAL_TDA_SCALE=context.real_tda_scale,
            REAL_TDA_BINS=context.real_tda_bins,
            HORIZON=context.horizon,
            DEVICE=selected_device,
            PREDICTOR_TYPE=cfg.predictor_type,
        )
        ml_tda_latent.configure_runtime(
            DATASET=vjepa_namespace,
            LATENT_DIM=vjepa_latent_dim,
            HIDDEN_DIM=context.hidden_dim,
            RETRAIN_ENCODER=False,
            HORIZON=context.horizon,
            LATENT_TDA_WINDOW=window,
            LATENT_TDA_BINS=bins,
            LATENT_TDA_PREDICTOR_EPOCHS=latent_epochs,
            LATENT_TDA_LR=latent_lr,
            RETRAIN_LATENT_TDA_PREDICTOR=cfg.retrain_predictor,
            RECOMPUTE_LATENT_TDA_FEATURES=recompute_features,
        )
        diagnostics = ml_tda_latent.latent_geometry_diagnostics(x_test_subset, test_payload["z"])

        for mode in modes:
            print(f"\n--- vjepa_latent_tda mode={mode} seed={seed} ---")
            train_features, test_features = ml_tda_latent.features_for_latent_tda_mode_pair(
                train_payload,
                test_payload,
                mode,
                train_control_seed=seed,
                test_control_seed=seed + 10_000,
            )
            model = ml_tda_latent.train_or_load_latent_tda_predictor(seed, mode, train_features, train_payload["z"])
            test_mse, _, latent_r2 = ml_tda_latent.eval_latent_tda_predictor(model, test_features, test_payload["z"])
            row = {
                "dataset": cfg.dataset,
                "encoder": "vjepa",
                "seed": seed,
                "mode": mode,
                "horizon": context.horizon,
                "test_mse": float(test_mse),
                "latent_r2": float(latent_r2),
                "vjepa_repo": repo,
                "vjepa_num_frames": int(cfg.vjepa_num_frames),
                "latent_dim": vjepa_latent_dim,
                **ml_tda_latent.topology_gate_statistics(model),
                **diagnostics,
            }
            rows.append(row)
            print("vjepa_latent_tda summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=ml_tda_latent.latent_result_metric_columns(results_df),
        sort_metric="test_mse",
    )
    print("\nvjepa_latent_tda mean +/- std:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def _load_geo_autoencoder_for_seed(
    cfg: RunConfig,
    context: VideoContext,
    seed: int,
) -> tuple[SpatialEncoder, SpatialDecoder, Path, Path, str]:
    geo_epochs = int(_override(cfg.geo_ae_epochs, context.dataset_config.get("GEO_AE_EPOCHS", 3)))
    model_namespace = _geo_model_namespace(cfg.dataset, cfg.geo_ae_lambda)
    encoder, decoder, encoder_path, decoder_path = ml_tda_geoae.load_or_train_geo_autoencoder(
        context.x_train,
        dataset_name=cfg.dataset,
        model_namespace=model_namespace,
        seed=seed,
        latent_dim=context.latent_dim,
        geo_lambda=cfg.geo_ae_lambda,
        ae_epochs=geo_epochs,
        frame_batch_size=context.ae_frame_batch_size or 256,
        max_frames_per_epoch=context.ae_max_frames_per_epoch,
        pair_batch_size=cfg.geo_ae_pair_batch_size,
        retrain=cfg.retrain_encoder,
        decoder_type=cfg.decoder_type,
    )
    return encoder, decoder, encoder_path, decoder_path, model_namespace


def _load_topo_autoencoder_for_seed(
    cfg: RunConfig,
    context: VideoContext,
    seed: int,
) -> tuple[SpatialEncoder, SpatialDecoder, Path, Path, str]:
    topo_epochs = int(_override(cfg.topo_ae_epochs, context.dataset_config.get("TOPO_AE_EPOCHS", 3)))
    model_namespace = _topo_model_namespace(cfg.dataset, cfg.topo_ae_lambda, cfg.topo_ae_distance)
    encoder, decoder, encoder_path, decoder_path = ml_tda_topoae.load_or_train_topo_autoencoder(
        context.x_train,
        dataset_name=cfg.dataset,
        model_namespace=model_namespace,
        seed=seed,
        latent_dim=context.latent_dim,
        topo_lambda=cfg.topo_ae_lambda,
        topo_distance=cfg.topo_ae_distance,
        ae_epochs=topo_epochs,
        frame_batch_size=context.ae_frame_batch_size or 256,
        max_frames_per_epoch=context.ae_max_frames_per_epoch,
        pair_batch_size=cfg.topo_ae_pair_batch_size,
        retrain=cfg.retrain_encoder,
        decoder_type=cfg.decoder_type,
    )
    return encoder, decoder, encoder_path, decoder_path, model_namespace


def _run_decode_z(
    cfg: RunConfig,
    context: VideoContext,
    *,
    ae_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(3))))
    fg_weight = float(
        _override(cfg.pixel_tda_fg_weight, context.dataset_config.get("PIXEL_TDA_FG_WEIGHT", 10.0))
    )
    fg_threshold = float(
        _override(cfg.pixel_tda_fg_threshold, context.dataset_config.get("PIXEL_TDA_FG_THRESHOLD", 0.05))
    )
    if ae_kind == "baseline":
        scenario_name = "decode_z"
        model_namespace = cfg.dataset
        ae_label = "baseline_ae"
        ae_lambda = np.nan
        ae_distance = ""
    elif ae_kind == "geo":
        scenario_name = "geo_decode_z"
        model_namespace = _geo_model_namespace(cfg.dataset, cfg.geo_ae_lambda)
        ae_label = "geo_ae"
        ae_lambda = cfg.geo_ae_lambda
        ae_distance = ""
    elif ae_kind == "topo":
        scenario_name = "topo_decode_z"
        model_namespace = _topo_model_namespace(cfg.dataset, cfg.topo_ae_lambda, cfg.topo_ae_distance)
        ae_label = "topo_ae"
        ae_lambda = cfg.topo_ae_lambda
        ae_distance = cfg.topo_ae_distance
    else:
        raise ValueError(f"Unknown AE kind: {ae_kind}")

    ml_tda.configure_runtime(
        DATASET=model_namespace,
        LATENT_DIM=context.latent_dim,
        HORIZON=context.horizon,
        DEVICE=_select_device(cfg.device),
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
    )
    ml_tda_pixel.configure_runtime(
        DATASET=model_namespace,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        HORIZON=context.horizon,
        PIXEL_TDA_FG_WEIGHT=fg_weight,
        PIXEL_TDA_FG_THRESHOLD=fg_threshold,
    )
    print(
        f"{scenario_name} config: dataset={cfg.dataset}, namespace={model_namespace}, "
        f"seeds={seeds}, horizon={context.horizon}, "
        f"predictor_epochs={context.predictor_epochs}, decoder_type={cfg.decoder_type}, "
        f"ae_lambda={ae_lambda}, ae_distance={ae_distance or 'n/a'}"
    )

    rows = []
    for seed in tqdm_progress_bar(seeds, desc=f"{scenario_name} seeds", total=len(seeds), leave=True):
        print(f"\n================ {scenario_name} seed={seed} ================")
        _set_all_seeds(seed)
        if ae_kind == "geo":
            encoder, decoder, encoder_path, decoder_path, model_namespace = _load_geo_autoencoder_for_seed(
                cfg,
                context,
                seed,
            )
        elif ae_kind == "topo":
            encoder, decoder, encoder_path, decoder_path, model_namespace = _load_topo_autoencoder_for_seed(
                cfg,
                context,
                seed,
            )
        else:
            encoder, decoder, encoder_path, decoder_path = _load_or_train_baseline_autoencoder(cfg, context, seed)
            model_namespace = cfg.dataset

        train_z, _ = ml_tda.build_real_tda_features(
            context.x_train,
            encoder,
            use_tda=False,
            split_name=f"{scenario_name} train z",
        )
        test_z, _ = ml_tda.build_real_tda_features(
            context.x_test,
            encoder,
            use_tda=False,
            split_name=f"{scenario_name} test z",
        )
        model, model_path = _train_or_load_decode_z_predictor(
            cfg,
            context,
            model_namespace,
            seed,
            train_z,
        )
        metrics = _evaluate_decode_z_to_future_x(model, decoder, test_z, context.x_test, context)
        figure_path = _save_decode_comparison_figure(
            metrics,
            dataset=cfg.dataset,
            encoder=ae_label,
            seed=seed,
        )
        row = {
            "dataset": cfg.dataset,
            "encoder": ae_label,
            "ae_lambda": ae_lambda,
            "ae_distance": ae_distance,
            "seed": seed,
            "mode": "z_decode",
            "horizon": context.horizon,
            "pixel_mse": float(metrics["pixel_mse"]),
            "pixel_r2": float(metrics["pixel_r2"]),
            "weighted_mse": float(metrics["weighted_mse"]),
            "foreground_mse": float(metrics["foreground_mse"]),
            "background_mse": float(metrics["background_mse"]),
            "oracle_pixel_mse": float(metrics["oracle_pixel_mse"]),
            "forecast_to_oracle_mse": float(metrics["forecast_to_oracle_mse"]),
            "fg_weight": fg_weight,
            "fg_threshold": fg_threshold,
            "encoder_path": str(encoder_path),
            "decoder_path": str(decoder_path),
            "model_path": str(model_path),
            "figure_path": str(figure_path),
        }
        rows.append(row)
        print(f"{scenario_name} summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=[
            "pixel_mse",
            "pixel_r2",
            "weighted_mse",
            "foreground_mse",
            "background_mse",
            "oracle_pixel_mse",
            "forecast_to_oracle_mse",
        ],
        sort_metric="weighted_mse",
    )
    print(f"\n{scenario_name} per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\n{scenario_name} mean +/- std:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def run_decode_z(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _run_decode_z(cfg, context, ae_kind="baseline")


def run_geo_decode_z(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _run_decode_z(cfg, context, ae_kind="geo")


def run_topo_decode_z(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _run_decode_z(cfg, context, ae_kind="topo")


def run_latent_classification(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_kind = context.dataset_config.get("kind")
    if dataset_kind not in {"synthetic_motion_classification", "aeon_raw_classification"}:
        raise ValueError(
            "latent_classification requires a synthetic_motion_classification "
            "or aeon_raw_classification dataset"
        )
    modes = _override(
        cfg.modes,
        context.dataset_config.get("CLASSIFICATION_MODES", ["z", "h1", "z_h1", "z_h1_shuffle"]),
    )
    unknown_modes = sorted(set(modes) - ml_tda_classification.VALID_MODES)
    if unknown_modes:
        raise ValueError(f"Unknown classification modes: {unknown_modes}")
    tasks = context.dataset_config.get("CLASSIFICATION_TASKS", ["motion", "object"])
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(5))))
    window = int(_override(cfg.latent_tda_window, context.dataset_config.get("LATENT_TDA_WINDOW", 20)))
    bins = int(_override(cfg.latent_tda_bins, context.dataset_config.get("LATENT_TDA_BINS", 16)))
    device = _select_device(cfg.device)

    ml_tda.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        DEVICE=device,
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
    )
    ml_tda_latent.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        LATENT_TDA_WINDOW=window,
        LATENT_TDA_BINS=bins,
        RECOMPUTE_LATENT_TDA_FEATURES=cfg.recompute_latent_tda_features,
    )
    print(
        f"Latent classification: tasks={tasks}, modes={modes}, seeds={seeds}, "
        f"window={window}, epochs={context.predictor_epochs}"
    )
    train_labels = {
        task: ml_tda_classification.labels_from_sources(context.dataset_config, "train", task)
        for task in tasks
    }
    test_labels = {
        task: ml_tda_classification.labels_from_sources(context.dataset_config, "test", task)
        for task in tasks
    }

    rows = []
    raw_payloads = None
    if dataset_kind == "aeon_raw_classification":
        train_z = context.x_train[:, :, 0, 0, :].float()
        test_z = context.x_test[:, :, 0, 0, :].float()
        train_h0, train_h1, train_diagrams = ml_tda_latent.latent_window_betti_features(
            train_z, window=window, n_bins=bins, return_diagrams=True
        )
        test_h0, test_h1, test_diagrams = ml_tda_latent.latent_window_betti_features(
            test_z, window=window, n_bins=bins, return_diagrams=True
        )
        raw_payloads = (
            {"z": train_z, "h0": train_h0, "h1": train_h1, "diagrams": train_diagrams},
            {"z": test_z, "h0": test_h0, "h1": test_h1, "diagrams": test_diagrams},
        )
    for seed in tqdm_progress_bar(seeds, desc="classification seeds", total=len(seeds), leave=True):
        _set_all_seeds(seed)
        if raw_payloads is None:
            encoder, _, encoder_path, _ = _load_or_train_baseline_autoencoder(cfg, context, seed)
            train_payload = ml_tda_latent.load_or_compute_latent_tda_features(
                seed, "classification_train", context.x_train, encoder, require_persistence=True
            )
            test_payload = ml_tda_latent.load_or_compute_latent_tda_features(
                seed, "classification_test", context.x_test, encoder, require_persistence=True
            )
            encoder_path_text = str(encoder_path)
        else:
            train_payload, test_payload = raw_payloads
            encoder_path_text = "raw_multivariate_trajectory"
        for mode in modes:
            train_x = ml_tda_classification.clip_features(train_payload, mode, window, seed=seed)
            test_x = ml_tda_classification.clip_features(test_payload, mode, window, seed=seed + 10_000)
            for task in tasks:
                metrics = ml_tda_classification.train_evaluate_classifier(
                    train_x,
                    train_labels[task],
                    test_x,
                    test_labels[task],
                    seed=seed,
                    hidden_dim=context.hidden_dim,
                    epochs=context.predictor_epochs,
                    learning_rate=context.predictor_learning_rate,
                    device=device,
                )
                row = {
                    "dataset": cfg.dataset,
                    "encoder": "raw" if raw_payloads is not None else "ae",
                    "seed": seed,
                    "task": task,
                    "mode": mode,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "encoder_path": encoder_path_text,
                }
                rows.append(row)
                print("Classification summary:", row)
    results_df = pd.DataFrame(rows)
    results_df["accuracy_gain_vs_z"] = np.nan
    results_df["paired_test"] = ""
    results_df["paired_p_value"] = np.nan
    results_df["paired_vs_shuffle_test"] = ""
    results_df["paired_vs_shuffle_p_value"] = np.nan
    for task in tasks:
        baseline = (
            results_df[(results_df["task"] == task) & (results_df["mode"] == "z")]
            .set_index("seed")["accuracy"]
        )
        for mode in modes:
            selected = (results_df["task"] == task) & (results_df["mode"] == mode)
            mode_accuracy = results_df.loc[selected].set_index("seed")["accuracy"]
            common = baseline.index.intersection(mode_accuracy.index)
            if common.empty:
                continue
            gains = mode_accuracy.loc[common] - baseline.loc[common]
            gain_by_seed = gains.to_dict()
            results_df.loc[selected, "accuracy_gain_vs_z"] = results_df.loc[selected, "seed"].map(gain_by_seed)
            if mode == "z_h0":
                p_value = ml_tda_classification.paired_signflip_test(gains.to_numpy())
                results_df.loc[selected, "paired_test"] = "exact_signflip_greater"
                results_df.loc[selected, "paired_p_value"] = p_value
                print(
                    f"Paired z_h0 > z test ({task}): mean_gain={gains.mean():.4f}, "
                    f"p={p_value:.6f}, n={len(gains)}"
                )
                shuffled = (
                    results_df[
                        (results_df["task"] == task)
                        & (results_df["mode"] == "z_h0_shuffle")
                    ]
                    .set_index("seed")["accuracy"]
                )
                shuffle_common = mode_accuracy.index.intersection(shuffled.index)
                if not shuffle_common.empty:
                    shuffle_gains = mode_accuracy.loc[shuffle_common] - shuffled.loc[shuffle_common]
                    shuffle_p = ml_tda_classification.paired_signflip_test(shuffle_gains.to_numpy())
                    results_df.loc[selected, "paired_vs_shuffle_test"] = "exact_signflip_greater"
                    results_df.loc[selected, "paired_vs_shuffle_p_value"] = shuffle_p
                    print(
                        f"Paired z_h0 > z_h0_shuffle test ({task}): "
                        f"mean_gain={shuffle_gains.mean():.4f}, p={shuffle_p:.6f}, "
                        f"n={len(shuffle_gains)}"
                    )
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols=["task", "mode"],
        metric_cols=["accuracy", "balanced_accuracy", "macro_f1", "accuracy_gain_vs_z"],
        sort_metric="accuracy",
    )
    print("\nLatent classification mean +/- std:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def run_simvp(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_modes = _override(cfg.modes, context.dataset_config.get("SIMVP_MODES", ["frames", "z", "z_fuse_h1"]))
    modes = []
    for mode in raw_modes:
        mode_text = str(mode).strip()
        if ml_tda_simvp.is_unconditioned_mode(mode_text):
            modes.append("frames")
        else:
            modes.extend(ml_tda_latent.canonicalize_latent_tda_modes([mode_text]))
    modes = list(dict.fromkeys(modes))
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(3))))
    window = int(_override(cfg.latent_tda_window, context.dataset_config.get("LATENT_TDA_WINDOW", 20)))
    bins = int(_override(cfg.latent_tda_bins, context.dataset_config.get("LATENT_TDA_BINS", 16)))
    max_train = _override(cfg.latent_tda_max_train, context.dataset_config.get("LATENT_TDA_MAX_TRAIN"))
    max_test = _override(cfg.latent_tda_max_test, context.dataset_config.get("LATENT_TDA_MAX_TEST"))
    simvp_epochs = int(_override(cfg.predictor_epochs, context.dataset_config.get("SIMVP_EPOCHS", context.predictor_epochs)))
    simvp_lr = float(_override(cfg.predictor_learning_rate, context.dataset_config.get("SIMVP_LR", context.predictor_learning_rate)))
    simvp_batch_size = int(
        _override(cfg.pixel_tda_batch_size, context.dataset_config.get("SIMVP_BATCH_SIZE", 16))
    )
    simvp_hidden = int(_override(cfg.hidden_dim, context.dataset_config.get("SIMVP_HIDDEN_DIM", context.hidden_dim)))
    simvp_input_frames = int(
        _override(cfg.simvp_input_frames, context.dataset_config.get("SIMVP_INPUT_FRAMES", 5))
    )
    recompute_features = cfg.recompute_latent_tda_features or context.dataset_config.get(
        "RECOMPUTE_LATENT_TDA_FEATURES",
        False,
    )
    device = _select_device(cfg.device)

    ml_tda.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        REAL_TDA_SCALE=context.real_tda_scale,
        REAL_TDA_BINS=context.real_tda_bins,
        HORIZON=context.horizon,
        DEVICE=device,
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
    )
    ml_tda_latent.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        RETRAIN_ENCODER=cfg.retrain_encoder,
        HORIZON=context.horizon,
        LATENT_TDA_WINDOW=window,
        LATENT_TDA_BINS=bins,
        RECOMPUTE_LATENT_TDA_FEATURES=recompute_features,
    )
    ml_tda_simvp.configure_runtime(
        DATASET=f"{cfg.dataset}_simvp",
        HORIZON=context.horizon,
        SIMVP_INPUT_FRAMES=simvp_input_frames,
        SIMVP_HIDDEN_DIM=simvp_hidden,
        SIMVP_EPOCHS=simvp_epochs,
        SIMVP_LR=simvp_lr,
        SIMVP_BATCH_SIZE=simvp_batch_size,
        SIMVP_RETRAIN=cfg.retrain_predictor,
    )
    print(
        f"SimVP config: dataset={cfg.dataset}, seeds={seeds}, modes={modes}, "
        f"input_frames={simvp_input_frames}, horizon={context.horizon}, epochs={simvp_epochs}, "
        f"lr={simvp_lr}, batch={simvp_batch_size}, hidden={simvp_hidden}, "
        f"max_train={max_train}, max_test={max_test}"
    )

    rows = []
    for seed in tqdm_progress_bar(seeds, desc="simvp seeds", total=len(seeds), leave=True):
        print(f"\n================ SimVP seed={seed} ================")
        _set_all_seeds(seed)
        train_video = ml_tda_latent.take_batch_subset(context.x_train, max_train, seed=seed)
        test_video = ml_tda_latent.take_batch_subset(context.x_test, max_test, seed=seed + 1)
        encoder, _, encoder_path, _ = _load_or_train_baseline_autoencoder(cfg, context, seed)

        train_payload = None
        test_payload = None
        if any(not ml_tda_simvp.is_unconditioned_mode(mode) for mode in modes):
            train_payload = ml_tda_latent.load_or_compute_latent_tda_features(seed, "train", train_video, encoder)
            test_payload = ml_tda_latent.load_or_compute_latent_tda_features(seed, "test", test_video, encoder)

        for mode in modes:
            print(f"\n--- SimVP mode={mode} seed={seed} ---")
            if ml_tda_simvp.is_unconditioned_mode(mode):
                train_cond = None
                test_cond = None
            else:
                if train_payload is None or test_payload is None:
                    raise RuntimeError("Latent payload was not computed for conditioned SimVP mode.")
                train_cond, test_cond = ml_tda_latent.features_for_latent_tda_mode_pair(
                    train_payload,
                    test_payload,
                    mode,
                    train_control_seed=seed,
                    test_control_seed=seed + 10_000,
                )
            model, model_path = ml_tda_simvp.train_or_load_simvp(
                seed=seed,
                mode=mode,
                train_video=train_video,
                cond_features=train_cond,
            )
            metrics = ml_tda_simvp.evaluate_simvp(
                model=model,
                test_video=test_video,
                cond_features=test_cond,
                encoder=encoder,
            )
            row = {
                "dataset": cfg.dataset,
                "encoder": "simvp",
                "seed": seed,
                "mode": mode,
                "horizon": context.horizon,
                "input_frames": simvp_input_frames,
                "pixel_mse": float(metrics["pixel_mse"]),
                "pixel_r2": float(metrics["pixel_r2"]),
                "weighted_mse": float(metrics["weighted_mse"]),
                "foreground_mse": float(metrics["foreground_mse"]),
                "background_mse": float(metrics["background_mse"]),
                "latent_mse": float(metrics.get("latent_mse", float("nan"))),
                "latent_r2": float(metrics.get("latent_r2", float("nan"))),
                "encoder_path": str(encoder_path),
                "model_path": str(model_path),
            }
            rows.append(row)
            print("SimVP summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=[
            "pixel_mse",
            "pixel_r2",
            "weighted_mse",
            "foreground_mse",
            "background_mse",
            "latent_mse",
            "latent_r2",
        ],
        sort_metric="pixel_mse",
    )
    print("\nSimVP per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSimVP mean +/- std:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def run_geo_pixel_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get("PIXEL_TDA_MODES", ["none", "h0", "h1", "both"]),
    )
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(3))))
    pixel_epochs = int(_override(cfg.predictor_epochs, context.dataset_config.get("PIXEL_TDA_PREDICTOR_EPOCHS", context.predictor_epochs)))
    pixel_lr = float(_override(cfg.predictor_learning_rate, context.dataset_config.get("PIXEL_TDA_LR", context.predictor_learning_rate)))
    pixel_batch_size = int(
        _override(cfg.pixel_tda_batch_size, context.dataset_config.get("PIXEL_TDA_BATCH_SIZE", 32))
    )
    fg_weight = float(
        _override(cfg.pixel_tda_fg_weight, context.dataset_config.get("PIXEL_TDA_FG_WEIGHT", 10.0))
    )
    fg_threshold = float(
        _override(cfg.pixel_tda_fg_threshold, context.dataset_config.get("PIXEL_TDA_FG_THRESHOLD", 0.05))
    )
    model_namespace = _geo_model_namespace(cfg.dataset, cfg.geo_ae_lambda)

    ml_tda.configure_runtime(
        DATASET=model_namespace,
        LATENT_DIM=context.latent_dim,
        REAL_TDA_SCALE=context.real_tda_scale,
        REAL_TDA_BINS=context.real_tda_bins,
        HORIZON=context.horizon,
        DEVICE=_select_device(cfg.device),
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
    )
    ml_tda_pixel.configure_runtime(
        DATASET=model_namespace,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        HORIZON=context.horizon,
        REAL_TDA_BINS=context.real_tda_bins,
        REAL_TDA_SCALE=context.real_tda_scale,
        PIXEL_TDA_PREDICTOR_EPOCHS=pixel_epochs,
        PIXEL_TDA_LR=pixel_lr,
        PIXEL_TDA_BATCH_SIZE=pixel_batch_size,
        PIXEL_TDA_FG_WEIGHT=fg_weight,
        PIXEL_TDA_FG_THRESHOLD=fg_threshold,
        PIXEL_TDA_RETRAIN_ENCODER=cfg.retrain_encoder,
        PIXEL_TDA_RETRAIN_PREDICTOR=cfg.retrain_predictor,
    )
    print(
        f"Geo pixel-TDA config: dataset={cfg.dataset}, namespace={model_namespace}, seeds={seeds}, "
        f"modes={modes}, geo_ae_lambda={cfg.geo_ae_lambda}, predictor_epochs={pixel_epochs}, "
        f"lr={pixel_lr}, batch={pixel_batch_size}, fg_weight={fg_weight}, "
        f"fg_threshold={fg_threshold}, horizon={context.horizon}"
    )

    rows = []
    for seed in tqdm_progress_bar(seeds, desc="geo_pixel_tda seeds", total=len(seeds), leave=True):
        print(f"\n================ geo pixel-TDA seed={seed} ================")
        _set_all_seeds(seed)
        encoder, encoder_path, _ = _load_geo_encoder_for_seed(cfg, context, seed)
        train_z, train_h0, train_h1 = ml_tda_pixel.compute_z_h0_h1(
            context.x_train,
            encoder,
            split_name="geo pixel train",
        )
        test_z, test_h0, test_h1 = ml_tda_pixel.compute_z_h0_h1(
            context.x_test,
            encoder,
            split_name="geo pixel test",
        )

        for mode in modes:
            print(f"\n--- geo pixel mode={mode} seed={seed} ---")
            train_features = ml_tda_pixel.make_mode_features(train_z, train_h0, train_h1, mode, seed)
            test_features = ml_tda_pixel.make_mode_features(test_z, test_h0, test_h1, mode, seed + 10_000)
            model, model_path = ml_tda_pixel.train_or_load_predictor(
                seed,
                mode,
                train_features,
                context.x_train,
            )
            metrics = ml_tda_pixel.evaluate_predictor(model, test_features, context.x_test)
            row = {
                "dataset": cfg.dataset,
                "encoder": "geo_ae",
                "geo_ae_lambda": cfg.geo_ae_lambda,
                "seed": seed,
                "mode": mode,
                "horizon": context.horizon,
                "pixel_mse": float(metrics["pixel_mse"]),
                "pixel_r2": float(metrics["pixel_r2"]),
                "weighted_mse": float(metrics["weighted_mse"]),
                "foreground_mse": float(metrics["foreground_mse"]),
                "background_mse": float(metrics["background_mse"]),
                "fg_weight": fg_weight,
                "fg_threshold": fg_threshold,
                "encoder_path": str(encoder_path),
                "model_path": str(model_path),
            }
            rows.append(row)
            print("geo pixel-TDA summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=[
            "pixel_mse",
            "pixel_r2",
            "weighted_mse",
            "foreground_mse",
            "background_mse",
        ],
        sort_metric="weighted_mse",
    )
    print("\nGeo pixel-TDA per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nGeo pixel-TDA mean +/- std:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def run_topo_pixel_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get("PIXEL_TDA_MODES", ["none", "h0", "h1", "both"]),
    )
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(3))))
    pixel_epochs = int(_override(cfg.predictor_epochs, context.dataset_config.get("PIXEL_TDA_PREDICTOR_EPOCHS", context.predictor_epochs)))
    pixel_lr = float(_override(cfg.predictor_learning_rate, context.dataset_config.get("PIXEL_TDA_LR", context.predictor_learning_rate)))
    pixel_batch_size = int(
        _override(cfg.pixel_tda_batch_size, context.dataset_config.get("PIXEL_TDA_BATCH_SIZE", 32))
    )
    fg_weight = float(
        _override(cfg.pixel_tda_fg_weight, context.dataset_config.get("PIXEL_TDA_FG_WEIGHT", 10.0))
    )
    fg_threshold = float(
        _override(cfg.pixel_tda_fg_threshold, context.dataset_config.get("PIXEL_TDA_FG_THRESHOLD", 0.05))
    )
    model_namespace = _topo_model_namespace(cfg.dataset, cfg.topo_ae_lambda, cfg.topo_ae_distance)

    ml_tda.configure_runtime(
        DATASET=model_namespace,
        LATENT_DIM=context.latent_dim,
        REAL_TDA_SCALE=context.real_tda_scale,
        REAL_TDA_BINS=context.real_tda_bins,
        HORIZON=context.horizon,
        DEVICE=_select_device(cfg.device),
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
    )
    ml_tda_pixel.configure_runtime(
        DATASET=model_namespace,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        HORIZON=context.horizon,
        REAL_TDA_BINS=context.real_tda_bins,
        REAL_TDA_SCALE=context.real_tda_scale,
        PIXEL_TDA_PREDICTOR_EPOCHS=pixel_epochs,
        PIXEL_TDA_LR=pixel_lr,
        PIXEL_TDA_BATCH_SIZE=pixel_batch_size,
        PIXEL_TDA_FG_WEIGHT=fg_weight,
        PIXEL_TDA_FG_THRESHOLD=fg_threshold,
        PIXEL_TDA_RETRAIN_ENCODER=cfg.retrain_encoder,
        PIXEL_TDA_RETRAIN_PREDICTOR=cfg.retrain_predictor,
    )
    print(
        f"Topo pixel-TDA config: dataset={cfg.dataset}, namespace={model_namespace}, seeds={seeds}, "
        f"modes={modes}, topo_ae_lambda={cfg.topo_ae_lambda}, "
        f"topo_ae_distance={cfg.topo_ae_distance}, predictor_epochs={pixel_epochs}, "
        f"lr={pixel_lr}, batch={pixel_batch_size}, fg_weight={fg_weight}, "
        f"fg_threshold={fg_threshold}, horizon={context.horizon}"
    )

    rows = []
    for seed in tqdm_progress_bar(seeds, desc="topo_pixel_tda seeds", total=len(seeds), leave=True):
        print(f"\n================ topo pixel-TDA seed={seed} ================")
        _set_all_seeds(seed)
        encoder, encoder_path, _ = _load_topo_encoder_for_seed(cfg, context, seed)
        train_z, train_h0, train_h1 = ml_tda_pixel.compute_z_h0_h1(
            context.x_train,
            encoder,
            split_name="topo pixel train",
        )
        test_z, test_h0, test_h1 = ml_tda_pixel.compute_z_h0_h1(
            context.x_test,
            encoder,
            split_name="topo pixel test",
        )

        for mode in modes:
            print(f"\n--- topo pixel mode={mode} seed={seed} ---")
            train_features = ml_tda_pixel.make_mode_features(train_z, train_h0, train_h1, mode, seed)
            test_features = ml_tda_pixel.make_mode_features(test_z, test_h0, test_h1, mode, seed + 10_000)
            model, model_path = ml_tda_pixel.train_or_load_predictor(
                seed,
                mode,
                train_features,
                context.x_train,
            )
            metrics = ml_tda_pixel.evaluate_predictor(model, test_features, context.x_test)
            row = {
                "dataset": cfg.dataset,
                "encoder": "topo_ae",
                "topo_ae_lambda": cfg.topo_ae_lambda,
                "topo_ae_distance": cfg.topo_ae_distance,
                "seed": seed,
                "mode": mode,
                "horizon": context.horizon,
                "pixel_mse": float(metrics["pixel_mse"]),
                "pixel_r2": float(metrics["pixel_r2"]),
                "weighted_mse": float(metrics["weighted_mse"]),
                "foreground_mse": float(metrics["foreground_mse"]),
                "background_mse": float(metrics["background_mse"]),
                "fg_weight": fg_weight,
                "fg_threshold": fg_threshold,
                "encoder_path": str(encoder_path),
                "model_path": str(model_path),
            }
            rows.append(row)
            print("topo pixel-TDA summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=[
            "pixel_mse",
            "pixel_r2",
            "weighted_mse",
            "foreground_mse",
            "background_mse",
        ],
        sort_metric="weighted_mse",
    )
    print("\nTopo pixel-TDA per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTopo pixel-TDA mean +/- std:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def run_pixel_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get("PIXEL_TDA_MODES", ["none", "h0", "h1", "both"]),
    )
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(3))))
    pixel_epochs = int(_override(cfg.predictor_epochs, context.dataset_config.get("PIXEL_TDA_PREDICTOR_EPOCHS", context.predictor_epochs)))
    pixel_lr = float(_override(cfg.predictor_learning_rate, context.dataset_config.get("PIXEL_TDA_LR", context.predictor_learning_rate)))
    pixel_batch_size = int(
        _override(cfg.pixel_tda_batch_size, context.dataset_config.get("PIXEL_TDA_BATCH_SIZE", 32))
    )
    fg_weight = float(
        _override(cfg.pixel_tda_fg_weight, context.dataset_config.get("PIXEL_TDA_FG_WEIGHT", 10.0))
    )
    fg_threshold = float(
        _override(cfg.pixel_tda_fg_threshold, context.dataset_config.get("PIXEL_TDA_FG_THRESHOLD", 0.05))
    )

    ml_tda.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        REAL_TDA_SCALE=context.real_tda_scale,
        REAL_TDA_BINS=context.real_tda_bins,
        HORIZON=context.horizon,
        DEVICE=_select_device(cfg.device),
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
    )
    ml_tda_pixel.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        HORIZON=context.horizon,
        REAL_TDA_BINS=context.real_tda_bins,
        REAL_TDA_SCALE=context.real_tda_scale,
        PIXEL_TDA_PREDICTOR_EPOCHS=pixel_epochs,
        PIXEL_TDA_LR=pixel_lr,
        PIXEL_TDA_BATCH_SIZE=pixel_batch_size,
        PIXEL_TDA_FG_WEIGHT=fg_weight,
        PIXEL_TDA_FG_THRESHOLD=fg_threshold,
        PIXEL_TDA_RETRAIN_ENCODER=cfg.retrain_encoder,
        PIXEL_TDA_RETRAIN_PREDICTOR=cfg.retrain_predictor,
    )
    print(
        f"Pixel TDA config: dataset={cfg.dataset}, seeds={seeds}, modes={modes}, "
        f"predictor_epochs={pixel_epochs}, lr={pixel_lr}, batch={pixel_batch_size}, "
        f"fg_weight={fg_weight}, fg_threshold={fg_threshold}, "
        f"horizon={context.horizon}"
    )
    results_df, summary_df, _ = ml_tda_pixel.run_pixel_tda_experiment(
        X_train=context.x_train,
        X_test=context.x_test,
        seeds=seeds,
        modes=modes,
        display_fn=None,
    )
    return results_df, summary_df


def run_video3d_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get("VIDEO3D_TDA_MODES", ["none", "h1", "h2", "all"]),
    )
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(3))))
    pixel_epochs = int(_override(cfg.predictor_epochs, context.dataset_config.get("VIDEO3D_TDA_PREDICTOR_EPOCHS", context.predictor_epochs)))
    pixel_lr = float(_override(cfg.predictor_learning_rate, context.dataset_config.get("VIDEO3D_TDA_LR", context.predictor_learning_rate)))
    batch_size = int(
        _override(cfg.pixel_tda_batch_size, context.dataset_config.get("VIDEO3D_TDA_BATCH_SIZE", 32))
    )
    fg_weight = float(
        _override(cfg.pixel_tda_fg_weight, context.dataset_config.get("PIXEL_TDA_FG_WEIGHT", 10.0))
    )
    fg_threshold = float(
        _override(cfg.pixel_tda_fg_threshold, context.dataset_config.get("PIXEL_TDA_FG_THRESHOLD", 0.05))
    )
    bins = int(_override(cfg.video3d_tda_bins, context.dataset_config.get("VIDEO3D_TDA_BINS", context.real_tda_bins)))
    scale = float(_override(cfg.video3d_tda_scale, context.dataset_config.get("VIDEO3D_TDA_SCALE", context.real_tda_scale)))
    boundary_slices = int(
        _override(cfg.video3d_tda_boundary_slices, context.dataset_config.get("VIDEO3D_TDA_BOUNDARY_SLICES", 2))
    )

    ml_tda.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        HORIZON=context.horizon,
        DEVICE=_select_device(cfg.device),
        AE_EPOCHS=context.ae_epochs,
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
    )
    ml_tda_pixel.configure_runtime(
        HORIZON=context.horizon,
        PIXEL_TDA_BATCH_SIZE=batch_size,
        PIXEL_TDA_FG_WEIGHT=fg_weight,
        PIXEL_TDA_FG_THRESHOLD=fg_threshold,
    )
    persistence_3d.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        HORIZON=context.horizon,
        VIDEO3D_TDA_BINS=bins,
        VIDEO3D_TDA_SCALE=scale,
        VIDEO3D_TDA_BOUNDARY_SLICES=boundary_slices,
        VIDEO3D_TDA_PREDICTOR_EPOCHS=pixel_epochs,
        VIDEO3D_TDA_LR=pixel_lr,
        VIDEO3D_TDA_BATCH_SIZE=batch_size,
        VIDEO3D_TDA_RETRAIN_PREDICTOR=cfg.retrain_predictor,
    )
    print(
        f"Video3D-TDA config: dataset={cfg.dataset}, seeds={seeds}, modes={modes}, "
        f"bins={bins}, scale={scale:g}, boundary_slices={boundary_slices}, "
        f"predictor_epochs={pixel_epochs}, lr={pixel_lr}, batch={batch_size}, "
        f"fg_weight={fg_weight}, fg_threshold={fg_threshold}, horizon={context.horizon}"
    )

    rows = []
    for seed in tqdm_progress_bar(seeds, desc="video3d_tda seeds", total=len(seeds), leave=True):
        print(f"\n================ video3d-tda seed={seed} ================")
        _set_all_seeds(seed)
        encoder, _, encoder_path, _ = _load_or_train_baseline_autoencoder(cfg, context, seed)
        train_z, train_h0, train_h1, train_h2, train_payload = persistence_3d.compute_z_h0_h1_h2(
            context.x_train,
            encoder,
            split_name="train",
            seed=seed,
            recompute=cfg.recompute_video3d_tda_features,
        )
        test_z, test_h0, test_h1, test_h2, test_payload = persistence_3d.compute_z_h0_h1_h2(
            context.x_test,
            encoder,
            split_name="test",
            seed=seed,
            recompute=cfg.recompute_video3d_tda_features,
        )
        persistence_3d.print_payload_feature_stats(train_payload, label=f"seed={seed} train")
        persistence_3d.print_payload_feature_stats(test_payload, label=f"seed={seed} test")
        test_tda_stats = persistence_3d.payload_feature_stats(test_payload, prefix="test")

        for mode in modes:
            print(f"\n--- video3d mode={mode} seed={seed} ---")
            train_features = persistence_3d.make_mode_features(train_z, train_h0, train_h1, train_h2, mode, seed)
            test_features = persistence_3d.make_mode_features(test_z, test_h0, test_h1, test_h2, mode, seed + 10_000)
            model, model_path = persistence_3d.train_or_load_predictor(
                seed,
                mode,
                train_features,
                context.x_train,
            )
            metrics = persistence_3d.evaluate_predictor(model, test_features, context.x_test)
            row = {
                "dataset": cfg.dataset,
                "encoder": "ae",
                "seed": seed,
                "mode": mode,
                "horizon": context.horizon,
                "pixel_mse": float(metrics["pixel_mse"]),
                "pixel_r2": float(metrics["pixel_r2"]),
                "weighted_mse": float(metrics["weighted_mse"]),
                "foreground_mse": float(metrics["foreground_mse"]),
                "background_mse": float(metrics["background_mse"]),
                "video3d_backend": train_payload.get("backend", "unknown"),
                "video3d_bins": bins,
                "video3d_scale": scale,
                "boundary_slices": boundary_slices,
                "encoder_path": str(encoder_path),
                "model_path": str(model_path),
            }
            row.update(test_tda_stats)
            rows.append(row)
            print("video3d-TDA summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=[
            "pixel_mse",
            "pixel_r2",
            "weighted_mse",
            "foreground_mse",
            "background_mse",
        ],
        sort_metric="weighted_mse",
    )
    print("\nVideo3D-TDA per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nVideo3D-TDA mean +/- std:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def parse_args(argv: list[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="aux_tda",
        help="Experiment scenario(s), comma-separated, e.g. real_tda,geo_real_tda.",
    )
    parser.add_argument("--dataset", default="moving_mnist", help="Dataset(s), comma-separated.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seeds", type=_parse_int_list, default=None, help="Comma-separated seeds, e.g. 0,1,2")
    parser.add_argument("--modes", type=_parse_str_list, default=None, help="Comma-separated mode names")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--predictor-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--predictor-learning-rate",
        type=float,
        default=None,
        help="Predictor-only learning rate; leaves encoder training at --learning-rate.",
    )
    parser.add_argument("--real-tda-scale", type=float, default=None)
    parser.add_argument("--real-tda-bins", type=int, default=None)
    parser.add_argument("--ae-epochs", type=int, default=None, help="Baseline AE pretraining epochs.")
    parser.add_argument("--ae-frame-batch-size", type=int, default=None)
    parser.add_argument("--ae-max-frames-per-epoch", type=int, default=None)
    parser.add_argument("--force-rebuild-data-cache", action="store_true")
    parser.add_argument("--num-train-clips", type=int, default=None)
    parser.add_argument("--num-test-clips", type=int, default=None)
    parser.add_argument(
        "--retrain-encoder",
        "--include-retrain-encoder",
        dest="retrain_encoder",
        action="store_true",
        help="Refresh the shared encoder once per seed, then freeze it for all modes.",
    )
    parser.add_argument(
        "--reuse-predictor",
        action="store_true",
        help="Load existing predictors when present. By default, retrain predictors.",
    )
    parser.add_argument("--aux-tda-lambda", type=float, default=None)
    parser.add_argument("--latent-tda-window", type=int, default=None)
    parser.add_argument("--latent-tda-bins", type=int, default=None)
    parser.add_argument("--latent-tda-max-train", type=int, default=None)
    parser.add_argument("--latent-tda-max-test", type=int, default=None)
    parser.add_argument(
        "--stability-noise-levels",
        type=_parse_float_list,
        default=None,
        help="Comma-separated input-noise std levels for latent_stability, e.g. 0.01,0.03,0.05,0.10.",
    )
    parser.add_argument(
        "--fidelity-windows",
        type=int,
        default=100,
        help="Number of deterministic full test windows sampled per seed for representation_fidelity.",
    )
    parser.add_argument(
        "--recompute-latent-tda-features",
        action="store_true",
        help="Recompute latent-trajectory TDA caches even when matching cached files exist.",
    )
    parser.add_argument("--geo-ae-lambda", type=float, default=0.1)
    parser.add_argument("--geo-ae-epochs", type=int, default=None)
    parser.add_argument("--geo-ae-pair-batch-size", type=int, default=64)
    parser.add_argument(
        "--manifold-signatures",
        type=_parse_str_list,
        default=["e16", "h8_e8", "s8_e8", "h6_s6_e4"],
        help="Mixed-GeoAE signatures, e.g. e16,h8_e8,s8_e8,h6_s6_e4.",
    )
    parser.add_argument(
        "--manifold-signature-policy",
        choices=["manual", "triangle_shrinkage"],
        default="manual",
        help="Use explicit signatures or generate candidates from training-only triangle diagnostics.",
    )
    parser.add_argument(
        "--signature-shrinkages",
        type=_parse_float_list,
        default=[0.0, 0.5, 0.75, 1.0],
        help="Curvature strengths for triangle_shrinkage; released H/S dimensions become Euclidean.",
    )
    parser.add_argument("--route-lambda", type=float, default=0.1)
    parser.add_argument("--route-knn", type=int, default=5)
    parser.add_argument("--route-temperature", type=float, default=0.1)
    parser.add_argument(
        "--vr-distance",
        choices=sorted(ml_tda_manifold_ph.VR_DISTANCES),
        default="product_manifold",
        help="Distance used to build VR persistence in manifold_mixed_geo_latent_tda.",
    )
    parser.add_argument("--triangle-knn", type=_parse_int_list, default=[4, 8, 12])
    parser.add_argument("--triangle-samples", type=int, default=10_000)
    parser.add_argument("--triangle-max-points", type=int, default=512)
    parser.add_argument("--triangle-flat-threshold", type=float, default=1e-6)
    parser.add_argument("--pwgeo-ae-lambda", type=float, default=0.1)
    parser.add_argument("--pwgeo-ae-epochs", type=int, default=None)
    parser.add_argument("--pwgeo-ae-pair-batch-size", type=int, default=64)
    parser.add_argument(
        "--pwgeo-blend",
        type=_parse_float_list,
        default=[0.25],
        help="One value or a comma-separated critical-edge fraction sweep, e.g. 0,0.1,0.25,0.5.",
    )
    parser.add_argument("--pwgeo-knn", type=int, default=5, help="PW-GeoAE local neighbours per point.")
    parser.add_argument("--pwgeo-h0-weight", type=float, default=2.0, help="Extra weight on H0-critical MST edges.")
    parser.add_argument("--pwgeo-h1-weight", type=float, default=1.0, help="Weight on cycle-closing H1-candidate edges.")
    parser.add_argument("--topo-ae-lambda", type=float, default=0.1)
    parser.add_argument("--topo-ae-epochs", type=int, default=None)
    parser.add_argument("--topo-ae-pair-batch-size", type=int, default=64)
    parser.add_argument(
        "--topo-ae-distance",
        choices=sorted(ml_tda_topoae.TOPO_AE_DISTANCES),
        default="signature",
    )
    parser.add_argument(
        "--decoder-type",
        choices=sorted(ml_tda.DECODER_TYPES),
        default="mlp",
        help="Autoencoder decoder architecture: mlp keeps old flat decoder; conv uses a convolutional upsampling decoder.",
    )
    parser.add_argument(
        "--predictor-type",
        choices=sorted(ml_tda.PREDICTOR_TYPES),
        default="lstm",
        help="Temporal predictor architecture.",
    )
    parser.add_argument("--vae-beta", type=float, default=1e-3, help="VAE KL weight for vae_latent_tda.")
    parser.add_argument("--vae-epochs", type=int, default=None, help="VAE pretraining epochs.")
    parser.add_argument("--byol-epochs", type=int, default=None, help="BYOL-style pretraining epochs.")
    parser.add_argument("--byol-noise-std", type=float, default=0.05, help="BYOL augmentation noise std.")
    parser.add_argument(
        "--vjepa-repo",
        default=ml_tda_vjepa.DEFAULT_VJEPA_REPO,
        help="Hugging Face repo for vjepa_latent_tda.",
    )
    parser.add_argument("--vjepa-batch-size", type=int, default=2, help="V-JEPA encoding batch size.")
    parser.add_argument(
        "--vjepa-num-frames",
        type=int,
        default=16,
        help="Number of frames per V-JEPA context window ending at each time step.",
    )
    parser.add_argument(
        "--dinov2-repo",
        default=ml_tda_dinov2.DEFAULT_DINOV2_REPO,
        help="Hugging Face repo for dinov2_latent_tda.",
    )
    parser.add_argument("--dinov2-batch-size", type=int, default=64, help="DINOv2 frame encoding batch size.")
    parser.add_argument("--dinov2-image-size", type=int, default=224, help="DINOv2 frame resize dimension.")
    parser.add_argument("--dinov2-trainable-blocks", type=int, default=1, help="Number of final DINOv2 transformer blocks to fine-tune.")
    parser.add_argument("--dinov2-finetune-epochs", type=int, default=3, help="DINOv2 last-block fine-tuning epochs.")
    parser.add_argument("--dinov2-encoder-lr", type=float, default=3e-5, help="Learning rate for fine-tuned DINOv2 blocks.")
    parser.add_argument("--dinov2-clip-batch-size", type=int, default=8, help="Clip batch size for DINOv2 fine-tuning.")
    parser.add_argument(
        "--clip-repo",
        default=ml_tda_clip.DEFAULT_CLIP_REPO,
        help="Hugging Face repo for clip_latent_tda.",
    )
    parser.add_argument("--clip-batch-size", type=int, default=64, help="CLIP frame encoding batch size.")
    parser.add_argument("--clip-image-size", type=int, default=224, help="CLIP frame resize dimension.")
    parser.add_argument(
        "--simvp-input-frames",
        type=int,
        default=5,
        help="Number of past frames given to the SimVP pixel predictor.",
    )
    parser.add_argument(
        "--video3d-tda-bins",
        type=int,
        default=None,
        help="Betti-curve bins for video3d_tda H0/H1/H2 features.",
    )
    parser.add_argument(
        "--video3d-tda-scale",
        type=float,
        default=None,
        help="Count normalization scale for video3d_tda Betti curves.",
    )
    parser.add_argument(
        "--video3d-tda-boundary-slices",
        type=int,
        default=None,
        help="Rolling boundary-slice radius recorded by the video3d_tda streaming state.",
    )
    parser.add_argument(
        "--recompute-video3d-tda-features",
        action="store_true",
        help="Recompute video3d_tda H0/H1/H2 caches even when matching cached files exist.",
    )
    parser.add_argument("--pixel-tda-batch-size", type=int, default=None)
    parser.add_argument("--pixel-tda-fg-weight", type=float, default=None)
    parser.add_argument("--pixel-tda-fg-threshold", type=float, default=None)
    parser.add_argument(
        "--profile-run",
        action="store_true",
        help="Record wall time, peak memory, throughput, and topology-time fraction for each dataset/scenario run.",
    )
    parser.add_argument(
        "--profile-sizes",
        type=_parse_int_list,
        default=None,
        help="Comma-separated clip caps for scaling runs, e.g. 16,32,64,128. Applies to num/latent train-test clip caps.",
    )
    parser.add_argument(
        "--profile-warmup-runs",
        type=int,
        default=0,
        help="Number of profiling warmup runs to execute and discard for each dataset/scenario/mode/size.",
    )
    parser.add_argument(
        "--profile-repeats",
        type=int,
        default=1,
        help="Number of measured profiling repeats for each dataset/scenario/mode/size.",
    )
    parser.add_argument(
        "--hparam-file",
        type=Path,
        default=None,
        help="JSON file from scripts/tune_latent_z_hparams.py. Dataset-specific params are used unless overridden on CLI.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/ml_persistence"))
    parser.add_argument("--no-save", action="store_true")

    args = parser.parse_args(argv)
    return RunConfig(
        scenario=args.scenario,
        dataset=args.dataset,
        device=args.device,
        run_seeds=args.seeds,
        horizon=args.horizon,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        predictor_epochs=args.predictor_epochs,
        learning_rate=args.learning_rate,
        predictor_learning_rate=args.predictor_learning_rate,
        real_tda_scale=args.real_tda_scale,
        real_tda_bins=args.real_tda_bins,
        ae_epochs=args.ae_epochs,
        ae_frame_batch_size=args.ae_frame_batch_size,
        ae_max_frames_per_epoch=args.ae_max_frames_per_epoch,
        force_rebuild_data_cache=args.force_rebuild_data_cache,
        num_train_clips=args.num_train_clips,
        num_test_clips=args.num_test_clips,
        modes=args.modes,
        retrain_encoder=args.retrain_encoder,
        retrain_predictor=not args.reuse_predictor,
        aux_tda_lambda=args.aux_tda_lambda,
        latent_tda_window=args.latent_tda_window,
        latent_tda_bins=args.latent_tda_bins,
        latent_tda_max_train=args.latent_tda_max_train,
        latent_tda_max_test=args.latent_tda_max_test,
        recompute_latent_tda_features=args.recompute_latent_tda_features,
        stability_noise_levels=args.stability_noise_levels,
        fidelity_windows=args.fidelity_windows,
        geo_ae_lambda=args.geo_ae_lambda,
        geo_ae_epochs=args.geo_ae_epochs,
        geo_ae_pair_batch_size=args.geo_ae_pair_batch_size,
        manifold_signatures=args.manifold_signatures,
        manifold_signature_policy=args.manifold_signature_policy,
        signature_shrinkages=args.signature_shrinkages,
        route_lambda=args.route_lambda,
        route_knn=args.route_knn,
        route_temperature=args.route_temperature,
        vr_distance=args.vr_distance,
        triangle_knn=args.triangle_knn,
        triangle_samples=args.triangle_samples,
        triangle_max_points=args.triangle_max_points,
        triangle_flat_threshold=args.triangle_flat_threshold,
        pwgeo_ae_lambda=args.pwgeo_ae_lambda,
        pwgeo_ae_epochs=args.pwgeo_ae_epochs,
        pwgeo_ae_pair_batch_size=args.pwgeo_ae_pair_batch_size,
        pwgeo_blend=args.pwgeo_blend,
        pwgeo_knn=args.pwgeo_knn,
        pwgeo_h0_weight=args.pwgeo_h0_weight,
        pwgeo_h1_weight=args.pwgeo_h1_weight,
        topo_ae_lambda=args.topo_ae_lambda,
        topo_ae_epochs=args.topo_ae_epochs,
        topo_ae_pair_batch_size=args.topo_ae_pair_batch_size,
        topo_ae_distance=args.topo_ae_distance,
        decoder_type=args.decoder_type,
        predictor_type=args.predictor_type,
        vae_beta=args.vae_beta,
        vae_epochs=args.vae_epochs,
        byol_epochs=args.byol_epochs,
        byol_noise_std=args.byol_noise_std,
        vjepa_repo=args.vjepa_repo,
        vjepa_batch_size=args.vjepa_batch_size,
        vjepa_num_frames=args.vjepa_num_frames,
        dinov2_repo=args.dinov2_repo,
        dinov2_batch_size=args.dinov2_batch_size,
        dinov2_image_size=args.dinov2_image_size,
        dinov2_trainable_blocks=args.dinov2_trainable_blocks,
        dinov2_finetune_epochs=args.dinov2_finetune_epochs,
        dinov2_encoder_lr=args.dinov2_encoder_lr,
        dinov2_clip_batch_size=args.dinov2_clip_batch_size,
        clip_repo=args.clip_repo,
        clip_batch_size=args.clip_batch_size,
        clip_image_size=args.clip_image_size,
        simvp_input_frames=args.simvp_input_frames,
        video3d_tda_bins=args.video3d_tda_bins,
        video3d_tda_scale=args.video3d_tda_scale,
        video3d_tda_boundary_slices=args.video3d_tda_boundary_slices,
        recompute_video3d_tda_features=args.recompute_video3d_tda_features,
        pixel_tda_batch_size=args.pixel_tda_batch_size,
        pixel_tda_fg_weight=args.pixel_tda_fg_weight,
        pixel_tda_fg_threshold=args.pixel_tda_fg_threshold,
        profile_run=args.profile_run,
        profile_sizes=args.profile_sizes,
        profile_size=None,
        profile_warmup_runs=max(0, int(args.profile_warmup_runs)),
        profile_repeats=max(1, int(args.profile_repeats)),
        profile_repeat=None,
        profile_is_warmup=False,
        hparam_file=args.hparam_file,
        output_dir=args.output_dir,
        no_save=args.no_save,
    )


def _validate_requested_items(requested: list[str], valid: set[str], label: str) -> None:
    unknown = [item for item in requested if item not in valid]
    if unknown:
        valid_text = ", ".join(sorted(valid))
        unknown_text = ", ".join(unknown)
        raise ValueError(f"Unknown {label}: {unknown_text}. Valid {label}s: {valid_text}")


def _run_scenario(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if cfg.scenario == "real_tda":
        return run_real_tda(cfg, context)
    if cfg.scenario == "decode_z":
        return run_decode_z(cfg, context)
    if cfg.scenario == "geo_real_tda":
        return run_geo_real_tda(cfg, context)
    if cfg.scenario == "topo_real_tda":
        return run_topo_real_tda(cfg, context)
    if cfg.scenario == "latent_tda":
        return run_latent_tda(cfg, context)
    if cfg.scenario == "latent_stability":
        return run_latent_stability(cfg, context)
    if cfg.scenario == "geo_latent_stability":
        return run_latent_stability(cfg, context, encoder_kind="geo")
    if cfg.scenario == "topo_latent_stability":
        return run_latent_stability(cfg, context, encoder_kind="topo")
    if cfg.scenario == "representation_fidelity":
        return run_representation_fidelity(cfg, context)
    if cfg.scenario == "geo_representation_fidelity":
        return run_representation_fidelity(cfg, context, encoder_kind="geo")
    if cfg.scenario == "topo_representation_fidelity":
        return run_representation_fidelity(cfg, context, encoder_kind="topo")
    if cfg.scenario == "geo_latent_tda":
        return run_geo_latent_tda(cfg, context)
    if cfg.scenario == "pwgeo_latent_tda":
        return run_pwgeo_latent_tda(cfg, context)
    if cfg.scenario == "mixed_geo_latent_tda":
        return run_mixed_geo_latent_tda(cfg, context)
    if cfg.scenario == "routed_mixed_geo_latent_tda":
        return run_routed_mixed_geo_latent_tda(cfg, context)
    if cfg.scenario == "manifold_mixed_geo_latent_tda":
        return run_manifold_mixed_geo_latent_tda(cfg, context)
    if cfg.scenario == "triangle_curvature":
        return run_triangle_curvature(cfg, context)
    if cfg.scenario == "geo_latent_spectrum":
        return run_geo_latent_spectrum(cfg, context)
    if cfg.scenario == "topo_latent_tda":
        return run_topo_latent_tda(cfg, context)
    if cfg.scenario == "vae_latent_tda":
        return run_vae_latent_tda(cfg, context)
    if cfg.scenario == "byol_latent_tda":
        return run_byol_latent_tda(cfg, context)
    if cfg.scenario == "vjepa_latent_tda":
        return run_vjepa_latent_tda(cfg, context)
    if cfg.scenario == "dinov2_latent_tda":
        return run_dinov2_latent_tda(cfg, context)
    if cfg.scenario == "dinov2_finetune_latent_tda":
        return run_dinov2_latent_tda(cfg, context, fine_tune=True)
    if cfg.scenario == "clip_latent_tda":
        return run_clip_latent_tda(cfg, context)
    if cfg.scenario == "simvp":
        return run_simvp(cfg, context)
    if cfg.scenario == "latent_classification":
        return run_latent_classification(cfg, context)
    if cfg.scenario == "video3d_tda":
        return run_video3d_tda(cfg, context)
    if cfg.scenario == "aux_tda":
        return run_aux_tda(cfg, context)
    if cfg.scenario == "geo_pixel_tda":
        return run_geo_pixel_tda(cfg, context)
    if cfg.scenario == "geo_decode_z":
        return run_geo_decode_z(cfg, context)
    if cfg.scenario == "topo_pixel_tda":
        return run_topo_pixel_tda(cfg, context)
    if cfg.scenario == "topo_decode_z":
        return run_topo_decode_z(cfg, context)
    if cfg.scenario == "pixel_tda":
        return run_pixel_tda(cfg, context)
    raise ValueError(f"Unsupported scenario: {cfg.scenario}")


def _run_one_config(cfg: RunConfig) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    print(f"Control panel: scenario={cfg.scenario}, dataset={cfg.dataset}")
    context = load_video_context(cfg)
    profile_df = None
    if cfg.profile_run:
        with RunProfiler(cfg) as profiler:
            results_df, summary_df = _run_scenario(cfg, context)
        profile_df = pd.DataFrame([profiler.to_row(cfg, context)])
        print("\nProfiling metrics:")
        print(profile_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    else:
        results_df, summary_df = _run_scenario(cfg, context)

    _save_results(cfg, context, results_df, summary_df, profile_df)
    return results_df, summary_df, profile_df


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flat = df.copy()
    flat.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in flat.columns
    ]
    return flat


def _tag_run_frame(df: pd.DataFrame | None, cfg: RunConfig, *, summary: bool = False) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    tagged = df.reset_index() if summary else df.copy()
    tagged = _flatten_columns(tagged)
    if "dataset" in tagged.columns:
        tagged["dataset"] = cfg.dataset
    else:
        tagged.insert(0, "dataset", cfg.dataset)
    if "scenario" in tagged.columns:
        tagged["scenario"] = cfg.scenario
    else:
        tagged.insert(1, "scenario", cfg.scenario)
    return tagged


def _drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    protected_cols = {"dataset", "scenario", "seed", "mode", "horizon", "profile_size"}
    keep_cols = [
        col
        for col in df.columns
        if col in protected_cols or not df[col].isna().all()
    ]
    return df.loc[:, keep_cols]


def _format_mean_std(value: object, std: object) -> str:
    if pd.isna(value) and pd.isna(std):
        return "—"
    if pd.isna(std):
        return f"{float(value):.4f}"
    if pd.isna(value):
        return f"{float(std):.4f}"
    return f"{float(value):.4f} {float(std):.4f}"


def _combine_summary_mean_std_columns(df: pd.DataFrame) -> pd.DataFrame:
    combined = df.copy()
    output = pd.DataFrame(index=combined.index)
    used_cols: set[str] = set()

    for col in combined.columns:
        if col in used_cols:
            continue
        if col.endswith("_mean"):
            metric = col[: -len("_mean")]
            std_col = f"{metric}_std"
            if std_col in combined.columns:
                output[f"{metric} mean +/- std"] = [
                    _format_mean_std(value, std)
                    for value, std in zip(combined[col], combined[std_col])
                ]
                used_cols.update({col, std_col})
                continue
        if col.endswith("_std") and f"{col[: -len('_std')]}_mean" in combined.columns:
            continue
        output[col] = combined[col]
        used_cols.add(col)

    return output


def _print_grouped_aggregate_frame(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}:")
    group_cols = [col for col in ["dataset", "scenario"] if col in df.columns]
    if not group_cols:
        compact = _drop_empty_columns(df)
        print(compact.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        return

    grouped_blocks: dict[tuple[str, ...], list[pd.DataFrame]] = {}
    for _, group in df.groupby(group_cols, sort=False, dropna=False):
        compact = _drop_empty_columns(group)
        grouped_blocks.setdefault(tuple(compact.columns), []).append(compact)

    for _, blocks in grouped_blocks.items():
        combined = pd.concat(blocks, ignore_index=True, sort=False)
        print("\n" + "-" * 88)
        print(combined.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def _print_aggregate_tables(
    result_frames: list[pd.DataFrame],
    summary_frames: list[pd.DataFrame],
    profile_frames: list[pd.DataFrame] | None = None,
) -> None:
    if not result_frames and not summary_frames and not profile_frames:
        return
    print("\n\n================ FINAL COMBINED RESULTS ================")
    with pd.option_context("display.width", 240, "display.max_columns", None):
        if result_frames:
            combined_results = pd.concat(result_frames, ignore_index=True, sort=False)
            gate_cols = ["gate_h0_mean", "gate_h0_std", "gate_h1_mean", "gate_h1_std"]
            main_results = combined_results.drop(columns=gate_cols, errors="ignore")
            _print_grouped_aggregate_frame("All per-run results", main_results)
            if all(column in combined_results.columns for column in gate_cols):
                gate_results = combined_results.dropna(subset=["gate_h0_mean"])
                if not gate_results.empty:
                    group_cols = [
                        column
                        for column in ("dataset", "scenario")
                        if column in gate_results.columns
                    ]
                    gate_summary = (
                        gate_results.groupby(group_cols, as_index=False)
                        .agg(
                            n_runs=("gate_h0_mean", "size"),
                            gate_h0_mean=("gate_h0_mean", "mean"),
                            gate_h0_seed_std=("gate_h0_mean", "std"),
                            gate_h0_window_std=("gate_h0_std", "mean"),
                            gate_h1_mean=("gate_h1_mean", "mean"),
                            gate_h1_seed_std=("gate_h1_mean", "std"),
                            gate_h1_window_std=("gate_h1_std", "mean"),
                        )
                    )
                    _print_grouped_aggregate_frame(
                        "Group-gate diagnostics (gated mode only)",
                        gate_summary,
                    )
        if summary_frames:
            combined_summary = pd.concat(summary_frames, ignore_index=True, sort=False)
            combined_summary = _combine_summary_mean_std_columns(combined_summary)
            _print_grouped_aggregate_frame("All mean +/- std summaries", combined_summary)
        if profile_frames:
            combined_profile = pd.concat(profile_frames, ignore_index=True, sort=False)
            combined_profile = _add_profile_overhead(combined_profile)
            _print_grouped_aggregate_frame("All profiling results", combined_profile)


def _add_profile_overhead(df: pd.DataFrame) -> pd.DataFrame:
    if "mode" not in df.columns or "wall_time_sec" not in df.columns:
        return df
    prof = df.copy()
    prof["profile_baseline_mode"] = None
    prof["overhead_vs_baseline"] = np.nan
    prof["added_time_vs_baseline_sec"] = np.nan
    group_cols = [
        col
        for col in ["dataset", "scenario", "profile_size", "profile_repeat", "n_train_clips", "n_test_clips", "sequence_len"]
        if col in prof.columns
    ]
    for _, idx in prof.groupby(group_cols, sort=False, dropna=False).groups.items():
        group = prof.loc[idx]
        baseline_mode = "z" if (group["mode"] == "z").any() else None
        if baseline_mode is None and (group["mode"] == "frames").any():
            baseline_mode = "frames"
        if baseline_mode is None and (group["mode"] == "none").any():
            baseline_mode = "none"
        if baseline_mode is None:
            continue
        baseline_time = float(group.loc[group["mode"] == baseline_mode, "wall_time_sec"].iloc[0])
        if baseline_time <= 0:
            continue
        prof.loc[idx, "profile_baseline_mode"] = baseline_mode
        prof.loc[idx, "overhead_vs_baseline"] = prof.loc[idx, "wall_time_sec"] / baseline_time
        prof.loc[idx, "added_time_vs_baseline_sec"] = prof.loc[idx, "wall_time_sec"] - baseline_time
    return prof


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    scenarios = _parse_str_list(cfg.scenario) or [cfg.scenario]
    datasets = _parse_str_list(cfg.dataset) or [cfg.dataset]
    pwgeo_blends = cfg.pwgeo_blend if isinstance(cfg.pwgeo_blend, list) else [cfg.pwgeo_blend]
    scenario_variants = [
        (scenario, float(blend))
        for scenario in scenarios
        for blend in (pwgeo_blends if scenario == "pwgeo_latent_tda" else [pwgeo_blends[0]])
    ]
    profile_sizes = cfg.profile_sizes or [None]
    profile_modes = cfg.modes if cfg.profile_run and cfg.modes and len(cfg.modes) > 1 else [None]
    _validate_requested_items(scenarios, RUNNER_SCENARIOS, "scenario")
    _validate_requested_items(datasets, set(topo_config.DATASET_CONFIGS), "dataset")

    repeats_per_config = (cfg.profile_warmup_runs + cfg.profile_repeats) if cfg.profile_run else 1
    total = len(scenario_variants) * len(datasets) * len(profile_sizes) * len(profile_modes) * repeats_per_config
    run_idx = 0
    result_frames = []
    summary_frames = []
    profile_frames = []
    for dataset in datasets:
        for scenario, pwgeo_blend in scenario_variants:
            for profile_size in profile_sizes:
                for profile_mode in profile_modes:
                    repeat_labels = (
                        [("warmup", i + 1, True) for i in range(cfg.profile_warmup_runs)]
                        + [("repeat", i + 1, False) for i in range(cfg.profile_repeats)]
                        if cfg.profile_run
                        else [("run", None, False)]
                    )
                    for repeat_label, repeat_idx, is_warmup in repeat_labels:
                        run_idx += 1
                        size_text = "" if profile_size is None else f" profile_size={profile_size}"
                        mode_text = "" if profile_mode is None else f" mode={profile_mode}"
                        blend_text = f" pwgeo_blend={pwgeo_blend:g}" if scenario == "pwgeo_latent_tda" else ""
                        repeat_text = "" if repeat_idx is None else f" {repeat_label}={repeat_idx}"
                        print(
                            f"\n######## run {run_idx}/{total}: dataset={dataset} scenario={scenario}"
                            f"{blend_text}{size_text}{mode_text}{repeat_text} ########"
                        )
                        run_cfg = replace(
                            cfg,
                            dataset=dataset,
                            scenario=scenario,
                            pwgeo_blend=pwgeo_blend,
                            modes=[profile_mode] if profile_mode is not None else cfg.modes,
                            profile_size=profile_size,
                            profile_repeat=repeat_idx if cfg.profile_run else None,
                            profile_is_warmup=is_warmup,
                            no_save=cfg.no_save or is_warmup,
                            num_train_clips=profile_size if profile_size is not None else cfg.num_train_clips,
                            num_test_clips=profile_size if profile_size is not None else cfg.num_test_clips,
                            latent_tda_max_train=profile_size if profile_size is not None else cfg.latent_tda_max_train,
                            latent_tda_max_test=profile_size if profile_size is not None else cfg.latent_tda_max_test,
                        )
                        run_cfg = _apply_tuned_hparams(run_cfg)
                        results_df, summary_df, profile_df = _run_one_config(run_cfg)
                        if is_warmup:
                            print("Discarded profiling warmup run.")
                            print("=" * 88)
                            continue
                        tagged_results = _tag_run_frame(results_df, run_cfg)
                        tagged_summary = _tag_run_frame(summary_df, run_cfg, summary=True)
                        tagged_profile = _tag_run_frame(profile_df, run_cfg)
                        if tagged_results is not None and not tagged_results.empty:
                            result_frames.append(tagged_results)
                        if tagged_summary is not None and not tagged_summary.empty:
                            summary_frames.append(tagged_summary)
                        if tagged_profile is not None and not tagged_profile.empty:
                            profile_frames.append(tagged_profile)
                        print("=" * 88)

    has_fidelity = any("representation_fidelity" in scenario for scenario in scenarios)
    # Per-window fidelity rows remain in results.csv; printing hundreds of them
    # again in the combined terminal table obscures the model-level result.
    printed_results = [] if has_fidelity else result_frames
    _print_aggregate_tables(printed_results, summary_frames, profile_frames)

    forecast_frames = [
        frame for frame in result_frames
        if "scenario" in frame and frame["scenario"].isin(
            {"latent_tda", "geo_latent_tda", "pwgeo_latent_tda", "topo_latent_tda"}
        ).any()
    ]
    fidelity_frames = [
        frame for frame in summary_frames
        if "scenario" in frame and frame["scenario"].str.contains("representation_fidelity").any()
    ]
    if forecast_frames and fidelity_frames:
        from scripts.analyze_representation_fidelity import (
            _ensure_encoder,
            correlation_table,
            topology_gain,
            write_text_report,
        )

        forecast = pd.concat(forecast_frames, ignore_index=True, sort=False)
        fidelity = _ensure_encoder(pd.concat(fidelity_frames, ignore_index=True, sort=False))
        gain = topology_gain(forecast, "z_h1", "test_mse")
        keys = ["dataset", "encoder", "encoder_variant", "seed"]
        merged = fidelity.merge(gain, on=keys, how="inner", validate="one_to_one")
        if merged.empty:
            print("\nNo matched fidelity/forecast rows; skipped Spearman report.")
        else:
            correlations = correlation_table(merged)
            report_path = Path("latex_tables/fidelity_runs.txt")
            write_text_report(merged, correlations, report_path)
            print(f"\nSaved compact fidelity/Spearman report to {report_path}")


if __name__ == "__main__":
    main()
