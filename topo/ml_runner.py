"""Command-line entrypoint for video topology forecasting experiments.

This runner trains/evaluates the AE, geoAE, latent-TDA, frame-TDA, and
decode-to-frame scenarios without modifying the exploratory notebook.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import topo.config as topo_config
import topo.ml_tda as ml_tda
import topo.ml_tda_aux as ml_tda_aux
import topo.ml_tda_latent as ml_tda_latent
import topo.ml_tda_pixel as ml_tda_pixel
import topo.ml_tda_geoae as ml_tda_geoae
import topo.ml_tda_topoae as ml_tda_topoae
import topo.ml_tda_repr as ml_tda_repr
import topo.ml_tda_vjepa as ml_tda_vjepa
import topo.ml_tda_simvp as ml_tda_simvp
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
    "geo_latent_tda",
    "topo_latent_tda",
    "vae_latent_tda",
    "byol_latent_tda",
    "vjepa_latent_tda",
    "simvp",
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
    geo_ae_lambda: float
    geo_ae_epochs: int | None
    geo_ae_pair_batch_size: int
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
    simvp_input_frames: int
    pixel_tda_batch_size: int | None
    pixel_tda_fg_weight: float | None
    pixel_tda_fg_threshold: float | None
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


def _save_results(
    cfg: RunConfig,
    context: VideoContext | None,
    results_df: pd.DataFrame | None,
    summary_df: pd.DataFrame | None,
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
        results_df.to_csv(out_dir / "results.csv", index=False, float_format="%.4f")
    if summary_df is not None:
        summary_df.to_csv(out_dir / "summary.csv", float_format="%.4f")
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
    elif run_config.get("kind") == "aeon_classification":
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

    train_array, test_array = load_video_dataset(run_config)
    x_train = torch.from_numpy(train_array).unsqueeze(2)
    x_test = torch.from_numpy(test_array).unsqueeze(2)
    if x_train.dtype != torch.uint8:
        x_train = x_train.float()
        x_test = x_test.float()

    latent_dim = int(_override(cfg.latent_dim, run_config.get("LATENT_DIM", 128)))
    hidden_dim = int(_override(cfg.hidden_dim, run_config.get("HIDDEN_DIM", 128)))
    predictor_epochs = int(_override(cfg.predictor_epochs, run_config.get("PREDICTOR_EPOCHS", 10)))
    learning_rate = float(_override(cfg.learning_rate, run_config.get("learning_rate", 3e-4)))
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
                context.learning_rate,
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
            _, latent_r2 = _mse_r2(pred_z, target_z)
            row = {
                "dataset": cfg.dataset,
                "seed": seed,
                "mode": mode,
                "test_mse": float(test_mse),
                "latent_r2": float(latent_r2),
            }
            rows.append(row)
            print("real-TDA summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=["test_mse", "latent_r2"],
        sort_metric="test_mse",
    )
    print("\nReal-TDA per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nReal-TDA mean +/- std by mode:")
    print(summary_df)
    return results_df, summary_df


def _geo_model_namespace(dataset: str, geo_lambda: float) -> str:
    return f"{dataset}_geoae_lam{geo_lambda:g}"


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=context.learning_rate)
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
    target_x = ml_tda_pixel.target_frames(x_test[context.horizon:]).cpu()
    weighted_mse, pixel_mse, fg_mse, bg_mse = ml_tda_pixel.pixel_losses(pred_x, target_x)
    _, pixel_r2 = _mse_r2(pred_x, target_x)
    per_time_pixel_mse = ((pred_x - target_x) ** 2).mean(dim=(1, 2, 3, 4))
    return {
        "weighted_mse": float(weighted_mse),
        "pixel_mse": float(pixel_mse),
        "pixel_r2": float(pixel_r2),
        "foreground_mse": float(fg_mse),
        "background_mse": float(bg_mse),
        "per_time_pixel_mse": per_time_pixel_mse,
    }


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
        learning_rate=context.learning_rate,
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
            _, latent_r2 = _mse_r2(pred_z, target_z)
            row = {
                "dataset": cfg.dataset,
                "encoder": "geo_ae",
                "geo_ae_lambda": cfg.geo_ae_lambda,
                "seed": seed,
                "mode": mode,
                "test_mse": float(test_mse),
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
        metric_cols=["test_mse", "latent_r2"],
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
                    learning_rate=context.learning_rate,
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
            _, latent_r2 = _mse_r2(pred_z, target_z)
            row = {
                "dataset": cfg.dataset,
                "encoder": "topo_ae",
                "topo_ae_lambda": cfg.topo_ae_lambda,
                "topo_ae_distance": cfg.topo_ae_distance,
                "seed": seed,
                "mode": mode,
                "test_mse": float(test_mse),
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
        metric_cols=["test_mse", "latent_r2"],
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
    aux_lr = float(_override(cfg.learning_rate, context.dataset_config.get("AUX_TDA_LR", context.learning_rate)))

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
    latent_lr = float(_override(cfg.learning_rate, context.learning_rate))
    max_train = _override(cfg.latent_tda_max_train, context.dataset_config.get("LATENT_TDA_MAX_TRAIN"))
    max_test = _override(cfg.latent_tda_max_test, context.dataset_config.get("LATENT_TDA_MAX_TEST"))
    recompute_features = cfg.recompute_latent_tda_features or context.dataset_config.get(
        "RECOMPUTE_LATENT_TDA_FEATURES",
        True,
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
    )
    return results_df, summary_df


def run_geo_latent_tda(cfg: RunConfig, context: VideoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    latent_lr = float(_override(cfg.learning_rate, context.learning_rate))
    max_train = _override(cfg.latent_tda_max_train, context.dataset_config.get("LATENT_TDA_MAX_TRAIN"))
    max_test = _override(cfg.latent_tda_max_test, context.dataset_config.get("LATENT_TDA_MAX_TEST"))
    model_namespace = _geo_model_namespace(cfg.dataset, cfg.geo_ae_lambda)
    recompute_features = cfg.recompute_latent_tda_features or context.dataset_config.get(
        "RECOMPUTE_LATENT_TDA_FEATURES",
        True,
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
        f"Geo latent-TDA config: dataset={cfg.dataset}, namespace={model_namespace}, "
        f"seeds={seeds}, modes={modes}, geo_ae_lambda={cfg.geo_ae_lambda}, "
        f"window={window}, bins={bins}, predictor_epochs={latent_epochs}, lr={latent_lr}, "
        f"horizon={context.horizon}, "
        f"max_train={max_train}, max_test={max_test}, recompute_features={recompute_features}"
    )

    rows = []
    for seed in tqdm_progress_bar(seeds, desc="geo_latent_tda seeds", total=len(seeds), leave=True):
        print(f"\n================ geo latent TDA seed={seed} ================")
        _set_all_seeds(seed)
        encoder, encoder_path, _ = _load_geo_encoder_for_seed(cfg, context, seed)
        x_train_subset = ml_tda_latent.take_batch_subset(context.x_train, max_train, seed=seed)
        x_test_subset = ml_tda_latent.take_batch_subset(context.x_test, max_test, seed=seed + 1)
        train_payload = ml_tda_latent.load_or_compute_latent_tda_features(
            seed,
            "train",
            x_train_subset,
            encoder,
        )
        test_payload = ml_tda_latent.load_or_compute_latent_tda_features(
            seed,
            "test",
            x_test_subset,
            encoder,
        )
        diagnostics = ml_tda_latent.latent_geometry_diagnostics(x_test_subset, test_payload["z"])

        for mode in modes:
            print(f"\n--- geo latent mode={mode} seed={seed} ---")
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
                "encoder": "geo_ae",
                "geo_ae_lambda": cfg.geo_ae_lambda,
                "seed": seed,
                "mode": mode,
                "horizon": context.horizon,
                "test_mse": float(test_mse),
                "latent_r2": float(latent_r2),
                "encoder_path": str(encoder_path),
                **diagnostics,
            }
            rows.append(row)
            print("geo latent TDA summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=["test_mse", "latent_r2"],
        sort_metric="test_mse",
    )
    print("\nGeo latent-TDA per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nGeo latent-TDA mean +/- std:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


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
    latent_lr = float(_override(cfg.learning_rate, context.learning_rate))
    max_train = _override(cfg.latent_tda_max_train, context.dataset_config.get("LATENT_TDA_MAX_TRAIN"))
    max_test = _override(cfg.latent_tda_max_test, context.dataset_config.get("LATENT_TDA_MAX_TEST"))
    model_namespace = _topo_model_namespace(cfg.dataset, cfg.topo_ae_lambda, cfg.topo_ae_distance)
    recompute_features = cfg.recompute_latent_tda_features or context.dataset_config.get(
        "RECOMPUTE_LATENT_TDA_FEATURES",
        True,
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
        )
        test_payload = ml_tda_latent.load_or_compute_latent_tda_features(
            seed,
            "test",
            x_test_subset,
            encoder,
        )
        diagnostics = ml_tda_latent.latent_geometry_diagnostics(x_test_subset, test_payload["z"])

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
                **diagnostics,
            }
            rows.append(row)
            print("topo latent TDA summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=["test_mse", "latent_r2"],
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
    latent_lr = float(_override(cfg.learning_rate, context.learning_rate))
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
        True,
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
                **diagnostics,
            }
            rows.append(row)
            print(f"{scenario_name} summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=["test_mse", "latent_r2"],
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
    latent_lr = float(_override(cfg.learning_rate, context.learning_rate))
    max_train = _override(cfg.latent_tda_max_train, context.dataset_config.get("LATENT_TDA_MAX_TRAIN"))
    max_test = _override(cfg.latent_tda_max_test, context.dataset_config.get("LATENT_TDA_MAX_TEST"))
    repo = str(cfg.vjepa_repo)
    vjepa_namespace = f"{cfg.dataset}_vjepa_{repo.replace('/', '__')}"
    recompute_features = cfg.recompute_latent_tda_features or context.dataset_config.get(
        "RECOMPUTE_LATENT_TDA_FEATURES",
        True,
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
                **diagnostics,
            }
            rows.append(row)
            print("vjepa_latent_tda summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=["test_mse", "latent_r2"],
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
            "fg_weight": fg_weight,
            "fg_threshold": fg_threshold,
            "encoder_path": str(encoder_path),
            "decoder_path": str(decoder_path),
            "model_path": str(model_path),
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
    simvp_lr = float(_override(cfg.learning_rate, context.dataset_config.get("SIMVP_LR", context.learning_rate)))
    simvp_batch_size = int(
        _override(cfg.pixel_tda_batch_size, context.dataset_config.get("SIMVP_BATCH_SIZE", 16))
    )
    simvp_hidden = int(_override(cfg.hidden_dim, context.dataset_config.get("SIMVP_HIDDEN_DIM", context.hidden_dim)))
    simvp_input_frames = int(
        _override(cfg.simvp_input_frames, context.dataset_config.get("SIMVP_INPUT_FRAMES", 5))
    )
    recompute_features = cfg.recompute_latent_tda_features or context.dataset_config.get(
        "RECOMPUTE_LATENT_TDA_FEATURES",
        True,
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
    pixel_lr = float(_override(cfg.learning_rate, context.dataset_config.get("PIXEL_TDA_LR", context.learning_rate)))
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
    pixel_lr = float(_override(cfg.learning_rate, context.dataset_config.get("PIXEL_TDA_LR", context.learning_rate)))
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
    pixel_lr = float(_override(cfg.learning_rate, context.dataset_config.get("PIXEL_TDA_LR", context.learning_rate)))
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
        "--recompute-latent-tda-features",
        action="store_true",
        help="Recompute latent-trajectory TDA caches even when matching cached files exist.",
    )
    parser.add_argument("--geo-ae-lambda", type=float, default=0.1)
    parser.add_argument("--geo-ae-epochs", type=int, default=None)
    parser.add_argument("--geo-ae-pair-batch-size", type=int, default=64)
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
        "--simvp-input-frames",
        type=int,
        default=5,
        help="Number of past frames given to the SimVP pixel predictor.",
    )
    parser.add_argument("--pixel-tda-batch-size", type=int, default=None)
    parser.add_argument("--pixel-tda-fg-weight", type=float, default=None)
    parser.add_argument("--pixel-tda-fg-threshold", type=float, default=None)
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
        geo_ae_lambda=args.geo_ae_lambda,
        geo_ae_epochs=args.geo_ae_epochs,
        geo_ae_pair_batch_size=args.geo_ae_pair_batch_size,
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
        simvp_input_frames=args.simvp_input_frames,
        pixel_tda_batch_size=args.pixel_tda_batch_size,
        pixel_tda_fg_weight=args.pixel_tda_fg_weight,
        pixel_tda_fg_threshold=args.pixel_tda_fg_threshold,
        output_dir=args.output_dir,
        no_save=args.no_save,
    )


def _validate_requested_items(requested: list[str], valid: set[str], label: str) -> None:
    unknown = [item for item in requested if item not in valid]
    if unknown:
        valid_text = ", ".join(sorted(valid))
        unknown_text = ", ".join(unknown)
        raise ValueError(f"Unknown {label}: {unknown_text}. Valid {label}s: {valid_text}")


def _run_one_config(cfg: RunConfig) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    print(f"Control panel: scenario={cfg.scenario}, dataset={cfg.dataset}")
    context = load_video_context(cfg)
    if cfg.scenario == "real_tda":
        results_df, summary_df = run_real_tda(cfg, context)
    elif cfg.scenario == "decode_z":
        results_df, summary_df = run_decode_z(cfg, context)
    elif cfg.scenario == "geo_real_tda":
        results_df, summary_df = run_geo_real_tda(cfg, context)
    elif cfg.scenario == "topo_real_tda":
        results_df, summary_df = run_topo_real_tda(cfg, context)
    elif cfg.scenario == "latent_tda":
        results_df, summary_df = run_latent_tda(cfg, context)
    elif cfg.scenario == "geo_latent_tda":
        results_df, summary_df = run_geo_latent_tda(cfg, context)
    elif cfg.scenario == "topo_latent_tda":
        results_df, summary_df = run_topo_latent_tda(cfg, context)
    elif cfg.scenario == "vae_latent_tda":
        results_df, summary_df = run_vae_latent_tda(cfg, context)
    elif cfg.scenario == "byol_latent_tda":
        results_df, summary_df = run_byol_latent_tda(cfg, context)
    elif cfg.scenario == "vjepa_latent_tda":
        results_df, summary_df = run_vjepa_latent_tda(cfg, context)
    elif cfg.scenario == "simvp":
        results_df, summary_df = run_simvp(cfg, context)
    elif cfg.scenario == "aux_tda":
        results_df, summary_df = run_aux_tda(cfg, context)
    elif cfg.scenario == "geo_pixel_tda":
        results_df, summary_df = run_geo_pixel_tda(cfg, context)
    elif cfg.scenario == "geo_decode_z":
        results_df, summary_df = run_geo_decode_z(cfg, context)
    elif cfg.scenario == "topo_pixel_tda":
        results_df, summary_df = run_topo_pixel_tda(cfg, context)
    elif cfg.scenario == "topo_decode_z":
        results_df, summary_df = run_topo_decode_z(cfg, context)
    elif cfg.scenario == "pixel_tda":
        results_df, summary_df = run_pixel_tda(cfg, context)
    else:  # pragma: no cover - validated before dispatch.
        raise ValueError(f"Unsupported scenario: {cfg.scenario}")

    _save_results(cfg, context, results_df, summary_df)
    return results_df, summary_df


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
    tagged.insert(1, "scenario", cfg.scenario)
    return tagged


def _drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    protected_cols = {"dataset", "scenario", "seed", "mode", "horizon"}
    keep_cols = [
        col
        for col in df.columns
        if col in protected_cols or not df[col].isna().all()
    ]
    return df.loc[:, keep_cols]


def _format_mean_std(value: object, std: object) -> str:
    if pd.isna(value) and pd.isna(std):
        return ""
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
) -> None:
    if not result_frames and not summary_frames:
        return
    print("\n\n================ FINAL COMBINED RESULTS ================")
    with pd.option_context("display.width", 240, "display.max_columns", None):
        if result_frames:
            combined_results = pd.concat(result_frames, ignore_index=True, sort=False)
            _print_grouped_aggregate_frame("All per-run results", combined_results)
        if summary_frames:
            combined_summary = pd.concat(summary_frames, ignore_index=True, sort=False)
            combined_summary = _combine_summary_mean_std_columns(combined_summary)
            _print_grouped_aggregate_frame("All mean +/- std summaries", combined_summary)


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    scenarios = _parse_str_list(cfg.scenario) or [cfg.scenario]
    datasets = _parse_str_list(cfg.dataset) or [cfg.dataset]
    _validate_requested_items(scenarios, RUNNER_SCENARIOS, "scenario")
    _validate_requested_items(datasets, set(topo_config.DATASET_CONFIGS), "dataset")

    total = len(scenarios) * len(datasets)
    run_idx = 0
    result_frames = []
    summary_frames = []
    for dataset in datasets:
        for scenario in scenarios:
            run_idx += 1
            print(f"\n######## run {run_idx}/{total}: dataset={dataset} scenario={scenario} ########")
            run_cfg = replace(cfg, dataset=dataset, scenario=scenario)
            results_df, summary_df = _run_one_config(run_cfg)
            tagged_results = _tag_run_frame(results_df, run_cfg)
            tagged_summary = _tag_run_frame(summary_df, run_cfg, summary=True)
            if tagged_results is not None and not tagged_results.empty:
                result_frames.append(tagged_results)
            if tagged_summary is not None and not tagged_summary.empty:
                summary_frames.append(tagged_summary)
            print("=" * 88)

    _print_aggregate_tables(result_frames, summary_frames)


if __name__ == "__main__":
    main()
