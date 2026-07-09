"""Terminal runner for ML persistence experiments.

This is the command-line counterpart to ``topo/ml_persistence.ipynb``.  The
notebook remains useful for interactive exploration; this module keeps
repeatable experiment runs in a normal Python entrypoint.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
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
import topo.ml_tda_topoae as ml_tda_topoae
from topo.ml_tda import (
    SpatialDecoder,
    SpatialEncoder,
    TopologicalPredictor,
    load_or_train_model,
    load_video_dataset,
    parse_tda_control_mode,
    test_predictor,
)


SEQUENCE_SCENARIOS = {"sequence", "topo_sequence", "latent_tda", "topo_latent_tda", "aux_tda", "pixel_tda"}


@dataclass
class SequenceContext:
    dataset_config: dict[str, Any]
    x_train: torch.Tensor
    x_test: torch.Tensor
    latent_dim: int
    hidden_dim: int
    epochs: int
    learning_rate: float
    betti_scale: int | float
    n_steps: int
    predict_steps_ahead: int
    ae_frame_batch_size: int | None
    ae_max_frames_per_epoch: int | None


@dataclass
class RunConfig:
    scenario: str
    dataset: str
    device: str
    run_seeds: list[int] | None
    predict_steps_ahead: int | None
    latent_dim: int | None
    hidden_dim: int | None
    epochs: int | None
    learning_rate: float | None
    betti_scale: float | None
    n_steps: int | None
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
    topo_ae_lambda: float
    topo_ae_epochs: int | None
    topo_ae_pair_batch_size: int
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


def _save_results(
    cfg: RunConfig,
    context: SequenceContext | None,
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
        "epochs": context.epochs,
        "learning_rate": context.learning_rate,
        "betti_scale": context.betti_scale,
        "n_steps": context.n_steps,
        "predict_steps_ahead": context.predict_steps_ahead,
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


def load_sequence_context(cfg: RunConfig) -> SequenceContext:
    dataset_configs = topo_config.DATASET_CONFIGS
    if cfg.dataset not in dataset_configs:
        valid = ", ".join(sorted(dataset_configs))
        raise ValueError(f"Unknown dataset {cfg.dataset!r}. Valid datasets: {valid}")

    run_config = dict(dataset_configs[cfg.dataset])
    if run_config.get("kind") == "ctc_tif_clips":
        raise ValueError(
            f"Dataset {cfg.dataset!r} uses the celltracking runner, which has not been "
            "migrated to topo.ml_runner yet."
        )

    if cfg.dataset == "bouncing_balls":
        if cfg.num_train_clips is not None:
            run_config["num_train_clips"] = cfg.num_train_clips
        if cfg.num_test_clips is not None:
            run_config["num_test_clips"] = cfg.num_test_clips
        run_config["force_rebuild_cache"] = cfg.force_rebuild_data_cache

    train_array, test_array = load_video_dataset(run_config)
    x_train = torch.from_numpy(train_array).unsqueeze(2)
    x_test = torch.from_numpy(test_array).unsqueeze(2)
    if x_train.dtype != torch.uint8:
        x_train = x_train.float()
        x_test = x_test.float()

    latent_dim = int(_override(cfg.latent_dim, run_config.get("LATENT_DIM", 128)))
    hidden_dim = int(_override(cfg.hidden_dim, run_config.get("HIDDEN_DIM", 128)))
    epochs = int(_override(cfg.epochs, run_config.get("EPOCHS", 10)))
    learning_rate = float(_override(cfg.learning_rate, run_config.get("learning_rate", 3e-4)))
    betti_scale = _override(cfg.betti_scale, run_config.get("BETTI_SCALE", 15))
    n_steps = int(_override(cfg.n_steps, run_config.get("N_STEPS", 25)))
    predict_steps_ahead = int(
        _override(cfg.predict_steps_ahead, run_config.get("PREDICT_STEPS_AHEAD", 1))
    )
    ae_frame_batch_size = _override(
        cfg.ae_frame_batch_size,
        run_config.get("AE_FRAME_BATCH_SIZE", 256),
    )
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
        BETTI_SCALE=betti_scale,
        N_STEPS=n_steps,
        PREDICT_STEPS_AHEAD=predict_steps_ahead,
        EPOCHS=epochs,
        DEVICE=_select_device(cfg.device),
        AE_FRAME_BATCH_SIZE=ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=ae_max_frames_per_epoch,
    )
    print(f"Using sequence compute device: {ml_tda.get_runtime_device()}")

    return SequenceContext(
        dataset_config=run_config,
        x_train=x_train,
        x_test=x_test,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        betti_scale=betti_scale,
        n_steps=n_steps,
        predict_steps_ahead=predict_steps_ahead,
        ae_frame_batch_size=ae_frame_batch_size,
        ae_max_frames_per_epoch=ae_max_frames_per_epoch,
    )


def run_sequence(cfg: RunConfig, context: SequenceContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get("LEGACY_TDA_MODES", ["none", "h0", "h1", "both"]),
    )
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(5))))
    print(
        f"Sequence config: seeds={seeds}, modes={modes}, "
        f"predict_steps_ahead={context.predict_steps_ahead}"
    )

    rows = []
    for seed in seeds:
        print(f"\n================ sequence seed={seed} ================")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for mode in modes:
            ml_tda.configure_runtime(EXPERIMENT_SEED=seed, TDA_MODE=mode)
            base_mode, _ = parse_tda_control_mode(mode)
            use_tda = base_mode != "none"
            if base_mode == "none":
                tda_dim = 0
            elif base_mode in {"h0", "h1"}:
                tda_dim = context.n_steps
            else:
                tda_dim = 2 * context.n_steps
            input_dim = context.latent_dim + tda_dim

            print(f"\n-------- DATASET={cfg.dataset} seed={seed} TDA_MODE={mode} --------")
            encoder = SpatialEncoder(latent_dim=context.latent_dim)
            decoder = SpatialDecoder(
                latent_dim=context.latent_dim,
                output_size=context.x_train.shape[-2:],
            )
            model = TopologicalPredictor(input_dim=input_dim, hidden_dim=context.hidden_dim)
            load_or_train_model(
                encoder,
                decoder,
                model,
                context.x_train,
                cfg.retrain_encoder,
                cfg.retrain_predictor,
                use_tda,
                context.learning_rate,
            )
            test_mse, per_frame_mse = test_predictor(model, encoder, context.x_test, use_tda)
            row = {
                "dataset": cfg.dataset,
                "seed": seed,
                "mode": mode,
                "test_mse": float(test_mse),
                "warmup_excluded_mse": (
                    float(per_frame_mse[1:].mean()) if len(per_frame_mse) > 1 else float(test_mse)
                ),
            }
            rows.append(row)
            print("sequence summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=["test_mse", "warmup_excluded_mse"],
        sort_metric="test_mse",
    )
    print("\nSequence per-run results:")
    print(results_df.to_string(index=False))
    print("\nSequence mean +/- std by mode:")
    print(summary_df)
    return results_df, summary_df


def _topo_model_namespace(dataset: str, topo_lambda: float) -> str:
    return f"{dataset}_topoae_lam{topo_lambda:g}"


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _topo_predictor_path(
    model_namespace: str,
    seed: int,
    mode: str,
    context: SequenceContext,
    input_dim: int,
) -> Path:
    model_dir = Path("models") / model_namespace / "topo_sequence_predictors"
    model_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{seed}_mode{mode}_pred{context.predict_steps_ahead}_"
        f"T{context.x_train.shape[0]}_B{context.x_train.shape[1]}_"
        f"latent{context.latent_dim}_input{input_dim}"
    )
    return model_dir / f"{tag}.pt"


def _topo_feature_dim(mode: str, context: SequenceContext) -> tuple[bool, int]:
    base_mode, _ = parse_tda_control_mode(mode)
    if base_mode == "none":
        return False, context.latent_dim
    if base_mode in {"h0", "h1"}:
        return True, context.latent_dim + context.n_steps
    if base_mode == "both":
        return True, context.latent_dim + 2 * context.n_steps
    raise ValueError(f"Unknown TDA mode: {mode}")


def _train_or_load_topo_sequence_predictor(
    *,
    cfg: RunConfig,
    context: SequenceContext,
    encoder: SpatialEncoder,
    model_namespace: str,
    seed: int,
    mode: str,
) -> tuple[TopologicalPredictor, Path]:
    use_tda, input_dim = _topo_feature_dim(mode, context)
    model = TopologicalPredictor(input_dim=input_dim, hidden_dim=context.hidden_dim)
    model_path = _topo_predictor_path(model_namespace, seed, mode, context, input_dim)
    device = ml_tda.get_runtime_device()

    if model_path.exists() and not cfg.retrain_predictor:
        print(f"Loading topo-sequence predictor: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return model, model_path

    reason = "retraining" if model_path.exists() else "missing; training once"
    print(f"Topo-sequence predictor {reason}: {model_path}")
    model.to(device)
    ml_tda.train_predictor(
        model,
        encoder,
        context.x_train,
        epochs=context.epochs,
        use_tda=use_tda,
        learning_rate=context.learning_rate,
    )
    torch.save(model.state_dict(), model_path)
    print(f"Saved topo-sequence predictor: {model_path}")
    return model, model_path


def _load_topo_encoder_for_seed(
    cfg: RunConfig,
    context: SequenceContext,
    seed: int,
) -> tuple[SpatialEncoder, Path, str]:
    topo_epochs = int(_override(cfg.topo_ae_epochs, context.dataset_config.get("TOPO_AE_EPOCHS", 3)))
    model_namespace = _topo_model_namespace(cfg.dataset, cfg.topo_ae_lambda)
    encoder, encoder_path = ml_tda_topoae.load_or_train_topo_encoder(
        context.x_train,
        dataset_name=cfg.dataset,
        model_namespace=model_namespace,
        seed=seed,
        latent_dim=context.latent_dim,
        topo_lambda=cfg.topo_ae_lambda,
        epochs=topo_epochs,
        frame_batch_size=context.ae_frame_batch_size or 256,
        max_frames_per_epoch=context.ae_max_frames_per_epoch,
        pair_batch_size=cfg.topo_ae_pair_batch_size,
        retrain=cfg.retrain_encoder,
    )
    return encoder, encoder_path, model_namespace


def run_topo_sequence(cfg: RunConfig, context: SequenceContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get("LEGACY_TDA_MODES", ["none", "h0", "h1", "both"]),
    )
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(5))))
    print(
        f"Topo-sequence config: dataset={cfg.dataset}, seeds={seeds}, modes={modes}, "
        f"topo_ae_lambda={cfg.topo_ae_lambda}, predict_steps_ahead={context.predict_steps_ahead}"
    )

    rows = []
    for seed in seeds:
        print(f"\n================ topo-sequence seed={seed} ================")
        _set_all_seeds(seed)
        encoder, encoder_path, model_namespace = _load_topo_encoder_for_seed(cfg, context, seed)

        for mode in modes:
            ml_tda.configure_runtime(EXPERIMENT_SEED=seed, TDA_MODE=mode)
            use_tda, _ = _topo_feature_dim(mode, context)
            print(f"\n-------- DATASET={cfg.dataset} seed={seed} TOPO_TDA_MODE={mode} --------")
            model, model_path = _train_or_load_topo_sequence_predictor(
                cfg=cfg,
                context=context,
                encoder=encoder,
                model_namespace=model_namespace,
                seed=seed,
                mode=mode,
            )
            test_mse, per_frame_mse = test_predictor(model, encoder, context.x_test, use_tda)
            row = {
                "dataset": cfg.dataset,
                "encoder": "topo_ae",
                "topo_ae_lambda": cfg.topo_ae_lambda,
                "seed": seed,
                "mode": mode,
                "test_mse": float(test_mse),
                "warmup_excluded_mse": (
                    float(per_frame_mse[1:].mean()) if len(per_frame_mse) > 1 else float(test_mse)
                ),
                "encoder_path": str(encoder_path),
                "model_path": str(model_path),
            }
            rows.append(row)
            print("topo-sequence summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=["test_mse", "warmup_excluded_mse"],
        sort_metric="test_mse",
    )
    print("\nTopo-sequence per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTopo-sequence mean +/- std by mode:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def run_aux_tda(cfg: RunConfig, context: SequenceContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get("AUX_TDA_MODES", ["none", "aux_h0", "aux_h1", "aux_both"]),
    )
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(3))))
    aux_lambda = float(_override(cfg.aux_tda_lambda, context.dataset_config.get("AUX_TDA_LAMBDA", 0.1)))
    aux_epochs = int(_override(cfg.epochs, context.dataset_config.get("AUX_TDA_EPOCHS", context.epochs)))
    aux_lr = float(_override(cfg.learning_rate, context.dataset_config.get("AUX_TDA_LR", context.learning_rate)))

    ml_tda.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        BETTI_SCALE=context.betti_scale,
        N_STEPS=context.n_steps,
        PREDICT_STEPS_AHEAD=context.predict_steps_ahead,
        DEVICE=_select_device(cfg.device),
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
    )
    ml_tda_aux.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        PREDICT_STEPS_AHEAD=context.predict_steps_ahead,
        N_STEPS=context.n_steps,
        BETTI_SCALE=context.betti_scale,
        AUX_TDA_LAMBDA=aux_lambda,
        AUX_TDA_EPOCHS=aux_epochs,
        AUX_TDA_LR=aux_lr,
        AUX_TDA_RETRAIN_ENCODER=cfg.retrain_encoder,
        AUX_TDA_RETRAIN_PREDICTOR=cfg.retrain_predictor,
    )
    print(
        f"Aux TDA config: dataset={cfg.dataset}, seeds={seeds}, modes={modes}, "
        f"lambda={aux_lambda}, epochs={aux_epochs}, lr={aux_lr}, "
        f"predict_steps_ahead={context.predict_steps_ahead}"
    )
    results_df, summary_df, _ = ml_tda_aux.run_aux_tda_experiment(
        X_train=context.x_train,
        X_test=context.x_test,
        seeds=seeds,
        modes=modes,
        display_fn=None,
    )
    return results_df, summary_df


def run_latent_tda(cfg: RunConfig, context: SequenceContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get(
            "LATENT_TDA_MODES",
            ["z", "z_latent_h0", "z_latent_h1", "z_latent_both"],
        ),
    )
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(5))))
    window = int(_override(cfg.latent_tda_window, context.dataset_config.get("LATENT_TDA_WINDOW", 20)))
    bins = int(_override(cfg.latent_tda_bins, context.dataset_config.get("LATENT_TDA_BINS", 16)))
    latent_epochs = int(_override(cfg.epochs, context.dataset_config.get("LATENT_TDA_EPOCHS", context.epochs)))
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
        BETTI_SCALE=context.betti_scale,
        N_STEPS=context.n_steps,
        PREDICT_STEPS_AHEAD=context.predict_steps_ahead,
        DEVICE=_select_device(cfg.device),
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
    )
    ml_tda_latent.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        RETRAIN_ENCODER=cfg.retrain_encoder,
        PREDICT_STEPS_AHEAD=context.predict_steps_ahead,
        LATENT_TDA_WINDOW=window,
        LATENT_TDA_BINS=bins,
        LATENT_TDA_EPOCHS=latent_epochs,
        LATENT_TDA_LR=latent_lr,
        RETRAIN_LATENT_TDA_PREDICTOR=cfg.retrain_predictor,
        RECOMPUTE_LATENT_TDA_FEATURES=recompute_features,
    )
    print(
        f"Latent TDA config: dataset={cfg.dataset}, seeds={seeds}, modes={modes}, "
        f"window={window}, bins={bins}, epochs={latent_epochs}, lr={latent_lr}, "
        f"predict_steps_ahead={context.predict_steps_ahead}, "
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


def run_topo_latent_tda(cfg: RunConfig, context: SequenceContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get(
            "LATENT_TDA_MODES",
            ["z", "z_latent_h0", "z_latent_h1", "z_latent_both"],
        ),
    )
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(5))))
    window = int(_override(cfg.latent_tda_window, context.dataset_config.get("LATENT_TDA_WINDOW", 20)))
    bins = int(_override(cfg.latent_tda_bins, context.dataset_config.get("LATENT_TDA_BINS", 16)))
    latent_epochs = int(_override(cfg.epochs, context.dataset_config.get("LATENT_TDA_EPOCHS", context.epochs)))
    latent_lr = float(_override(cfg.learning_rate, context.learning_rate))
    max_train = _override(cfg.latent_tda_max_train, context.dataset_config.get("LATENT_TDA_MAX_TRAIN"))
    max_test = _override(cfg.latent_tda_max_test, context.dataset_config.get("LATENT_TDA_MAX_TEST"))
    model_namespace = _topo_model_namespace(cfg.dataset, cfg.topo_ae_lambda)
    recompute_features = cfg.recompute_latent_tda_features or context.dataset_config.get(
        "RECOMPUTE_LATENT_TDA_FEATURES",
        True,
    )

    ml_tda.configure_runtime(
        DATASET=model_namespace,
        LATENT_DIM=context.latent_dim,
        BETTI_SCALE=context.betti_scale,
        N_STEPS=context.n_steps,
        PREDICT_STEPS_AHEAD=context.predict_steps_ahead,
        DEVICE=_select_device(cfg.device),
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
    )
    ml_tda_latent.configure_runtime(
        DATASET=model_namespace,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        RETRAIN_ENCODER=cfg.retrain_encoder,
        PREDICT_STEPS_AHEAD=context.predict_steps_ahead,
        LATENT_TDA_WINDOW=window,
        LATENT_TDA_BINS=bins,
        LATENT_TDA_EPOCHS=latent_epochs,
        LATENT_TDA_LR=latent_lr,
        RETRAIN_LATENT_TDA_PREDICTOR=cfg.retrain_predictor,
        RECOMPUTE_LATENT_TDA_FEATURES=recompute_features,
    )
    print(
        f"Topo latent-TDA config: dataset={cfg.dataset}, namespace={model_namespace}, "
        f"seeds={seeds}, modes={modes}, topo_ae_lambda={cfg.topo_ae_lambda}, "
        f"window={window}, bins={bins}, epochs={latent_epochs}, lr={latent_lr}, "
        f"predict_steps_ahead={context.predict_steps_ahead}, "
        f"max_train={max_train}, max_test={max_test}, recompute_features={recompute_features}"
    )

    rows = []
    for seed in seeds:
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

        for mode in modes:
            print(f"\n--- topo latent mode={mode} seed={seed} ---")
            train_features = ml_tda_latent.features_for_latent_tda_mode(train_payload, mode)
            test_features = ml_tda_latent.features_for_latent_tda_mode(test_payload, mode)
            model = ml_tda_latent.train_or_load_latent_tda_predictor(
                seed,
                mode,
                train_features,
                train_payload["z"],
            )
            test_mse, per_frame_mse = ml_tda_latent.eval_latent_tda_predictor(
                model,
                test_features,
                test_payload["z"],
            )
            row = {
                "dataset": cfg.dataset,
                "encoder": "topo_ae",
                "topo_ae_lambda": cfg.topo_ae_lambda,
                "seed": seed,
                "mode": mode,
                "predict_steps_ahead": context.predict_steps_ahead,
                "test_mse": float(test_mse),
                "warmup_excluded_mse": (
                    float(per_frame_mse[1:].mean()) if len(per_frame_mse) > 1 else float(test_mse)
                ),
                "encoder_path": str(encoder_path),
            }
            rows.append(row)
            print("topo latent TDA summary:", row)

    results_df = pd.DataFrame(rows)
    summary_df = ml_tda.summarize_metric_runs(
        results_df,
        group_cols="mode",
        metric_cols=["test_mse", "warmup_excluded_mse"],
        sort_metric="test_mse",
    )
    print("\nTopo latent-TDA per-run results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTopo latent-TDA mean +/- std:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary_df)
    return results_df, summary_df


def run_pixel_tda(cfg: RunConfig, context: SequenceContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = _override(
        cfg.modes,
        context.dataset_config.get("PIXEL_TDA_MODES", ["none", "h0", "h1", "both"]),
    )
    seeds = _override(cfg.run_seeds, context.dataset_config.get("RUN_SEEDS", list(range(3))))
    pixel_epochs = int(_override(cfg.epochs, context.dataset_config.get("PIXEL_TDA_EPOCHS", context.epochs)))
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
        BETTI_SCALE=context.betti_scale,
        N_STEPS=context.n_steps,
        PREDICT_STEPS_AHEAD=context.predict_steps_ahead,
        DEVICE=_select_device(cfg.device),
        AE_FRAME_BATCH_SIZE=context.ae_frame_batch_size,
        AE_MAX_FRAMES_PER_EPOCH=context.ae_max_frames_per_epoch,
    )
    ml_tda_pixel.configure_runtime(
        DATASET=cfg.dataset,
        LATENT_DIM=context.latent_dim,
        HIDDEN_DIM=context.hidden_dim,
        PREDICT_STEPS_AHEAD=context.predict_steps_ahead,
        N_STEPS=context.n_steps,
        BETTI_SCALE=context.betti_scale,
        PIXEL_TDA_EPOCHS=pixel_epochs,
        PIXEL_TDA_LR=pixel_lr,
        PIXEL_TDA_BATCH_SIZE=pixel_batch_size,
        PIXEL_TDA_FG_WEIGHT=fg_weight,
        PIXEL_TDA_FG_THRESHOLD=fg_threshold,
        PIXEL_TDA_RETRAIN_ENCODER=cfg.retrain_encoder,
        PIXEL_TDA_RETRAIN_PREDICTOR=cfg.retrain_predictor,
    )
    print(
        f"Pixel TDA config: dataset={cfg.dataset}, seeds={seeds}, modes={modes}, "
        f"epochs={pixel_epochs}, lr={pixel_lr}, batch={pixel_batch_size}, "
        f"fg_weight={fg_weight}, fg_threshold={fg_threshold}, "
        f"predict_steps_ahead={context.predict_steps_ahead}"
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
        choices=sorted(SEQUENCE_SCENARIOS),
        default="aux_tda",
        help="Notebook scenario to run. Initial migration covers sequence datasets only.",
    )
    parser.add_argument("--dataset", default="moving_mnist")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seeds", type=_parse_int_list, default=None, help="Comma-separated seeds, e.g. 0,1,2")
    parser.add_argument("--modes", type=_parse_str_list, default=None, help="Comma-separated mode names")
    parser.add_argument("--predict-steps-ahead", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--betti-scale", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--ae-frame-batch-size", type=int, default=None)
    parser.add_argument("--ae-max-frames-per-epoch", type=int, default=None)
    parser.add_argument("--force-rebuild-data-cache", action="store_true")
    parser.add_argument("--num-train-clips", type=int, default=None)
    parser.add_argument("--num-test-clips", type=int, default=None)
    parser.add_argument("--retrain-encoder", action="store_true")
    parser.add_argument(
        "--reuse-predictor",
        action="store_true",
        help="Load existing predictors when present. By default, match the notebook and retrain predictors.",
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
    parser.add_argument("--topo-ae-lambda", type=float, default=0.1)
    parser.add_argument("--topo-ae-epochs", type=int, default=None)
    parser.add_argument("--topo-ae-pair-batch-size", type=int, default=64)
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
        predict_steps_ahead=args.predict_steps_ahead,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        betti_scale=args.betti_scale,
        n_steps=args.n_steps,
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
        topo_ae_lambda=args.topo_ae_lambda,
        topo_ae_epochs=args.topo_ae_epochs,
        topo_ae_pair_batch_size=args.topo_ae_pair_batch_size,
        pixel_tda_batch_size=args.pixel_tda_batch_size,
        pixel_tda_fg_weight=args.pixel_tda_fg_weight,
        pixel_tda_fg_threshold=args.pixel_tda_fg_threshold,
        output_dir=args.output_dir,
        no_save=args.no_save,
    )


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    print(f"Control panel: scenario={cfg.scenario}, dataset={cfg.dataset}")

    context = load_sequence_context(cfg)
    if cfg.scenario == "sequence":
        results_df, summary_df = run_sequence(cfg, context)
    elif cfg.scenario == "topo_sequence":
        results_df, summary_df = run_topo_sequence(cfg, context)
    elif cfg.scenario == "latent_tda":
        results_df, summary_df = run_latent_tda(cfg, context)
    elif cfg.scenario == "topo_latent_tda":
        results_df, summary_df = run_topo_latent_tda(cfg, context)
    elif cfg.scenario == "aux_tda":
        results_df, summary_df = run_aux_tda(cfg, context)
    elif cfg.scenario == "pixel_tda":
        results_df, summary_df = run_pixel_tda(cfg, context)
    else:  # pragma: no cover - argparse choices prevent this.
        raise ValueError(f"Unsupported scenario: {cfg.scenario}")

    _save_results(cfg, context, results_df, summary_df)


if __name__ == "__main__":
    main()
