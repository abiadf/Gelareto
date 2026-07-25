#!/usr/bin/env python3
"""Make qualitative ground-truth vs z-only vs topo prediction figures.

The script reuses cached latent-TDA payloads, trained latent predictors, and
autoencoder decoders. It does not train models. It selects test examples where
the decoded z-only prediction has high pixel error and the topo-augmented
prediction improves over it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topo import config as topo_config
from topo import ml_tda
from topo import ml_tda_geoae
from topo import ml_tda_latent
from topo import ml_tda_topoae
from topo.ml_tda import SpatialEncoder, load_video_dataset, make_predictor, make_spatial_decoder


SCENARIO_LABELS = {
    "latent_tda": "AE",
    "geo_latent_tda": "GeoAE",
    "topo_latent_tda": "TopoAE",
}
SCENARIO_LINESTYLES = {
    "latent_tda": "-",
    "geo_latent_tda": "--",
    "topo_latent_tda": ":",
}
METHOD_COLORS = {
    "z": "#d55e00",
    "best_topo": "#0072b2",
    "best_pca": "#009e73",
    "best_kpca": "#cc79a7",
}
METHOD_LABELS = {
    "z": r"$z$",
    "best_topo": "Best topology",
    "best_pca": "Best PCA",
    "best_kpca": "Best KPCA",
}
DISPLAY_DATASET_LABELS = {
    "bouncing_disks": "Bouncing disks",
    "bouncing_rings": "Bouncing rings",
    "orbiting_disks": "Orbiting disks",
    "orbiting_rings": "Orbiting rings",
    "moving_mnist": "Moving MNIST",
    "lorenz96": "Lorenz-96",
    "electric_devices": "ElectricDevices",
    "glioblastoma": "Glioblastoma",
    "hela": "HeLa",
}
TOPO_FUSE_MODES = ["z_fuse_h1", "z_fuse_pi_h1", "z_fuse_landscape_h1", "z_fuse_perslay_h1"]
PCA_MODES = ["z_pca_h1", "z_fuse_pca_h1"]
KPCA_MODES = ["z_kpca_h1", "z_fuse_kpca_h1"]


def _namespace(dataset: str, scenario: str, geo_lambda: float, topo_lambda: float, topo_distance: str) -> str:
    if scenario == "latent_tda":
        return dataset
    if scenario == "geo_latent_tda":
        return f"{dataset}_geoae_lam{geo_lambda:g}"
    if scenario == "topo_latent_tda":
        return f"{dataset}_topoae_lam{topo_lambda:g}_{topo_distance}"
    raise ValueError(f"Unsupported scenario={scenario!r}. Use latent_tda, geo_latent_tda, or topo_latent_tda.")


def _display_dataset(dataset: str) -> str:
    return DISPLAY_DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())


def _latest(paths: list[Path], what: str) -> Path:
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError(f"No matching {what} found.")
    return max(paths, key=lambda p: p.stat().st_mtime)


def _safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _load_dataset_tensors(dataset: str) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = dict(topo_config.DATASET_CONFIGS[dataset])
    x_train_np, x_test_np = load_video_dataset(cfg)
    x_train = torch.from_numpy(x_train_np).unsqueeze(2)
    x_test = torch.from_numpy(x_test_np).unsqueeze(2)
    return x_train, x_test


def _payload_path(namespace: str, seed: int, split: str, window: int, bins: int) -> Path:
    feature_dir = Path("models") / namespace / "latent_tda_features"
    candidates = list(feature_dir.glob(f"seed{seed}_{split}_T*_B*_H*_W*_latent*_win{window}_bins{bins}.pt"))
    return _latest(candidates, f"{split} latent payload in {feature_dir}")


def _decoder_path(namespace: str, scenario: str, seed: int, height: int, width: int, latent_dim: int) -> Path:
    if scenario == "geo_latent_tda":
        root = Path("models") / namespace / "geoae_decoders"
    elif scenario == "topo_latent_tda":
        root = Path("models") / namespace / "topoae_decoders"
    else:
        root = Path("models") / namespace / "decoders"
    candidates = list(root.glob(f"decoder_seed{seed}_T*_B*_H{height}_W{width}_latent{latent_dim}*.pt"))
    return _latest(candidates, f"decoder in {root}")


def _predictor_path(
    namespace: str,
    seed: int,
    horizon: int,
    mode: str,
    predictor_type: str,
    window: int,
    bins: int,
) -> Path:
    root = Path("models") / namespace / "latent_tda_predictors"
    architecture = "fusion" if mode.startswith("z_fuse_") else "direct"
    candidates = list(
        root.glob(
            f"model_seed{seed}_pred{horizon}_{mode}_{architecture}_*stdz_"
            f"{predictor_type}_win{window}_bins{bins}.pt"
        )
    )
    return _latest(candidates, f"predictor for mode={mode} in {root}")


def _best_modes_for_dataset_encoder(args: argparse.Namespace, scenario: str) -> dict[str, str]:
    summary_path = Path(args.control_summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing control summary table: {summary_path}")
    df = pd.read_csv(summary_path)
    row = df[(df["dataset"] == args.dataset) & (df["scenario"] == scenario)]
    if row.empty:
        raise ValueError(f"No row for dataset={args.dataset}, scenario={scenario} in {summary_path}")
    row = row.iloc[0]
    best_topo_mode = str(row["best_topo_mode"])
    pca_values = {
        "z_pca_h1": float(row["z_pca_mean"]),
        "z_fuse_pca_h1": float(row["fuse_pca_mean"]),
    }
    kpca_values = {
        "z_kpca_h1": float(row["z_kpca_mean"]),
        "z_fuse_kpca_h1": float(row["fuse_kpca_mean"]),
    }
    return {
        "z": args.baseline_mode,
        "best_topo": best_topo_mode,
        "best_pca": min(pca_values, key=pca_values.get),
        "best_kpca": min(kpca_values, key=kpca_values.get),
    }


def _load_predictor(path: Path, mode: str, input_dim: int, latent_dim: int, hidden_dim: int, predictor_type: str):
    ml_tda.configure_runtime(LATENT_DIM=latent_dim, PREDICTOR_TYPE=predictor_type)
    ml_tda_latent.configure_runtime(LATENT_DIM=latent_dim, HIDDEN_DIM=hidden_dim)
    if mode.startswith("z_fuse_"):
        topo_dim = input_dim - latent_dim
        model = ml_tda_latent.FusedTopologicalPredictor(
            latent_dim=latent_dim,
            topo_dim=topo_dim,
            fused_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )
    else:
        model = make_predictor(input_dim=input_dim, hidden_dim=hidden_dim, predictor_type=predictor_type)

    checkpoint = _safe_load(path)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        ml_tda_latent._attach_latent_standardizers(
            model,
            checkpoint["feature_mean"],
            checkpoint["feature_std"],
            checkpoint["target_mean"],
            checkpoint["target_std"],
        )
    else:
        model.load_state_dict(checkpoint)
        feature_mean = torch.zeros((1, 1, input_dim), dtype=torch.float32)
        feature_std = torch.ones((1, 1, input_dim), dtype=torch.float32)
        target_mean = torch.zeros((1, 1, latent_dim), dtype=torch.float32)
        target_std = torch.ones((1, 1, latent_dim), dtype=torch.float32)
        ml_tda_latent._attach_latent_standardizers(model, feature_mean, feature_std, target_mean, target_std)
    model.eval()
    return model


@torch.no_grad()
def _predict_latents(model, features: torch.Tensor, horizon: int) -> torch.Tensor:
    device = ml_tda.get_runtime_device()
    model.to(device).eval()
    feature_mean = model.feature_mean.to(device)
    feature_std = model.feature_std.to(device)
    target_mean = model.target_mean.to(device)
    target_std = model.target_std.to(device)
    x = ml_tda_latent._standardize(features[:-horizon].to(device), feature_mean, feature_std)
    pred_standardized = model(x)
    return (pred_standardized * target_std + target_mean).detach().cpu()


def _latent_errors_for_modes(
    namespace: str,
    seed: int,
    horizon: int,
    window: int,
    bins: int,
    latent_dim: int,
    hidden_dim: int,
    predictor_type: str,
    train_payload,
    test_payload,
    method_to_mode: dict[str, str],
) -> dict[str, torch.Tensor]:
    errors = {}
    target_z = test_payload["z"][horizon:]
    for method, mode in method_to_mode.items():
        train_features, test_features = ml_tda_latent.features_for_latent_tda_mode_pair(
            train_payload,
            test_payload,
            mode,
            train_control_seed=seed,
            test_control_seed=seed + 10_000,
        )
        model_path = _predictor_path(namespace, seed, horizon, mode, predictor_type, window, bins)
        model = _load_predictor(model_path, mode, train_features.shape[-1], latent_dim, hidden_dim, predictor_type)
        pred_latent = _predict_latents(model, test_features, horizon)
        errors[method] = ((pred_latent - target_z) ** 2).mean(dim=-1)
    return errors


@torch.no_grad()
def _decode(decoder, z: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    device = ml_tda.get_runtime_device()
    decoder.to(device).eval()
    flat = z.reshape(-1, z.shape[-1])
    out = []
    for start in range(0, len(flat), batch_size):
        out.append(decoder(flat[start : start + batch_size].to(device)).detach().cpu())
    return torch.cat(out, dim=0).reshape(z.shape[0], z.shape[1], 1, *decoder.output_size)


def _take_matching_test_subset(x_test: torch.Tensor, payload: dict, seed: int) -> torch.Tensor:
    target_b = int(payload["z"].shape[1])
    return ml_tda_latent.take_batch_subset(x_test, target_b, seed=seed + 1)


def _select_examples(
    err_z: torch.Tensor,
    err_topo: torch.Tensor,
    n: int,
    min_gain: float,
    min_source_t: int,
) -> list[tuple[int, int]]:
    gain = err_z - err_topo
    if min_source_t > 0:
        gain = gain.clone()
        gain[:min_source_t] = -torch.inf
    flat_gain = gain.flatten()
    order = torch.argsort(flat_gain, descending=True).tolist()
    selected = []
    used_clips = set()
    _, b_count = gain.shape
    for idx in order:
        t = idx // b_count
        b = idx % b_count
        if float(flat_gain[idx]) < min_gain:
            continue
        if int(b) in used_clips and len(used_clips) < b_count:
            continue
        selected.append((int(t), int(b)))
        used_clips.add(int(b))
        if len(selected) >= n:
            break
    if len(selected) < n:
        for idx in order:
            t = idx // b_count
            b = idx % b_count
            pair = (int(t), int(b))
            if pair not in selected:
                selected.append(pair)
            if len(selected) >= n:
                break
    return selected


def _to_image(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().squeeze().cpu().numpy()
    return np.clip(arr, 0.0, 1.0)


def _plot_frame_row(
    axes,
    row: int,
    gt,
    pred_z,
    pred_topo,
    title_prefix: str,
    baseline_label: str,
    topo_label: str,
    err_z: float,
    err_topo: float,
):
    panels = [
        ("Ground truth", gt),
        (f"{baseline_label}\nMSE={err_z:.4f}", pred_z),
        (f"{topo_label}\nMSE={err_topo:.4f}", pred_topo),
    ]
    for col, (title, img) in enumerate(panels):
        ax = axes[row, col]
        ax.imshow(_to_image(img), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(title, fontsize=12)
        if col == 0:
            ax.set_ylabel(title_prefix, fontsize=11)


def _plot_timeseries_row(
    axes,
    row: int,
    gt,
    pred_z,
    pred_topo,
    title_prefix: str,
    baseline_label: str,
    topo_label: str,
    err_z: float,
    err_topo: float,
):
    curves = [
        ("Ground truth", _to_image(gt).mean(axis=0), "#222222"),
        (f"{baseline_label} MSE={err_z:.4f}", _to_image(pred_z).mean(axis=0), "#d55e00"),
        (f"{topo_label} MSE={err_topo:.4f}", _to_image(pred_topo).mean(axis=0), "#0072b2"),
    ]
    for col, (title, y, color) in enumerate(curves):
        ax = axes[row, col]
        ax.plot(y, color=color, linewidth=2)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25, linewidth=0.5)
        if row == 0:
            ax.set_title(title, fontsize=12)
        if col == 0:
            ax.set_ylabel(title_prefix, fontsize=11)


def _plot_error_curves(
    args: argparse.Namespace,
    errors_by_mode: dict[str, torch.Tensor],
    selected: list[tuple[int, int]],
    scenario_label: str,
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.2, 1.0]})
    first_err = next(iter(errors_by_mode.values()))
    time = np.arange(first_err.shape[0]) + args.horizon_value
    colors = {
        args.baseline_mode: "#d55e00",
        args.pca_direct_mode: "#cc79a7",
        args.pca_mode: "#009e73",
        args.topo_mode: "#0072b2",
    }

    for mode, err in errors_by_mode.items():
        color = colors.get(mode, "#666666")
        for _, clip_idx in selected:
            axes[0].plot(time, err[:, clip_idx].numpy(), color=color, alpha=0.18, linewidth=1.0)
        mean_curve = torch.stack([err[:, b] for _, b in selected], dim=0).mean(dim=0)
        axes[0].plot(time, mean_curve.numpy(), color=color, linewidth=2.2, label=mode)

    axes[0].set_title("Latent prediction error over time")
    axes[0].set_xlabel("target sequence time step")
    axes[0].set_ylabel("latent MSE")
    axes[0].set_yscale(args.error_yscale)
    axes[0].grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False)

    labels = list(errors_by_mode)
    per_mode_clip_means = []
    for mode in labels:
        vals = torch.stack([errors_by_mode[mode][:, b] for _, b in selected], dim=0).mean(dim=1)
        per_mode_clip_means.append(vals.numpy())
    x = np.arange(len(labels))
    per_clip = np.stack(per_mode_clip_means, axis=1)
    for clip_i, row in enumerate(per_clip):
        axes[1].scatter(x, row, color="#999999", alpha=0.32, s=11, zorder=2)
    means = per_clip.mean(axis=0)
    for i, mode in enumerate(labels):
        axes[1].scatter(
            x[i],
            means[i],
            s=68,
            color=colors.get(mode, "#666666"),
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
            label=mode,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_title("Per-sequence mean error")
    axes[1].set_ylabel("")
    axes[1].set_yscale(args.error_yscale)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)

    fig.suptitle(
        f"{_display_dataset(args.dataset)}: latent forecasting error curves ({scenario_label})",
        fontsize=13,
        y=1.03,
    )
    fig.tight_layout()
    mode_tag = "_vs_".join(labels)
    scale_tag = "" if args.error_yscale == "linear" else f"_{args.error_yscale}"
    stem = f"latent_error_curves_{args.dataset}_{args.scenario}_seed{args.seed}_{mode_tag}{scale_tag}"
    out_path = out_dir / f"{stem}.{args.format}"
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    if args.format != "png":
        fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"Saved latent error curve figure: {out_path}")
    return out_path


def _plot_encoder_comparison_error_curves(
    args: argparse.Namespace,
    errors_by_scenario_method: dict[str, dict[str, torch.Tensor]],
    mode_by_scenario_method: dict[str, dict[str, str]],
    selected: list[tuple[int, int]],
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.55), gridspec_kw={"width_ratios": [1.22, 1.0]})
    first_err = next(iter(next(iter(errors_by_scenario_method.values())).values()))
    time = np.arange(first_err.shape[0]) + args.horizon_value
    methods = ["z", "best_topo", "best_pca", "best_kpca"]

    for scenario, errors_by_method in errors_by_scenario_method.items():
        linestyle = SCENARIO_LINESTYLES[scenario]
        for method in methods:
            err = errors_by_method[method]
            mean_curve = torch.stack([err[:, b] for _, b in selected], dim=0).mean(dim=0)
            axes[0].plot(
                time,
                mean_curve.numpy(),
                color=METHOD_COLORS[method],
                linestyle=linestyle,
                linewidth=1.7,
                alpha=0.95,
            )

    axes[0].set_title("Latent prediction error over time")
    axes[0].set_xlabel("target sequence time step")
    axes[0].set_ylabel("latent MSE")
    axes[0].set_yscale(args.error_yscale)
    axes[0].grid(alpha=0.25, linewidth=0.6)

    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], linewidth=1.9, label=METHOD_LABELS[m])
        for m in methods
    ]
    encoder_handles = [
        Line2D([0], [0], color="#333333", linestyle=SCENARIO_LINESTYLES[s], linewidth=1.7, label=SCENARIO_LABELS[s])
        for s in errors_by_scenario_method
    ]
    first_legend = axes[0].legend(handles=method_handles, frameon=False, loc="upper left", fontsize=8.8)
    axes[0].add_artist(first_legend)
    axes[0].legend(handles=encoder_handles, frameon=False, loc="upper right", fontsize=8.8)

    x = np.arange(len(methods))
    for scenario, errors_by_method in errors_by_scenario_method.items():
        for i, method in enumerate(methods):
            vals = torch.stack([errors_by_method[method][:, b] for _, b in selected], dim=0).mean(dim=1).numpy()
            axes[1].scatter(
                np.full_like(vals, x[i], dtype=float),
                vals,
                color=METHOD_COLORS[method],
                alpha=0.24,
                s=10,
                linewidth=0,
                zorder=2,
            )
            axes[1].scatter(
                x[i],
                vals.mean(),
                color=METHOD_COLORS[method],
                edgecolor="#333333",
                linewidth=0.65,
                s=42,
                alpha=0.95,
                zorder=3,
            )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha="right")
    axes[1].set_title("Per-sequence mean error")
    axes[1].set_ylabel("")
    axes[1].set_yscale(args.error_yscale)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)

    fig.suptitle(f"{_display_dataset(args.dataset)}: latent forecasting error curves", fontsize=13, y=1.02)
    fig.tight_layout()
    scenario_tag = "_".join(SCENARIO_LABELS[s].lower() for s in errors_by_scenario_method)
    scale_tag = "" if args.error_yscale == "linear" else f"_{args.error_yscale}"
    stem = f"latent_error_curves_{args.dataset}_{scenario_tag}_best_topo_pca_kpca_seed{args.seed}{scale_tag}"
    out_path = out_dir / f"{stem}.{args.format}"
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    if args.format != "png":
        fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"Saved latent error curve figure: {out_path}")
    print("Modes used:")
    for scenario, mode_by_method in mode_by_scenario_method.items():
        print(f"  {SCENARIO_LABELS[scenario]}: {mode_by_method}")
    return out_path


def make_figure(args: argparse.Namespace) -> Path:
    dataset_cfg = topo_config.DATASET_CONFIGS[args.dataset]
    horizon = int(args.horizon if args.horizon is not None else dataset_cfg.get("HORIZON", 5))
    args.horizon_value = horizon
    window = int(args.window if args.window is not None else dataset_cfg.get("LATENT_TDA_WINDOW", 20))
    bins = int(args.bins if args.bins is not None else dataset_cfg.get("LATENT_TDA_BINS", 16))
    latent_dim = int(dataset_cfg.get("LATENT_DIM", 128))
    hidden_dim = int(dataset_cfg.get("HIDDEN_DIM", 128))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_encoders and args.plot_kind == "error_curve":
        scenarios = [part.strip() for part in args.compare_scenarios.split(",") if part.strip()]
        errors_by_scenario_method = {}
        mode_by_scenario_method = {}
        for scenario in scenarios:
            namespace = _namespace(args.dataset, scenario, args.geo_lambda, args.topo_lambda, args.topo_distance)
            ml_tda.configure_runtime(
                DATASET=namespace,
                LATENT_DIM=latent_dim,
                HORIZON=horizon,
                DEVICE=args.device,
                PREDICTOR_TYPE=args.predictor_type,
            )
            ml_tda_latent.configure_runtime(
                DATASET=namespace,
                LATENT_DIM=latent_dim,
                HIDDEN_DIM=hidden_dim,
                HORIZON=horizon,
                LATENT_TDA_WINDOW=window,
                LATENT_TDA_BINS=bins,
                RETRAIN_LATENT_TDA_PREDICTOR=False,
            )
            train_payload = _safe_load(_payload_path(namespace, args.seed, "train", window, bins))
            test_payload = _safe_load(_payload_path(namespace, args.seed, "test", window, bins))
            method_to_mode = _best_modes_for_dataset_encoder(args, scenario)
            mode_by_scenario_method[scenario] = method_to_mode
            errors_by_scenario_method[scenario] = _latent_errors_for_modes(
                namespace,
                args.seed,
                horizon,
                window,
                bins,
                latent_dim,
                hidden_dim,
                args.predictor_type,
                train_payload,
                test_payload,
                method_to_mode,
            )

        primary_scenario = args.scenario if args.scenario in errors_by_scenario_method else scenarios[0]
        primary_errors = errors_by_scenario_method[primary_scenario]
        min_source_t = int(args.min_source_t if args.min_source_t is not None else max(0, min(window - 1, primary_errors["z"].shape[0] - 1)))
        selected = _select_examples(
            primary_errors["z"],
            primary_errors["best_topo"],
            args.n_examples,
            args.min_gain,
            min_source_t,
        )
        return _plot_encoder_comparison_error_curves(
            args,
            errors_by_scenario_method,
            mode_by_scenario_method,
            selected,
            out_dir,
        )

    namespace = _namespace(args.dataset, args.scenario, args.geo_lambda, args.topo_lambda, args.topo_distance)
    ml_tda.configure_runtime(
        DATASET=namespace,
        LATENT_DIM=latent_dim,
        HORIZON=horizon,
        DEVICE=args.device,
        PREDICTOR_TYPE=args.predictor_type,
    )
    ml_tda_latent.configure_runtime(
        DATASET=namespace,
        LATENT_DIM=latent_dim,
        HIDDEN_DIM=hidden_dim,
        HORIZON=horizon,
        LATENT_TDA_WINDOW=window,
        LATENT_TDA_BINS=bins,
        RETRAIN_LATENT_TDA_PREDICTOR=False,
    )

    x_train, x_test = _load_dataset_tensors(args.dataset)
    train_payload = _safe_load(_payload_path(namespace, args.seed, "train", window, bins))
    test_payload = _safe_load(_payload_path(namespace, args.seed, "test", window, bins))
    x_test_subset = _take_matching_test_subset(x_test, test_payload, args.seed)

    train_z_features, test_z_features = ml_tda_latent.features_for_latent_tda_mode_pair(
        train_payload,
        test_payload,
        args.baseline_mode,
        train_control_seed=args.seed,
        test_control_seed=args.seed + 10_000,
    )
    train_topo_features, test_topo_features = ml_tda_latent.features_for_latent_tda_mode_pair(
        train_payload,
        test_payload,
        args.topo_mode,
        train_control_seed=args.seed,
        test_control_seed=args.seed + 10_000,
    )
    pca_mode = args.pca_mode.strip() if args.pca_mode else ""
    pca_direct_mode = args.pca_direct_mode.strip() if args.pca_direct_mode else ""
    pca_modes = [mode for mode in [pca_direct_mode, pca_mode] if mode]
    pca_feature_pairs = {}
    for mode in pca_modes:
        pca_feature_pairs[mode] = ml_tda_latent.features_for_latent_tda_mode_pair(
            train_payload,
            test_payload,
            mode,
            train_control_seed=args.seed,
            test_control_seed=args.seed + 10_000,
        )

    z_model_path = _predictor_path(namespace, args.seed, horizon, args.baseline_mode, args.predictor_type, window, bins)
    topo_model_path = _predictor_path(namespace, args.seed, horizon, args.topo_mode, args.predictor_type, window, bins)
    z_model = _load_predictor(z_model_path, args.baseline_mode, train_z_features.shape[-1], latent_dim, hidden_dim, args.predictor_type)
    topo_model = _load_predictor(topo_model_path, args.topo_mode, train_topo_features.shape[-1], latent_dim, hidden_dim, args.predictor_type)
    pca_model_paths = {}
    pca_models = {}
    for mode, (train_features, _) in pca_feature_pairs.items():
        pca_model_paths[mode] = _predictor_path(namespace, args.seed, horizon, mode, args.predictor_type, window, bins)
        pca_models[mode] = _load_predictor(
            pca_model_paths[mode],
            mode,
            train_features.shape[-1],
            latent_dim,
            hidden_dim,
            args.predictor_type,
        )

    decoder_path = _decoder_path(namespace, args.scenario, args.seed, x_train.shape[-2], x_train.shape[-1], latent_dim)
    decoder = make_spatial_decoder(args.decoder_type, latent_dim=latent_dim, output_size=x_train.shape[-2:])
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

    pred_z_latent = _predict_latents(z_model, test_z_features, horizon)
    pred_topo_latent = _predict_latents(topo_model, test_topo_features, horizon)
    pred_pca_latents = {
        mode: _predict_latents(pca_models[mode], pca_feature_pairs[mode][1], horizon)
        for mode in pca_models
    }
    target_z = test_payload["z"][horizon:]
    latent_err_z = ((pred_z_latent - target_z) ** 2).mean(dim=-1)
    latent_err_topo = ((pred_topo_latent - target_z) ** 2).mean(dim=-1)
    latent_err_pcas = {
        mode: ((pred_latent - target_z) ** 2).mean(dim=-1)
        for mode, pred_latent in pred_pca_latents.items()
    }
    err_z = latent_err_z
    err_topo = latent_err_topo
    min_source_t = int(args.min_source_t if args.min_source_t is not None else max(0, min(window - 1, err_z.shape[0] - 1)))
    selected = _select_examples(err_z, err_topo, args.n_examples, args.min_gain, min_source_t)

    scenario_label = {
        "latent_tda": "AE",
        "geo_latent_tda": "GeoAE",
        "topo_latent_tda": "TopoAE",
    }[args.scenario]
    if args.plot_kind == "error_curve":
        errors_by_mode = {args.baseline_mode: latent_err_z}
        for mode in pca_modes:
            errors_by_mode[mode] = latent_err_pcas[mode]
        errors_by_mode[args.topo_mode] = latent_err_topo
        out_path = _plot_error_curves(args, errors_by_mode, selected, scenario_label, out_dir)
        for t, b in selected:
            gain = float(latent_err_z[t, b] - latent_err_topo[t, b])
            msg = (
                f"selected clip={b} source_t={t} target_t={t + horizon} "
                f"z_latent_mse={float(latent_err_z[t,b]):.6f} "
            )
            for mode in pca_modes:
                msg += f"{mode}_latent_mse={float(latent_err_pcas[mode][t,b]):.6f} "
            msg += f"topo_latent_mse={float(latent_err_topo[t,b]):.6f} gain={gain:.6f}"
            print(
                msg
            )
        return out_path

    pred_z_x = _decode(decoder, pred_z_latent)
    pred_topo_x = _decode(decoder, pred_topo_latent)
    if args.target_space == "decoded":
        target_x = _decode(decoder, target_z)
        target_label = "Decoded target"
    else:
        target_x = ml_tda.tensor_to_model_float(x_test_subset[horizon:])
        target_label = "Ground truth"
    err_z = ((pred_z_x - target_x) ** 2).mean(dim=(2, 3, 4))
    err_topo = ((pred_topo_x - target_x) ** 2).mean(dim=(2, 3, 4))
    selected = _select_examples(err_z, err_topo, args.n_examples, args.min_gain, min_source_t)

    is_timeseries = args.dataset in {"lorenz96", "electric_devices"}
    fig, axes = plt.subplots(args.n_examples, 3, figsize=(7.2, 2.35 * args.n_examples), squeeze=False)
    plot_row = _plot_timeseries_row if is_timeseries else _plot_frame_row
    for row, (t, b) in enumerate(selected):
        title_prefix = f"clip {b}, t+{horizon}={t + horizon}"
        plot_row(
            axes,
            row,
            target_x[t, b],
            pred_z_x[t, b],
            pred_topo_x[t, b],
            title_prefix,
            args.baseline_mode,
            args.topo_mode,
            float(err_z[t, b]),
            float(err_topo[t, b]),
        )

    fig.suptitle(
        f"{_display_dataset(args.dataset)}: qualitative next-frame prediction ({scenario_label}, {args.baseline_mode} vs {args.topo_mode}, target={args.target_space})",
        fontsize=13,
        y=1.01,
    )
    axes[0, 0].set_title(target_label, fontsize=12)
    fig.tight_layout()
    stem = f"qualitative_{args.dataset}_{args.scenario}_seed{args.seed}_{args.baseline_mode}_vs_{args.topo_mode}_{args.target_space}"
    out_path = out_dir / f"{stem}.{args.format}"
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    if args.format != "png":
        fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    print(f"Saved qualitative figure: {out_path}")
    print(f"Decoder: {decoder_path}")
    print(f"z predictor: {z_model_path}")
    for mode, path in pca_model_paths.items():
        print(f"{mode} predictor: {path}")
    print(f"topo predictor: {topo_model_path}")
    for t, b in selected:
        gain = float(err_z[t, b] - err_topo[t, b])
        print(f"selected clip={b} source_t={t} target_t={t + horizon} z_mse={float(err_z[t,b]):.6f} topo_mse={float(err_topo[t,b]):.6f} gain={gain:.6f}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="bouncing_rings")
    parser.add_argument("--scenario", default="geo_latent_tda", choices=["latent_tda", "geo_latent_tda", "topo_latent_tda"])
    parser.add_argument("--baseline-mode", default="z")
    parser.add_argument("--topo-mode", default="z_fuse_h1")
    parser.add_argument("--pca-direct-mode", default="", help="Optional direct PCA-control mode, e.g. z_pca_h1.")
    parser.add_argument("--pca-mode", default="", help="Optional fusion PCA-control mode, e.g. z_fuse_pca_h1.")
    parser.add_argument("--compare-encoders", action="store_true", help="Plot AE/GeoAE/TopoAE together using best topo/PCA/KPCA modes from the control summary.")
    parser.add_argument("--compare-scenarios", default="latent_tda,geo_latent_tda,topo_latent_tda")
    parser.add_argument("--control-summary", default="latex_tables/pca_kpca_control_summary.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--n-examples", type=int, default=3)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--min-source-t", type=int, default=None, help="Earliest source time to consider. Default uses window-1.")
    parser.add_argument("--geo-lambda", type=float, default=0.1)
    parser.add_argument("--topo-lambda", type=float, default=0.1)
    parser.add_argument("--topo-distance", default="signature")
    parser.add_argument("--decoder-type", default="mlp")
    parser.add_argument("--predictor-type", default="lstm")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="images/qualitative")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    parser.add_argument("--target-space", default="raw", choices=["raw", "decoded"], help="raw compares to X; decoded compares to decoded target z.")
    parser.add_argument("--plot-kind", default="panels", choices=["panels", "error_curve"])
    parser.add_argument("--error-yscale", default="linear", choices=["linear", "log"], help="Y-axis scale for error-curve plots.")
    return parser.parse_args()


if __name__ == "__main__":
    make_figure(parse_args())
