from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topo import ml_runner


DEFAULT_DATASETS = (
    "bouncing_rings,bouncing_disks,orbiting_rings,orbiting_disks,"
    "moving_mnist,lorenz96,electric_devices,glioblastoma,hela,growing_tree,hirros"
)


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strs(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def split_train_val(x_train: torch.Tensor, *, val_fraction: float, max_train: int | None, max_val: int | None):
    n_total = int(x_train.shape[1])
    n_val = max(1, int(round(n_total * val_fraction)))
    n_val = min(n_val, n_total - 1)
    val = x_train[:, -n_val:]
    train = x_train[:, :-n_val]
    if max_train is not None:
        train = train[:, : min(max_train, int(train.shape[1]))]
    if max_val is not None:
        val = val[:, : min(max_val, int(val.shape[1]))]
    return train, val


def make_base_cfg(dataset: str, args: argparse.Namespace) -> ml_runner.RunConfig:
    cfg = ml_runner.parse_args(
        [
            "--scenario",
            args.scenario,
            "--dataset",
            dataset,
            "--device",
            args.device,
            "--modes",
            "z",
            "--seeds",
            ",".join(str(seed) for seed in args.seeds),
            "--latent-dim",
            str(args.latent_dim),
            "--horizon",
            str(args.horizon),
            "--latent-tda-window",
            str(args.latent_tda_window),
            "--ae-epochs",
            str(args.ae_epochs),
            "--learning-rate",
            str(args.encoder_learning_rate),
            "--output-dir",
            str(args.output_dir / "_internal"),
            "--no-save",
            "--dinov2-repo",
            args.dinov2_repo,
            "--dinov2-batch-size",
            str(args.dinov2_batch_size),
            "--dinov2-image-size",
            str(args.dinov2_image_size),
            "--dinov2-trainable-blocks",
            str(args.dinov2_trainable_blocks),
            "--dinov2-finetune-epochs",
            str(args.dinov2_finetune_epochs),
            "--dinov2-encoder-lr",
            str(args.dinov2_encoder_lr),
            "--dinov2-clip-batch-size",
            str(args.dinov2_clip_batch_size),
            "--clip-repo",
            args.clip_repo,
            "--clip-batch-size",
            str(args.clip_batch_size),
            "--clip-image-size",
            str(args.clip_image_size),
        ]
    )
    return replace(
        cfg,
        retrain_predictor=True,
        recompute_latent_tda_features=False,
        latent_tda_max_train=None,
        latent_tda_max_test=None,
    )


def run_scenario(cfg: ml_runner.RunConfig, context: ml_runner.VideoContext):
    if cfg.scenario == "latent_tda":
        return ml_runner.run_latent_tda(cfg, context)
    if cfg.scenario == "dinov2_latent_tda":
        return ml_runner.run_dinov2_latent_tda(cfg, context)
    if cfg.scenario == "dinov2_finetune_latent_tda":
        return ml_runner.run_dinov2_latent_tda(cfg, context, fine_tune=True)
    if cfg.scenario == "vjepa_latent_tda":
        return ml_runner.run_vjepa_latent_tda(cfg, context)
    if cfg.scenario == "clip_latent_tda":
        return ml_runner.run_clip_latent_tda(cfg, context)
    raise ValueError(f"Unsupported tuning scenario: {cfg.scenario}")


def run_one_dataset(dataset: str, args: argparse.Namespace) -> tuple[list[dict], dict]:
    base_cfg = make_base_cfg(dataset, args)
    base_context = ml_runner.load_video_context(base_cfg)
    train, val = split_train_val(
        base_context.x_train,
        val_fraction=args.val_fraction,
        max_train=args.max_train_clips,
        max_val=args.max_val_clips,
    )
    context = replace(base_context, x_train=train, x_test=val)

    rows = []
    grid = list(itertools.product(args.learning_rates, args.hidden_dims, args.predictor_epochs))
    for idx, (lr, hidden_dim, epochs) in enumerate(grid, start=1):
        print(
            f"\n[{dataset}] grid {idx}/{len(grid)}: "
            f"lr={lr:g}, hidden_dim={hidden_dim}, predictor_epochs={epochs}"
        )
        cfg = replace(
            base_cfg,
            predictor_learning_rate=float(lr),
            hidden_dim=int(hidden_dim),
            predictor_epochs=int(epochs),
        )
        result_df, summary_df = run_scenario(
            cfg,
            replace(
                context,
                hidden_dim=int(hidden_dim),
                predictor_epochs=int(epochs),
                predictor_learning_rate=float(lr),
            ),
        )
        val_mse = float(result_df["test_mse"].mean())
        val_r2 = float(result_df["latent_r2"].mean())
        row = {
            "dataset": dataset,
            "predictor_learning_rate": float(lr),
            "hidden_dim": int(hidden_dim),
            "predictor_epochs": int(epochs),
            "n_runs": int(result_df["seed"].nunique()) if "seed" in result_df else len(result_df),
            "val_mse": val_mse,
            "val_r2": val_r2,
        }
        rows.append(row)
        print("tuning summary:", row)

    best = min(rows, key=lambda row: row["val_mse"])
    return rows, best


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick z-only latent predictor hyperparameter search.")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--scenario",
        default="latent_tda",
        choices=["latent_tda", "dinov2_latent_tda", "dinov2_finetune_latent_tda", "vjepa_latent_tda", "clip_latent_tda"],
        help="Scenario to tune using z-only validation MSE.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seeds", type=parse_csv_ints, default=[0])
    parser.add_argument("--learning-rates", type=parse_csv_floats, default=[1e-4, 3e-4, 1e-3, 3e-3, 5e-3, 1e-2])
    parser.add_argument("--hidden-dims", type=parse_csv_ints, default=[64, 128, 256])
    parser.add_argument("--predictor-epochs", type=parse_csv_ints, default=[8, 20, 40])
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--latent-tda-window", type=int, default=15)
    parser.add_argument("--ae-epochs", type=int, default=10)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-clips", type=int, default=None)
    parser.add_argument("--max-val-clips", type=int, default=None)
    parser.add_argument("--dinov2-repo", default="facebook/dinov2-small")
    parser.add_argument("--dinov2-batch-size", type=int, default=64)
    parser.add_argument("--dinov2-image-size", type=int, default=224)
    parser.add_argument("--dinov2-trainable-blocks", type=int, default=1)
    parser.add_argument("--dinov2-finetune-epochs", type=int, default=3)
    parser.add_argument("--dinov2-encoder-lr", type=float, default=3e-5)
    parser.add_argument("--dinov2-clip-batch-size", type=int, default=8)
    parser.add_argument("--clip-repo", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--clip-image-size", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, default=Path("results/hparam_search"))
    args = parser.parse_args()

    args.datasets = parse_csv_strs(args.datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_json = args.output_dir / "latent_z_best_hparams.json"
    grid_csv = args.output_dir / "latent_z_hparam_grid.csv"
    existing_payload = json.loads(best_json.read_text()) if best_json.exists() else {}
    existing_best = dict(existing_payload.get("best_by_dataset", {}))
    existing_grid = pd.read_csv(grid_csv) if grid_csv.exists() else pd.DataFrame()

    all_rows = []
    best_by_dataset = {}
    for dataset in args.datasets:
        rows, best = run_one_dataset(dataset, args)
        all_rows.extend(rows)
        best_by_dataset[dataset] = {
            "predictor_learning_rate": best["predictor_learning_rate"],
            "hidden_dim": best["hidden_dim"],
            "predictor_epochs": best["predictor_epochs"],
            "val_mse": best["val_mse"],
            "val_r2": best["val_r2"],
            "selection": "z-only validation MSE",
        }

    new_grid = pd.DataFrame(all_rows)
    if not existing_grid.empty:
        existing_grid = existing_grid[~existing_grid["dataset"].isin(args.datasets)]
        new_grid = pd.concat([existing_grid, new_grid], ignore_index=True)
    grid_df = new_grid.sort_values(["dataset", "val_mse"])
    best_txt = args.output_dir / "latent_z_best_hparams.txt"

    grid_df.to_csv(grid_csv, index=False)
    payload = {
        "protocol": (
            "For each dataset, tune predictor learning_rate, hidden_dim, and predictor_epochs "
            "on z-only validation MSE using a held-out split of the training clips. Reuse the "
            "selected dataset-level params for all feature modes and encoder scenarios."
        ),
        "grid": {
            "scenario": args.scenario,
            "predictor_learning_rates": args.learning_rates,
            "hidden_dims": args.hidden_dims,
            "predictor_epochs": args.predictor_epochs,
            "latent_dim": args.latent_dim,
            "horizon": args.horizon,
            "latent_tda_window": args.latent_tda_window,
            "ae_epochs": args.ae_epochs,
            "encoder_learning_rate": args.encoder_learning_rate,
            "seeds": args.seeds,
            "val_fraction": args.val_fraction,
            "max_train_clips": args.max_train_clips,
            "max_val_clips": args.max_val_clips,
        },
        "best_by_dataset": {**existing_best, **best_by_dataset},
    }
    best_json.write_text(json.dumps(payload, indent=2) + "\n")

    best_df = pd.DataFrame(
        [{"dataset": dataset, **params} for dataset, params in payload["best_by_dataset"].items()]
    ).sort_values("dataset")
    best_txt.write_text(best_df.to_string(index=False, float_format=lambda value: f"{value:.6g}") + "\n")

    print("\nBest z-only hyperparameters:")
    print(best_df.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nSaved grid: {grid_csv}")
    print(f"Saved best JSON: {best_json}")
    print(f"Saved best text: {best_txt}")


if __name__ == "__main__":
    main()
