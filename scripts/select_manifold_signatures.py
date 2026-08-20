"""Select one mixed-curvature signature per dataset without using test data.

The triangle diagnostic runs on a fitting subset of the original training
clips, proposes Euclidean-softened H/S/E candidates, and evaluates them on a
held-out validation subset. The lowest mean z+B(H0,H1) validation MSE is saved
for the final test runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tune_latent_z_hparams import parse_csv_floats, parse_csv_ints, parse_csv_strs, selection_folds
from topo import ml_runner


DEFAULT_DATASETS = (
    "bouncing_rings,bouncing_disks,orbiting_rings,orbiting_disks,"
    "moving_mnist,lorenz96,electric_devices,glioblastoma,hela,growing_tree,hirros"
)


def make_cfg(dataset: str, args: argparse.Namespace) -> ml_runner.RunConfig:
    argv = [
        "--scenario", "manif_geo_latent_tda",
        "--dataset", dataset,
        "--device", args.device,
        "--modes", "z_both",
        "--seeds", ",".join(map(str, args.seeds)),
        "--latent-dim", str(args.latent_dim),
        "--horizon", str(args.horizon),
        "--latent-tda-window", str(args.window),
        "--latent-tda-bins", str(args.bins),
        "--ae-epochs", str(args.ae_epochs),
        "--geo-ae-epochs", str(args.geo_ae_epochs),
        "--geo-ae-lambda", str(args.geo_ae_lambda),
        "--learning-rate", str(args.encoder_learning_rate),
        "--manifold-signature-policy", "triangle_shrinkage",
        "--signature-shrinkages", ",".join(map(str, args.shrinkages)),
        "--triangle-knn", ",".join(map(str, args.triangle_knn)),
        "--triangle-samples", str(args.triangle_samples),
        "--triangle-max-points", str(args.triangle_max_points),
        "--vr-distance", "product_manifold",
        "--hparam-file", str(args.hparam_file),
        "--output-dir", str(args.output_dir / "_internal"),
        "--no-save",
    ]
    cfg = ml_runner._apply_tuned_hparams(ml_runner.parse_args(argv))
    if dataset in args.candidate_signatures:
        cfg = replace(
            cfg,
            manifold_signature_policy="manual",
            manifold_signatures=args.candidate_signatures[dataset],
        )
    return replace(
        cfg,
        retrain_encoder=True,
        retrain_predictor=True,
        recompute_latent_tda_features=True,
        latent_tda_max_train=None,
        latent_tda_max_test=None,
    )


def select_dataset(dataset: str, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    cfg = make_cfg(dataset, args)
    base = ml_runner.load_video_context(cfg)
    folds = selection_folds(base, val_fraction=args.val_fraction, max_train=None, max_val=None)
    signatures = ml_runner._resolved_manifold_signatures(cfg, base)
    result_frames = []
    for signature in signatures:
        for fold_name, train, validation in folds:
            fold_cfg = replace(
                cfg,
                dataset=f"{dataset}_selection_{fold_name}",
                manifold_signature_file=None,
                manifold_signature_policy="manual",
                manifold_signatures=[signature],
            )
            context = replace(base, x_train=train, x_test=validation)
            results, _ = ml_runner.run_geo_latent_tda(
                fold_cfg, context, encoder_kind="manifold_mixedgeo"
            )
            results["validation_fold"] = fold_name
            results["manifold_signature"] = signature
            result_frames.append(results)
    results = pd.concat(result_frames, ignore_index=True)
    candidates = (
        results.groupby("manifold_signature", as_index=False)
        .agg(
            val_mse=("test_mse", "mean"),
            val_mse_std=("test_mse", "std"),
            n_runs=("test_mse", "size"),
        )
        .sort_values("val_mse")
    )
    candidates.insert(0, "dataset", dataset)
    best = candidates.iloc[0]
    selected = {
        "manifold_signature": str(best["manifold_signature"]),
        "val_mse": float(best["val_mse"]),
        "val_mse_std": float(best["val_mse_std"]),
        "n_runs": int(best["n_runs"]),
        "selection": (
            "skeleton-triangle candidates; lowest z_both validation MSE"
            if dataset in args.candidate_signatures
            else "training-only triangle candidates; lowest z_both validation MSE"
        ),
    }
    print(f"\nSelected {dataset}: {selected}")
    return candidates, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seeds", type=parse_csv_ints, default=[0, 1, 2])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--ae-epochs", type=int, default=10)
    parser.add_argument("--geo-ae-epochs", type=int, default=10)
    parser.add_argument("--geo-ae-lambda", type=float, default=0.1)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-3)
    parser.add_argument("--shrinkages", type=parse_csv_floats, default=[0.0, 0.5, 0.75, 1.0])
    parser.add_argument("--triangle-knn", type=parse_csv_ints, default=[4, 8, 12])
    parser.add_argument("--triangle-samples", type=int, default=10_000)
    parser.add_argument("--triangle-max-points", type=int, default=512)
    parser.add_argument("--hparam-file", type=Path, default=Path("results/hparam_search/latent_z_best_hparams.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/signature_selection"))
    parser.add_argument(
        "--candidate-signatures-json",
        type=Path,
        help="Optional dataset-to-signature-list JSON, e.g. skeleton-derived video candidates.",
    )
    args = parser.parse_args()
    args.datasets = parse_csv_strs(args.datasets)
    args.candidate_signatures = (
        json.loads(args.candidate_signatures_json.read_text())
        if args.candidate_signatures_json else {}
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    selected_path = args.output_dir / "selected_manifold_signatures.json"
    candidate_path = args.output_dir / "signature_validation_results.csv"
    old_payload = json.loads(selected_path.read_text()) if selected_path.exists() else {}
    selected_by_dataset = dict(old_payload.get("selected_by_dataset", {}))
    old_candidates = pd.read_csv(candidate_path) if candidate_path.exists() else pd.DataFrame()

    frames = []
    for dataset in args.datasets:
        candidates, selected = select_dataset(dataset, args)
        frames.append(candidates)
        selected_by_dataset[dataset] = selected

    if not old_candidates.empty:
        old_candidates = old_candidates[~old_candidates["dataset"].isin(args.datasets)]
        frames.insert(0, old_candidates)
    pd.concat(frames, ignore_index=True).to_csv(candidate_path, index=False, float_format="%.6g")
    selected_path.write_text(
        json.dumps(
            {
                "protocol": "Training-only triangle candidates; grouped source-sequence validation when available; select lowest mean z_both validation MSE using product-manifold VR.",
                "seeds": args.seeds,
                "selected_by_dataset": selected_by_dataset,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nSaved candidates: {candidate_path}")
    print(f"Saved selections: {selected_path}")


if __name__ == "__main__":
    main()
