#!/usr/bin/env python3
"""Probe whether topology features are linearly recoverable from frozen latents.

This diagnostic fits z -> topology on cached latent-TDA payloads. It is meant as
an information-redundancy check: low held-out R^2 means the topology descriptor
is not a trivial linear readout of z.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCENARIO_LABELS = {
    "latent_tda": "AE",
    "geo_latent_tda": "GeoAE",
    "topo_latent_tda": "TopoAE",
}

DATASET_LABELS = {
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

DEFAULT_DATASETS = [
    "bouncing_disks",
    "bouncing_rings",
    "orbiting_disks",
    "orbiting_rings",
    "moving_mnist",
    "lorenz96",
    "electric_devices",
    "glioblastoma",
    "hela",
]

DEFAULT_SCENARIOS = ["latent_tda", "geo_latent_tda", "topo_latent_tda"]

TARGET_LABELS = {
    "h0": r"H_0",
    "h1": r"H_1",
    "both": r"H_0+H_1",
}

PROBE_LABELS = {
    "ridge": "ridge",
    "mlp": "MLP",
}


def _namespace(dataset: str, scenario: str, geo_lambda: float, topo_lambda: float, topo_distance: str) -> str:
    if scenario == "latent_tda":
        return dataset
    if scenario == "geo_latent_tda":
        return f"{dataset}_geoae_lam{geo_lambda:g}"
    if scenario == "topo_latent_tda":
        return f"{dataset}_topoae_lam{topo_lambda:g}_{topo_distance}"
    raise ValueError(f"Unsupported scenario: {scenario}")


def _safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _payload_path(namespace: str, seed: int, split: str, window: int, bins: int) -> Path | None:
    feature_dir = Path("models") / namespace / "latent_tda_features"
    candidates = list(feature_dir.glob(f"seed{seed}_{split}_T*_B*_H*_W*_latent*_win{window}_bins{bins}.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _target_tensor(payload: dict, target: str) -> torch.Tensor:
    if target == "h0":
        return payload["h0"]
    if target == "h1":
        return payload["h1"]
    if target == "both":
        return torch.cat([payload["h0"], payload["h1"]], dim=-1)
    raise ValueError(f"Unknown target: {target}")


def _flatten_payload(payload: dict, target: str, min_t: int) -> tuple[np.ndarray, np.ndarray]:
    z = payload["z"][min_t:]
    y = _target_tensor(payload, target)[min_t:]
    return _flatten_tensors(z, y)


def _flatten_tensors(z: torch.Tensor, y: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    return (
        z.reshape(-1, z.shape[-1]).detach().cpu().numpy().astype(np.float32),
        y.reshape(-1, y.shape[-1]).detach().cpu().numpy().astype(np.float32),
    )


def _fit_probe(x_train: np.ndarray, y_train: np.ndarray, args: argparse.Namespace, split_seed: int):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train)
    y_train_s = y_scaler.fit_transform(y_train)
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(len(x_train_s))
    n_val = max(1, int(round(args.val_fraction * len(order))))
    val_idx = order[:n_val]
    fit_idx = order[n_val:]
    if len(fit_idx) == 0:
        fit_idx = order
        val_idx = order

    if args.model == "ridge":
        best_model = None
        best_score = -np.inf
        for alpha in args.alphas:
            model = Ridge(alpha=float(alpha), random_state=split_seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LinAlgWarning)
                model.fit(x_train_s[fit_idx], y_train_s[fit_idx])
            pred_val = model.predict(x_train_s[val_idx])
            score = r2_score(y_train_s[val_idx], pred_val, multioutput="variance_weighted")
            if score > best_score:
                best_score = score
                best_model = model
        return best_model, x_scaler, y_scaler

    model = MLPRegressor(
        hidden_layer_sizes=(args.mlp_hidden,),
        activation="relu",
        alpha=args.mlp_alpha,
        learning_rate_init=args.mlp_lr,
        max_iter=args.mlp_max_iter,
        early_stopping=True,
        validation_fraction=args.val_fraction,
        random_state=split_seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", DataConversionWarning)
        model.fit(x_train_s, y_train_s)
    return model, x_scaler, y_scaler


def _run_one(dataset: str, scenario: str, seed: int, args: argparse.Namespace) -> dict | None:
    namespace = _namespace(dataset, scenario, args.geo_lambda, args.topo_lambda, args.topo_distance)
    train_path = _payload_path(namespace, seed, "train", args.window, args.bins)
    test_path = _payload_path(namespace, seed, "test", args.window, args.bins)
    if train_path is None or test_path is None:
        if not args.quiet:
            print(f"Skipping missing payload: dataset={dataset} scenario={scenario} seed={seed}")
        return None

    train_payload = _safe_load(train_path)
    test_payload = _safe_load(test_path)
    min_t = max(0, int(args.min_t))
    if args.split_mode == "sequence_random":
        z_all = torch.cat([train_payload["z"][min_t:], test_payload["z"][min_t:]], dim=1)
        y_all = torch.cat([_target_tensor(train_payload, args.target)[min_t:], _target_tensor(test_payload, args.target)[min_t:]], dim=1)
        rng = np.random.default_rng(args.seed_split + seed)
        order = rng.permutation(z_all.shape[1])
        n_test = max(1, int(round(args.test_fraction * len(order))))
        test_idx = torch.as_tensor(order[:n_test], dtype=torch.long)
        train_idx = torch.as_tensor(order[n_test:], dtype=torch.long)
        x_train, y_train = _flatten_tensors(z_all[:, train_idx], y_all[:, train_idx])
        x_test, y_test = _flatten_tensors(z_all[:, test_idx], y_all[:, test_idx])
    else:
        x_train, y_train = _flatten_payload(train_payload, args.target, min_t=min_t)
        x_test, y_test = _flatten_payload(test_payload, args.target, min_t=min_t)
    if args.split_mode == "random":
        x_all = np.concatenate([x_train, x_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        rng = np.random.default_rng(args.seed_split + seed)
        order = rng.permutation(len(x_all))
        n_test = max(2, int(round(args.test_fraction * len(order))))
        test_idx = order[:n_test]
        train_idx = order[n_test:]
        x_train, y_train = x_all[train_idx], y_all[train_idx]
        x_test, y_test = x_all[test_idx], y_all[test_idx]
    if len(x_train) < 4 or len(x_test) < 2:
        return None
    target_std = y_train.std(axis=0)
    active_dims = target_std > float(args.target_std_threshold)
    if not np.any(active_dims):
        if not args.quiet:
            print(f"Skipping near-constant target: dataset={dataset} scenario={scenario} seed={seed}")
        return None
    y_train = y_train[:, active_dims]
    y_test = y_test[:, active_dims]

    model, x_scaler, y_scaler = _fit_probe(x_train, y_train, args, split_seed=args.seed_split + seed)
    pred_test_s = model.predict(x_scaler.transform(x_test))
    if pred_test_s.ndim == 1:
        pred_test_s = pred_test_s.reshape(-1, 1)
    pred_test = y_scaler.inverse_transform(pred_test_s)
    r2_uniform = r2_score(y_test, pred_test, multioutput="uniform_average")
    r2_var = r2_score(y_test, pred_test, multioutput="variance_weighted")
    mse = float(np.mean((pred_test - y_test) ** 2))
    return {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS.get(dataset, dataset),
        "scenario": scenario,
        "encoder": SCENARIO_LABELS[scenario],
        "seed": seed,
        "target": args.target,
        "model": args.model,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "split_mode": args.split_mode,
        "z_dim": int(x_train.shape[-1]),
        "target_dim": int(y_train.shape[-1]),
        "target_dim_total": int(len(active_dims)),
        "target_std_threshold": float(args.target_std_threshold),
        "r2_uniform": float(r2_uniform),
        "r2_variance_weighted": float(r2_var),
        "mse": mse,
        "train_payload": str(train_path),
        "test_payload": str(test_path),
    }


def _fmt(mean: float, std: float) -> str:
    return f"${mean:.3f}{{\\pm}}{std:.3f}$"


def _fmt_iqr(median: float, q25: float, q75: float) -> str:
    return f"${median:.3f}\\;[{q25:.3f},{q75:.3f}]$"


def _write_latex_tables(results: pd.DataFrame, out_dir: Path, target: str, model: str, output_prefix: str) -> None:
    target_label = TARGET_LABELS.get(target, target.upper())
    model_label = PROBE_LABELS.get(model, model)
    table_label = output_prefix.replace("_", "-")
    encoder_summary = (
        results.groupby("encoder", sort=False)
        .agg(
            r2_median=("r2_variance_weighted", "median"),
            r2_q25=("r2_variance_weighted", lambda x: x.quantile(0.25)),
            r2_q75=("r2_variance_weighted", lambda x: x.quantile(0.75)),
            r2_mean=("r2_variance_weighted", "mean"),
            r2_std=("r2_variance_weighted", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            n_runs=("r2_variance_weighted", "count"),
        )
        .reset_index()
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Probe test for topology-feature redundancy. A {model_label} probe is trained to predict the cached ${target_label}$ topology vector from frozen $z$ and evaluated on held-out examples. Near-zero values indicate that the topology vector is not reliably recoverable from $z$ by this probe. We report median $R^2$ with interquartile range.}}",
        rf"\label{{tab:{table_label}}}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Encoder & $R^2(z\rightarrow B)$ \\",
        r"\midrule",
    ]
    for row in encoder_summary.itertuples(index=False):
        lines.append(f"{row.encoder} & {_fmt_iqr(row.r2_median, row.r2_q25, row.r2_q75)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (out_dir / f"final_table_{output_prefix}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    detail = (
        results.groupby(["dataset_label", "encoder"], sort=False)
        .agg(r2_mean=("r2_variance_weighted", "mean"), r2_std=("r2_variance_weighted", "std"))
        .reset_index()
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Per-dataset $z\rightarrow {target_label}$ topology-probe $R^2$ on held-out examples. Values are mean $\pm$ standard deviation over seeds.}}",
        rf"\label{{tab:{table_label}-by-dataset}}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & AE & GeoAE & TopoAE \\",
        r"\midrule",
    ]
    pivot = detail.pivot(index="dataset_label", columns="encoder", values=["r2_mean", "r2_std"])
    for dataset in [DATASET_LABELS.get(d, d) for d in DEFAULT_DATASETS]:
        if dataset not in pivot.index:
            continue
        cells = []
        for encoder in ["AE", "GeoAE", "TopoAE"]:
            try:
                cells.append(_fmt(float(pivot.loc[dataset, ("r2_mean", encoder)]), float(pivot.loc[dataset, ("r2_std", encoder)])))
            except Exception:
                cells.append("--")
        lines.append(f"{dataset} & {' & '.join(cells)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    (out_dir / f"final_table_{output_prefix}_by_dataset.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_combined_latex_table(result_paths: list[Path], out_dir: Path, output_prefix: str) -> None:
    frames = [pd.read_csv(path) for path in result_paths]
    results = pd.concat(frames, ignore_index=True)
    if len(results["target"].dropna()):
        target = str(results["target"].dropna().iloc[0])
    else:
        target = "h1"
    target_label = TARGET_LABELS.get(target, target.upper())
    summary = (
        results.groupby(["model", "encoder"], sort=False)
        .agg(
            r2_median=("r2_variance_weighted", "median"),
            r2_q25=("r2_variance_weighted", lambda x: x.quantile(0.25)),
            r2_q75=("r2_variance_weighted", lambda x: x.quantile(0.75)),
            n_runs=("r2_variance_weighted", "count"),
        )
        .reset_index()
    )
    model_order = [model for model in ["ridge", "mlp"] if model in set(summary["model"])]
    encoder_order = ["AE", "GeoAE", "TopoAE"]
    table_label = output_prefix.replace("_", "-")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Probe test for topology-feature redundancy. Ridge and MLP probes are trained to predict the cached ${target_label}$ topology vector from frozen $z$ and evaluated on held-out examples. Values are median $R^2$ with interquartile range; values near zero indicate that the topology descriptor is not reliably recoverable from $z$ by the corresponding probe.}}",
        rf"\label{{tab:{table_label}}}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccc ccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Linear ridge probe} & \multicolumn{3}{c}{Nonlinear MLP probe} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r" & AE & GeoAE & TopoAE & AE & GeoAE & TopoAE \\",
        r"\midrule",
    ]
    cells = []
    for model in model_order:
        for encoder in encoder_order:
            row = summary[(summary["model"] == model) & (summary["encoder"] == encoder)]
            if row.empty:
                cells.append("--")
            else:
                item = row.iloc[0]
                cells.append(_fmt_iqr(float(item.r2_median), float(item.r2_q25), float(item.r2_q75)))
    lines.append(rf"${target_label}$ & {' & '.join(cells)} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    (out_dir / f"final_table_{output_prefix}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    detail = (
        results.groupby(["dataset_label", "model", "encoder"], sort=False)
        .agg(r2_mean=("r2_variance_weighted", "mean"), r2_std=("r2_variance_weighted", "std"))
        .reset_index()
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Per-dataset $z\rightarrow {target_label}$ topology-probe $R^2$ on held-out examples. Values are mean $\pm$ standard deviation over seeds for both the linear ridge and nonlinear MLP probes.}}",
        rf"\label{{tab:{table_label}-by-dataset}}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccc ccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Linear ridge probe} & \multicolumn{3}{c}{Nonlinear MLP probe} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Dataset & AE & GeoAE & TopoAE & AE & GeoAE & TopoAE \\",
        r"\midrule",
    ]
    pivot = detail.pivot(index="dataset_label", columns=["model", "encoder"], values=["r2_mean", "r2_std"])
    for dataset in [DATASET_LABELS.get(d, d) for d in DEFAULT_DATASETS]:
        if dataset not in pivot.index:
            continue
        cells = []
        for model in model_order:
            for encoder in encoder_order:
                try:
                    cells.append(_fmt(float(pivot.loc[dataset, ("r2_mean", model, encoder)]), float(pivot.loc[dataset, ("r2_std", model, encoder)])))
                except Exception:
                    cells.append("--")
        lines.append(f"{dataset} & {' & '.join(cells)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    (out_dir / f"final_table_{output_prefix}_by_dataset.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated dataset names.")
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS), help="Comma-separated scenarios.")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds.")
    parser.add_argument("--target", default="h1", choices=["h0", "h1", "both"])
    parser.add_argument("--model", default="ridge", choices=["ridge", "mlp"])
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--min-t", type=int, default=14, help="Drop early windows before this source time.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-mode",
        choices=["sequence_random", "random", "existing"],
        default="sequence_random",
        help="Use a random held-out sequence split, a random example split, or the existing train/test split.",
    )
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    parser.add_argument(
        "--target-std-threshold",
        type=float,
        default=1e-3,
        help="Drop topology-vector dimensions with train-set standard deviation at or below this threshold.",
    )
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--mlp-alpha", type=float, default=1e-4)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-max-iter", type=int, default=300)
    parser.add_argument("--seed-split", type=int, default=0)
    parser.add_argument("--geo-lambda", type=float, default=0.1)
    parser.add_argument("--topo-lambda", type=float, default=0.1)
    parser.add_argument("--topo-distance", default="signature")
    parser.add_argument("--out-dir", type=Path, default=Path("latex_tables"))
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for generated CSV/table files. Defaults to z_to_topology_probe_<model>_<target>_<split-mode>.",
    )
    parser.add_argument(
        "--from-results",
        type=Path,
        default=None,
        help="Regenerate summary/table files from an existing probe results CSV without reloading cached tensors.",
    )
    parser.add_argument(
        "--combine-results",
        type=Path,
        nargs="+",
        default=None,
        help="Write one combined LaTeX summary table from multiple probe result CSVs.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _make_summary(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["encoder", "target", "model"], sort=False)
        .agg(
            r2_median=("r2_variance_weighted", "median"),
            r2_q25=("r2_variance_weighted", lambda x: x.quantile(0.25)),
            r2_q75=("r2_variance_weighted", lambda x: x.quantile(0.75)),
            r2_mean=("r2_variance_weighted", "mean"),
            r2_std=("r2_variance_weighted", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            n_runs=("r2_variance_weighted", "count"),
        )
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix or f"z_to_topology_probe_{args.model}_{args.target}_{args.split_mode}"
    results_path = args.out_dir / f"{output_prefix}_results.csv"
    summary_path = args.out_dir / f"{output_prefix}_summary.csv"

    if args.combine_results is not None:
        _write_combined_latex_table(args.combine_results, args.out_dir, output_prefix)
        print(f"Wrote {args.out_dir / f'final_table_{output_prefix}.txt'}")
        return

    if args.from_results is not None:
        results = pd.read_csv(args.from_results)
        if "target" in results and len(results["target"].dropna()):
            args.target = str(results["target"].dropna().iloc[0])
        if "model" in results and len(results["model"].dropna()):
            args.model = str(results["model"].dropna().iloc[0])
    else:
        datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
        scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
        seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
        rows = []
        for dataset in datasets:
            for scenario in scenarios:
                for seed in seeds:
                    row = _run_one(dataset, scenario, seed, args)
                    if row is not None:
                        rows.append(row)
        if not rows:
            raise SystemExit("No probe rows produced. Check cached latent-TDA payloads.")
        results = pd.DataFrame(rows)
        results.to_csv(results_path, index=False)

    summary = _make_summary(results)
    summary.to_csv(summary_path, index=False)
    _write_latex_tables(results, args.out_dir, args.target, args.model, output_prefix)
    if args.from_results is None:
        print(f"Wrote {results_path}")
    else:
        print(f"Read {args.from_results}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.out_dir / f'final_table_{output_prefix}.txt'}")
    print(f"Wrote {args.out_dir / f'final_table_{output_prefix}_by_dataset.txt'}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
