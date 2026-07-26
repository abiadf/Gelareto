from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results/ml_persistence")
OUTPUT_PATH = Path("latex_tables/final_table_latent_stability.txt")

SCENARIO_ENCODER = {
    "latent_stability": "AE",
    "geo_latent_stability": "GeoAE",
    "topo_latent_stability": "TopoAE",
}

DATASET_LABELS = {
    "bouncing_rings": "Bouncing rings",
    "bouncing_disks": "Bouncing disks",
    "orbiting_rings": "Orbiting rings",
    "orbiting_disks": "Orbiting disks",
    "moving_mnist": "Moving MNIST",
    "lorenz96": "Lorenz-96",
    "electric_devices": "ElectricDevices",
    "glioblastoma": "Glioblastoma",
    "hela": "HeLa",
}

DATASET_ORDER = list(DATASET_LABELS)
ENCODER_ORDER = ["AE", "GeoAE", "TopoAE"]


def _load_latest_results() -> pd.DataFrame:
    rows = []
    chosen: dict[tuple[str, str], tuple[float, Path, dict]] = {}
    for config_path in RESULTS_DIR.glob("run_*/run_config.json"):
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        cfg = payload.get("run_config", {})
        scenario = cfg.get("scenario")
        dataset = cfg.get("dataset")
        if scenario not in SCENARIO_ENCODER or dataset not in DATASET_LABELS:
            continue
        results_path = config_path.parent / "results.csv"
        if not results_path.exists():
            continue
        key = (dataset, scenario)
        mtime = results_path.stat().st_mtime
        if key not in chosen or mtime > chosen[key][0]:
            chosen[key] = (mtime, results_path, cfg)

    for (_, scenario), (_, results_path, cfg) in chosen.items():
        df = pd.read_csv(results_path)
        if df.empty:
            continue
        df["scenario"] = scenario
        df["encoder"] = SCENARIO_ENCODER[scenario]
        df["dataset"] = cfg["dataset"]
        rows.append(df)
    if not rows:
        raise ValueError(f"No stability results found in {RESULTS_DIR}")
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["h0_ratio"] = out["h0_bottleneck"] / out["latent_hausdorff"]
    out["h1_ratio"] = out["h1_bottleneck"] / out["latent_hausdorff"]
    return out


def _fmt(mean: float, std: float, digits: int = 3) -> str:
    return f"${mean:.{digits}f}{{\\pm}}{std:.{digits}f}$"


def make_table(df: pd.DataFrame, sigma: float = 0.10) -> str:
    subset = df[df["sigma"].round(6) == round(float(sigma), 6)].copy()
    if subset.empty:
        available = ", ".join(f"{x:.4f}" for x in sorted(df["sigma"].dropna().unique()))
        raise ValueError(f"No rows found for sigma={sigma}; available sigma values: {available}")

    grouped = (
        subset.groupby(["dataset", "encoder"], sort=False)
        .agg(
            n_runs=("seed", "nunique"),
            dh_mean=("latent_hausdorff", "mean"),
            dh_std=("latent_hausdorff", "std"),
            h0_mean=("h0_bottleneck", "mean"),
            h0_std=("h0_bottleneck", "std"),
            h0_ratio_mean=("h0_ratio", "mean"),
            h0_ratio_std=("h0_ratio", "std"),
            h1_mean=("h1_bottleneck", "mean"),
            h1_std=("h1_bottleneck", "std"),
            h1_ratio_mean=("h1_ratio", "mean"),
            h1_ratio_std=("h1_ratio", "std"),
        )
        .reset_index()
    )
    grouped["dataset_order"] = grouped["dataset"].map({name: i for i, name in enumerate(DATASET_ORDER)})
    grouped["encoder_order"] = grouped["encoder"].map({name: i for i, name in enumerate(ENCODER_ORDER)})
    grouped = grouped.sort_values(["dataset_order", "encoder_order"])

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Latent persistence stability diagnostic at input noise level $\sigma={sigma:.2f}$. We add Gaussian noise to the input, re-encode the clean and perturbed sequences, and compare latent-window point clouds and Vietoris--Rips persistence diagrams. $d_H$ is the Hausdorff distance between clean and perturbed latent point clouds, and $d_B(H_k)/d_H$ reports the relative diagram change. Values below one indicate that the persistence diagram changes less than the latent point cloud under the same perturbation.}}",
        r"\label{tab:latent_stability}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Dataset & Encoder & $d_H(z,z')$ & $d_B(H_0)$ & $d_B(H_0)/d_H$ & $d_B(H_1)$ & $d_B(H_1)/d_H$ \\",
        r"\midrule",
    ]
    previous_dataset = None
    for row in grouped.itertuples(index=False):
        if previous_dataset is not None and row.dataset != previous_dataset:
            lines.append(r"\midrule")
        dataset_label = DATASET_LABELS[row.dataset] if row.dataset != previous_dataset else ""
        lines.append(
            f"{dataset_label} & {row.encoder} & "
            f"{_fmt(row.dh_mean, row.dh_std)} & "
            f"{_fmt(row.h0_mean, row.h0_std)} & "
            f"{_fmt(row.h0_ratio_mean, row.h0_ratio_std)} & "
            f"{_fmt(row.h1_mean, row.h1_std)} & "
            f"{_fmt(row.h1_ratio_mean, row.h1_ratio_std)} "
            + r"\\"
        )
        previous_dataset = row.dataset
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    df = _load_latest_results()
    OUTPUT_PATH.write_text(make_table(df), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
