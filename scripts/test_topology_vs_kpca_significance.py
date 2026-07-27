#!/usr/bin/env python3
"""Paired significance tests for topology vs. all non-topological controls.

The primary test averages each method over encoder families within a dataset,
then runs a paired Wilcoxon signed-rank test over datasets. This avoids treating
the three encoder variants for the same dataset as independent observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


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

DATASET_ORDER = [
    "bouncing_disks",
    "bouncing_rings",
    "orbiting_disks",
    "orbiting_rings",
    "moving_mnist",
    "glioblastoma",
    "hela",
    "lorenz96",
    "electric_devices",
]

CONTROLS = {
    r"$z$ only": "z_mean",
    "PCA": "best_pca_mean",
    "KPCA": "best_kpca_mean",
    "Laplacian": "best_laplacian_mean",
    "Diffusion": "best_diffusion_mean",
    "RFF": "best_rff_mean",
}


def _fmt_num(x: float, digits: int = 4) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def _fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{100.0 * x:.1f}\\%"


def _paired_stats(topology: np.ndarray, control: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(topology) & np.isfinite(control)
    topology = topology[valid]
    control = control[valid]
    diff = topology - control
    nonzero = np.abs(diff) > 1e-12
    if nonzero.sum() == 0:
        stat = 0.0
        p_less = 1.0
        p_two_sided = 1.0
    else:
        stat, p_less = wilcoxon(topology, control, alternative="less", zero_method="wilcox")
        _, p_two_sided = wilcoxon(topology, control, alternative="two-sided", zero_method="wilcox")
    rel_gain = (control - topology) / np.maximum(control, 1e-12)
    return {
        "n": int(len(diff)),
        "wins": int((diff < 0).sum()),
        "ties": int((np.abs(diff) <= 1e-12).sum()),
        "losses": int((diff > 0).sum()),
        "mean_topology_mse": float(np.mean(topology)),
        "mean_control_mse": float(np.mean(control)),
        "mean_relative_gain": float(np.mean(rel_gain)),
        "median_relative_gain": float(np.median(rel_gain)),
        "mean_abs_delta": float(np.mean(control - topology)),
        "wilcoxon_stat": float(stat),
        "p_less": float(p_less),
        "p_two_sided": float(p_two_sided),
    }


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["best_pca_mean"] = df[["z_pca_mean", "fuse_pca_mean"]].min(axis=1)
    df["best_kpca_mean"] = df[["z_kpca_mean", "fuse_kpca_mean"]].min(axis=1)
    df["best_topology_mean"] = df[["fuse_h1_mean", "best_topo_mean"]].min(axis=1)
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])
    return df


def _holm_adjust(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, idx in enumerate(order):
        running = max(running, (count - rank) * values[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def _write_latex(summary: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Post-hoc paired comparisons between the best observed persistence descriptor and non-topological controls. ``Best'' denotes the lowest test MSE among the evaluated direct and fused variants within each dataset--encoder setting. The primary test averages MSE across encoder families within each dataset and applies a one-sided Wilcoxon signed-rank test over nine datasets; Holm-adjusted values correct across the six controls. Lower MSE is better.}",
        r"\label{tab:topology-control-significance}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Control & Dataset wins & Avg. gain & $p$ & Holm $p$ & Dataset--encoder wins & $p$ \\",
        r"\midrule",
    ]
    for control in CONTROLS:
        dataset_row = summary[(summary["control"] == control) & (summary["level"] == "dataset_mean")].iloc[0]
        encoder_row = summary[(summary["control"] == control) & (summary["level"] == "dataset_encoder")].iloc[0]
        lines.append(
            f"{control} & {int(dataset_row.wins)}/{int(dataset_row.n)} & "
            f"{_fmt_pct(float(dataset_row.mean_relative_gain))} & "
            f"{_fmt_num(float(dataset_row.p_less), 4)} & "
            f"{_fmt_num(float(dataset_row.p_holm), 4)} & "
            f"{int(encoder_row.wins)}/{int(encoder_row.n)} & "
            f"{_fmt_num(float(encoder_row.p_less), 4)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    text = "\n".join(lines) + "\n"
    (out_dir / "final_table_topology_control_significance.txt").write_text(text, encoding="utf-8")
    (out_dir / "final_table_topology_kpca_significance.txt").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("latex_tables/pca_kpca_control_summary.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("latex_tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _prepare(pd.read_csv(args.input))

    columns = ["best_topology_mean", *CONTROLS.values()]
    dataset_level = rows.groupby(["dataset", "dataset_label"], sort=False)[columns].mean().reset_index()
    order = {DATASET_LABELS.get(name, name): idx for idx, name in enumerate(DATASET_ORDER)}
    dataset_level["order"] = dataset_level["dataset_label"].map(order).fillna(len(order))
    dataset_level = dataset_level.sort_values("order").drop(columns="order")
    summary_rows = []
    detail_rows = []
    for control, column in CONTROLS.items():
        summary_rows.append(
            {
                "control": control,
                "control_column": column,
                "level": "dataset_mean",
                **_paired_stats(dataset_level["best_topology_mean"].to_numpy(), dataset_level[column].to_numpy()),
            }
        )
        summary_rows.append(
            {
                "control": control,
                "control_column": column,
                "level": "dataset_encoder",
                **_paired_stats(rows["best_topology_mean"].to_numpy(), rows[column].to_numpy()),
            }
        )
        for item in dataset_level.itertuples(index=False):
            topology = float(item.best_topology_mean)
            control_mse = float(getattr(item, column))
            detail_rows.append(
                {
                    "dataset": item.dataset,
                    "dataset_label": item.dataset_label,
                    "control": control,
                    "best_topology_mean": topology,
                    "control_mean": control_mse,
                    "relative_gain": (control_mse - topology) / max(control_mse, 1e-12),
                    "topology_wins": topology < control_mse,
                }
            )
    summary = pd.DataFrame(summary_rows)
    for level in summary["level"].unique():
        mask = summary["level"] == level
        summary.loc[mask, "p_holm"] = _holm_adjust(summary.loc[mask, "p_less"])
    detail = pd.DataFrame(detail_rows)

    rows.to_csv(args.out_dir / "topology_control_significance_rows.csv", index=False)
    detail.to_csv(args.out_dir / "topology_control_significance_by_dataset.csv", index=False)
    summary.to_csv(args.out_dir / "topology_control_significance_summary.csv", index=False)
    _write_latex(summary, args.out_dir)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Wrote {args.out_dir / 'final_table_topology_control_significance.txt'}")


if __name__ == "__main__":
    main()
