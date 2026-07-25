#!/usr/bin/env python3
"""Paired significance tests for topology vs. KPCA controls.

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


def _fmt_num(x: float, digits: int = 4) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def _fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{100.0 * x:.1f}\\%"


def _paired_stats(topology: np.ndarray, kpca: np.ndarray) -> dict[str, float]:
    diff = topology - kpca
    nonzero = np.abs(diff) > 1e-12
    if nonzero.sum() == 0:
        stat = 0.0
        p_less = 1.0
        p_two_sided = 1.0
    else:
        stat, p_less = wilcoxon(topology, kpca, alternative="less", zero_method="wilcox")
        _, p_two_sided = wilcoxon(topology, kpca, alternative="two-sided", zero_method="wilcox")
    rel_gain = (kpca - topology) / np.maximum(kpca, 1e-12)
    return {
        "n": int(len(diff)),
        "wins": int((diff < 0).sum()),
        "ties": int((np.abs(diff) <= 1e-12).sum()),
        "losses": int((diff > 0).sum()),
        "mean_topology_mse": float(np.mean(topology)),
        "mean_kpca_mse": float(np.mean(kpca)),
        "mean_relative_gain": float(np.mean(rel_gain)),
        "median_relative_gain": float(np.median(rel_gain)),
        "mean_abs_delta": float(np.mean(kpca - topology)),
        "wilcoxon_stat": float(stat),
        "p_less": float(p_less),
        "p_two_sided": float(p_two_sided),
    }


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["best_kpca_mean"] = df[["z_kpca_mean", "fuse_kpca_mean"]].min(axis=1)
    df["best_topology_mean"] = df[["fuse_h1_mean", "best_topo_mean"]].min(axis=1)
    df["topology_minus_kpca"] = df["best_topology_mean"] - df["best_kpca_mean"]
    df["topology_rel_gain"] = (df["best_kpca_mean"] - df["best_topology_mean"]) / df["best_kpca_mean"]
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])
    return df


def _write_latex(summary: pd.DataFrame, detail: pd.DataFrame, out_dir: Path) -> None:
    primary = summary[summary["level"] == "dataset_mean"].iloc[0]
    row_level = summary[summary["level"] == "dataset_encoder"].iloc[0]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Post-hoc paired comparison between the best observed topology descriptor and the best observed KPCA control. ``Best'' denotes the lowest test MSE among the evaluated variants for each dataset--encoder setting. The primary consistency check averages MSE across encoder families within each dataset, then applies a one-sided Wilcoxon signed-rank test over datasets. Lower MSE is better.}",
        r"\label{tab:topology-kpca-significance}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Level & $n$ & Wins & Avg. gain & Wilcoxon $p$ & Two-sided $p$ \\",
        r"\midrule",
        (
            "Dataset mean"
            f" & {int(primary.n)}"
            f" & {int(primary.wins)}/{int(primary.n)}"
            f" & {_fmt_pct(float(primary.mean_relative_gain))}"
            f" & {_fmt_num(float(primary.p_less), 4)}"
            f" & {_fmt_num(float(primary.p_two_sided), 4)} \\\\"
        ),
        (
            "Dataset--encoder"
            f" & {int(row_level.n)}"
            f" & {int(row_level.wins)}/{int(row_level.n)}"
            f" & {_fmt_pct(float(row_level.mean_relative_gain))}"
            f" & {_fmt_num(float(row_level.p_less), 4)}"
            f" & {_fmt_num(float(row_level.p_two_sided), 4)} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (out_dir / "final_table_topology_kpca_significance.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    detail_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Dataset-level paired comparison used for the primary topology-vs-KPCA Wilcoxon test. Values average over AE, GeoAE, and TopoAE rows. Lower MSE is better.}",
        r"\label{tab:topology-kpca-significance-by-dataset}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Dataset & Best topology MSE & Best KPCA MSE & Relative gain & Topology wins? \\",
        r"\midrule",
    ]
    for row in detail.itertuples(index=False):
        detail_lines.append(
            f"{row.dataset_label} & {_fmt_num(row.best_topology_mean, 4)} & {_fmt_num(row.best_kpca_mean, 4)} & "
            f"{_fmt_pct(row.topology_rel_gain)} & {'Yes' if row.topology_minus_kpca < 0 else 'No'} \\\\"
        )
    detail_lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    (out_dir / "final_table_topology_kpca_significance_by_dataset.txt").write_text(
        "\n".join(detail_lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("latex_tables/pca_kpca_control_summary.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("latex_tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _prepare(pd.read_csv(args.input))

    dataset_level = (
        rows.groupby(["dataset", "dataset_label"], sort=False)[["best_topology_mean", "best_kpca_mean"]]
        .mean()
        .reset_index()
    )
    order = {DATASET_LABELS.get(name, name): idx for idx, name in enumerate(DATASET_ORDER)}
    dataset_level["order"] = dataset_level["dataset_label"].map(order).fillna(len(order))
    dataset_level = dataset_level.sort_values("order").drop(columns="order")
    dataset_level["topology_minus_kpca"] = dataset_level["best_topology_mean"] - dataset_level["best_kpca_mean"]
    dataset_level["topology_rel_gain"] = (
        dataset_level["best_kpca_mean"] - dataset_level["best_topology_mean"]
    ) / dataset_level["best_kpca_mean"]

    summary_rows = []
    summary_rows.append({"level": "dataset_mean", **_paired_stats(dataset_level["best_topology_mean"].to_numpy(), dataset_level["best_kpca_mean"].to_numpy())})
    summary_rows.append({"level": "dataset_encoder", **_paired_stats(rows["best_topology_mean"].to_numpy(), rows["best_kpca_mean"].to_numpy())})
    summary = pd.DataFrame(summary_rows)

    rows.to_csv(args.out_dir / "topology_kpca_significance_rows.csv", index=False)
    dataset_level.to_csv(args.out_dir / "topology_kpca_significance_by_dataset.csv", index=False)
    summary.to_csv(args.out_dir / "topology_kpca_significance_summary.csv", index=False)
    _write_latex(summary, dataset_level, args.out_dir)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Wrote {args.out_dir / 'final_table_topology_kpca_significance.txt'}")
    print(f"Wrote {args.out_dir / 'final_table_topology_kpca_significance_by_dataset.txt'}")


if __name__ == "__main__":
    main()
