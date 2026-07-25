#!/usr/bin/env python3
"""Fixed-method Wilcoxon tests for topology descriptors vs KPCA controls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from make_tuned_result_tables import DATASET_ORDER, SCENARIO_ORDER, parse_rows, row_map


INPUT_PATH = Path("latex_tables/hyperparam_tuned.txt")
OUT_DIR = Path("latex_tables")

PAIRS = [
    (r"Betti $H_1$", "z_h1", r"KPCA $H_1$", "z_kpca_h1"),
    (r"PI $H_1$", "z_pi_h1", r"KPCA $H_1$", "z_kpca_h1"),
    (r"Landscape $H_1$", "z_landscape_h1", r"KPCA $H_1$", "z_kpca_h1"),
    (r"PersLay $H_1$", "z_perslay_h1", r"KPCA $H_1$", "z_kpca_h1"),
    (r"Fused Betti $H_1$", "z_fuse_h1", r"Fused KPCA $H_1$", "z_fuse_kpca_h1"),
]


def _fmt_num(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{100.0 * value:.1f}\\%"


def _paired_stats(topology: np.ndarray, control: np.ndarray) -> dict[str, float]:
    diff = topology - control
    if np.all(np.abs(diff) <= 1e-12):
        p_less = 1.0
        p_two_sided = 1.0
    else:
        p_less = float(wilcoxon(topology, control, alternative="less", zero_method="wilcox").pvalue)
        p_two_sided = float(wilcoxon(topology, control, alternative="two-sided", zero_method="wilcox").pvalue)
    rel_gain = (control - topology) / np.maximum(control, 1e-12)
    return {
        "n": float(len(topology)),
        "wins": float((diff < 0).sum()),
        "mean_topology_mse": float(np.mean(topology)),
        "mean_control_mse": float(np.mean(control)),
        "avg_gain": float(np.mean(rel_gain)),
        "p_less": p_less,
        "p_two_sided": p_two_sided,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = row_map(parse_rows(INPUT_PATH))
    summary_rows = []
    scopes = [
        ("Encoder-avg.", SCENARIO_ORDER),
        ("GeoAE only", ["geo_latent_tda"]),
    ]
    for scope_label, scenarios in scopes:
        for topology_label, topology_mode, control_label, control_mode in PAIRS:
            paired = []
            for dataset in DATASET_ORDER:
                topology_vals = []
                control_vals = []
                for scenario in scenarios:
                    top_row = rows.get((dataset, scenario, topology_mode))
                    control_row = rows.get((dataset, scenario, control_mode))
                    if top_row is not None and control_row is not None:
                        topology_vals.append(top_row.mse_mean)
                        control_vals.append(control_row.mse_mean)
                if topology_vals:
                    paired.append((float(np.mean(topology_vals)), float(np.mean(control_vals))))
            topology = np.asarray([item[0] for item in paired], dtype=float)
            control = np.asarray([item[1] for item in paired], dtype=float)
            stats = _paired_stats(topology, control)
            summary_rows.append(
                {
                    "scope": scope_label,
                    "topology": topology_label,
                    "topology_mode": topology_mode,
                    "control": control_label,
                    "control_mode": control_mode,
                    **stats,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "fixed_topology_kpca_significance_summary.csv", index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Fixed-method paired Wilcoxon tests comparing pre-specified topology descriptors against KPCA controls. Encoder-avg. rows first average MSE over AE, GeoAE, and TopoAE within each dataset; GeoAE-only rows test the best-performing encoder family directly. A one-sided Wilcoxon signed-rank test is applied over the nine datasets with alternative \( \mathrm{MSE}_{\mathrm{topo}} < \mathrm{MSE}_{\mathrm{KPCA}} \). No best-of-test selection is used. Lower MSE is better.}",
        r"\label{tab:fixed-topology-kpca-significance}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Scope & Topology descriptor & Wins & Avg. gain & Wilcoxon \(p\) & Two-sided \(p\) \\",
        r"\midrule",
    ]
    previous_scope = None
    for row in summary.itertuples(index=False):
        if previous_scope is not None and row.scope != previous_scope:
            lines.append(r"\midrule")
        lines.append(
            f"{row.scope} & {row.topology} vs. {row.control} & {int(row.wins)}/{int(row.n)} & "
            f"{_fmt_pct(row.avg_gain)} & {_fmt_num(row.p_less)} & {_fmt_num(row.p_two_sided)} \\\\"
        )
        previous_scope = row.scope
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    (OUT_DIR / "final_table_fixed_topology_kpca_significance.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Wrote {OUT_DIR / 'final_table_fixed_topology_kpca_significance.txt'}")


if __name__ == "__main__":
    main()
