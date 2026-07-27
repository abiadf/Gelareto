#!/usr/bin/env python3
"""Descriptor-level paired tests against the strongest non-topological control."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from make_tuned_result_tables import DATASET_ORDER, SCENARIO_ORDER, parse_rows, row_map


INPUT_PATH = Path("latex_tables/raw_appendix_a_summary.txt")
OUT_DIR = Path("latex_tables")

DESCRIPTORS = {
    r"Betti $H_1$": ["z_h1", "z_fuse_h1"],
    r"PI $H_1$": ["z_pi_h1", "z_fuse_pi_h1"],
    r"Landscape $H_1$": ["z_landscape_h1", "z_fuse_landscape_h1"],
    r"PersLay $H_1$": ["z_perslay_h1", "z_fuse_perslay_h1"],
}

CONTROLS = {
    "PCA": ["z_fuse_pca_h1"],
    "KPCA": ["z_fuse_kpca_h1"],
    "Laplacian": ["z_laplacian_h1", "z_fuse_laplacian_h1"],
    "Diffusion": ["z_diffusion_h1", "z_fuse_diffusion_h1"],
    "RFF": ["z_rff_h1", "z_fuse_rff_h1"],
}


def _best_mse(rows, dataset: str, scenario: str, modes: list[str]) -> float | None:
    values = [
        rows[(dataset, scenario, mode)].mse_mean
        for mode in modes
        if (dataset, scenario, mode) in rows
    ]
    return min(values) if values else None


def _fmt_p(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    rows = row_map(parse_rows(INPUT_PATH))
    summary_rows = []
    detail_rows = []
    for label, modes in DESCRIPTORS.items():
        for control_label, control_modes in CONTROLS.items():
            setting_pairs = []
            for dataset in DATASET_ORDER:
                for scenario in SCENARIO_ORDER:
                    topology = _best_mse(rows, dataset, scenario, modes)
                    control = _best_mse(rows, dataset, scenario, control_modes)
                    if topology is not None and control is not None:
                        setting_pairs.append((dataset, scenario, topology, control))

            dataset_pairs = []
            for dataset in DATASET_ORDER:
                matches = [pair for pair in setting_pairs if pair[0] == dataset]
                if matches:
                    dataset_pairs.append(
                        (
                            dataset,
                            float(np.mean([pair[2] for pair in matches])),
                            float(np.mean([pair[3] for pair in matches])),
                        )
                    )
            topology = np.asarray([pair[1] for pair in dataset_pairs], dtype=float)
            control = np.asarray([pair[2] for pair in dataset_pairs], dtype=float)
            p_less = float(wilcoxon(topology, control, alternative="less", zero_method="wilcox").pvalue)
            p_two_sided = float(wilcoxon(topology, control, alternative="two-sided", zero_method="wilcox").pvalue)
            relative_gain = (control - topology) / np.maximum(control, 1e-12)
            summary_rows.append(
                {
                    "topology_descriptor": label,
                    "control": control_label,
                    "dataset_encoder_wins": int(sum(pair[2] < pair[3] for pair in setting_pairs)),
                    "dataset_encoder_n": len(setting_pairs),
                    "dataset_wins": int(np.sum(topology < control)),
                    "dataset_n": len(dataset_pairs),
                    "avg_gain": float(np.mean(relative_gain)),
                    "p_less": p_less,
                    "p_two_sided": p_two_sided,
                }
            )
            for dataset, topology_mse, control_mse in dataset_pairs:
                detail_rows.append(
                    {
                        "topology_descriptor": label,
                        "control": control_label,
                        "dataset": dataset,
                        "topology_mse": topology_mse,
                        "control_mse": control_mse,
                        "relative_gain": (control_mse - topology_mse) / max(control_mse, 1e-12),
                        "topology_wins": topology_mse < control_mse,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)
    summary.to_csv(OUT_DIR / "topology_descriptor_significance_summary.csv", index=False)
    detail.to_csv(OUT_DIR / "topology_descriptor_significance_by_dataset.csv", index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Fixed paired comparisons of persistence descriptors with non-topological controls. Wins are reported over all 27 dataset--encoder settings, whereas statistical tests use nine independent dataset averages to avoid treating encoder variants as independent samples. Lower MSE is better.}",
        r"\label{tab:topology-descriptor-significance}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Topology descriptor & Wins & Avg. gain & Wilcoxon \(p\) & Two-sided \(p\) \\",
        r"\midrule",
    ]
    previous_descriptor = None
    for row in summary.itertuples(index=False):
        if previous_descriptor is not None and row.topology_descriptor != previous_descriptor:
            lines.append(r"\midrule")
        lines.append(
            f"{row.topology_descriptor} vs. {row.control} & {int(row.dataset_encoder_wins)}/{int(row.dataset_encoder_n)} & "
            f"{100.0 * row.avg_gain:.1f}\\% & {_fmt_p(row.p_less)} & {_fmt_p(row.p_two_sided)} \\\\"
        )
        previous_descriptor = row.topology_descriptor
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    text = "\n".join(lines) + "\n"
    (OUT_DIR / "final_table_topology_descriptor_significance.txt").write_text(text, encoding="utf-8")
    (OUT_DIR / "final_table_topology_control_significance.txt").write_text(text, encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Wrote {OUT_DIR / 'final_table_topology_descriptor_significance.txt'}")


if __name__ == "__main__":
    main()
