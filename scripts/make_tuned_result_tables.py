#!/usr/bin/env python3
"""Generate paper tables from latex_tables/hyperparam_tuned.txt."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


INPUT_PATH = Path("latex_tables/hyperparam_tuned.txt")
OUT_DIR = Path("latex_tables")

DATASET_ORDER = [
    "bouncing_disks",
    "bouncing_rings",
    "orbiting_disks",
    "orbiting_rings",
    "moving_mnist",
    "lorenz96",
    "electric_devices",
    "hela",
    "glioblastoma",
]

DATASET_LABELS = {
    "bouncing_disks": "Bouncing disks",
    "bouncing_rings": "Bouncing rings",
    "orbiting_disks": "Orbiting disks",
    "orbiting_rings": "Orbiting rings",
    "moving_mnist": "Moving MNIST",
    "lorenz96": "Lorenz-96",
    "electric_devices": "ElectricDevices",
    "hela": "HeLa",
    "glioblastoma": "Glioblastoma",
}

SCENARIO_ORDER = ["latent_tda", "geo_latent_tda", "topo_latent_tda"]
ENCODER_LABELS = {
    "latent_tda": "AE",
    "geo_latent_tda": "GeoAE",
    "topo_latent_tda": "TopoAE",
}

FUSE_TOPO_MODES = {
    "z_fuse_h1": r"$H_1$",
    "z_fuse_pi_h1": "PI",
    "z_fuse_landscape_h1": "Landscape",
    "z_fuse_perslay_h1": "PersLay",
}

DIRECT_TOPO_LABELS = {
    "z_h0": r"$H_0$",
    "z_h1": r"$H_1$",
    "z_both": r"$H_0+H_1$",
    "z_pi_h0": r"PI $H_0$",
    "z_pi_h1": r"PI $H_1$",
    "z_pi_both": r"PI $H_0+H_1$",
    "z_landscape_h0": r"Landscape $H_0$",
    "z_landscape_h1": r"Landscape $H_1$",
    "z_landscape_both": r"Landscape $H_0+H_1$",
    "z_perslay_h0": r"PersLay $H_0$",
    "z_perslay_h1": r"PersLay $H_1$",
    "z_perslay_both": r"PersLay $H_0+H_1$",
}

DIAGRAM_MODES = {
    "z_fuse_h1": r"Betti $H_1$",
    "z_fuse_pi_h1": r"PI $H_1$",
    "z_fuse_landscape_h1": r"Landscape $H_1$",
    "z_fuse_perslay_h1": r"PersLay-style $H_1$",
}

PERTURBATION_MODES = ["z", "z_h1", "z_h1_zero", "z_h1_shuffle", "z_h1_noise", "z_h1_shift"]
CORRUPT_H1_MODES = ["z_h1_zero", "z_h1_shuffle", "z_h1_noise", "z_h1_shift"]


@dataclass(frozen=True)
class Row:
    dataset: str
    scenario: str
    mode: str
    n_runs: int
    mse_mean: float
    mse_std: float
    r2_mean: float
    r2_std: float


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def parse_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    tokens = path.read_text(encoding="utf-8").split()
    i = 0
    while i + 7 < len(tokens):
        if tokens[i] == "dataset":
            i += 1
            continue
        dataset, scenario, mode = tokens[i : i + 3]
        try:
            row = Row(
                dataset=dataset,
                scenario=scenario,
                mode=mode,
                n_runs=int(tokens[i + 3]),
                mse_mean=float(tokens[i + 4]),
                mse_std=float(tokens[i + 5]),
                r2_mean=float(tokens[i + 6]),
                r2_std=float(tokens[i + 7]),
            )
        except ValueError:
            i += 1
            continue
        rows.append(row)
        i += 8
    if not rows:
        raise SystemExit(f"No result rows parsed from {path}")
    return rows


def row_map(rows: list[Row]) -> dict[tuple[str, str, str], Row]:
    out: dict[tuple[str, str, str], Row] = {}
    for row in rows:
        out[(row.dataset, row.scenario, row.mode)] = row
    return out


def fmt(row: Row | None, *, digits: int = 4, bold: bool = False, star: bool = False, suffix: str = "") -> str:
    if row is None:
        return "--"
    body = f"{row.mse_mean:.{digits}f}{{\\pm}}{row.mse_std:.{digits}f}"
    if bold:
        body = rf"\mathbf{{{body}}}"
    if star:
        body += r"^{\star}"
    value = f"${body}$"
    if suffix:
        value += f" {suffix}"
    return value


def best(rows: list[Row]) -> Row | None:
    return min(rows, key=lambda row: row.mse_mean) if rows else None


def is_topology_direct(mode: str) -> bool:
    if mode == "z" or mode.startswith("topo_") or mode.startswith("z_fuse_"):
        return False
    if "pca" in mode or "kpca" in mode or "temporal" in mode or "corrupt" in mode:
        return False
    return mode.startswith(("z_h", "z_both", "z_pi_", "z_landscape_", "z_perslay_"))


def topology_label(mode: str) -> str:
    if mode in DIRECT_TOPO_LABELS:
        return DIRECT_TOPO_LABELS[mode]
    if mode in FUSE_TOPO_MODES:
        return f"Fuse {FUSE_TOPO_MODES[mode]}"
    return tex_escape(mode)


def make_main_table(rows_by_key: dict[tuple[str, str, str], Row], rows: list[Row]) -> None:
    unique_rows = list(rows_by_key.values())
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Main latent-forecasting results after selecting predictor hyperparameters on z-only validation MSE. Each dataset is evaluated with a standard AE, a geometry-regularized AE (GeoAE), and a topology-regularized AE (TopoAE). ``Direct'' denotes fixed topology descriptors appended to the latent state, while ``Fusion'' learns $z_{\mathrm{topo}}=\mathrm{MLP}([z,B])$ jointly with the predictor. Bold marks the best entry within each encoder row; $^\star$ marks the best result for the dataset. Lower MSE is better.}",
        r"\label{tab:main_latent_topology}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Encoder & $z$ only & Topology only & Best direct $z+$topology & Best fusion \\",
        r"\midrule",
    ]

    for dataset_idx, dataset in enumerate(DATASET_ORDER):
        dataset_candidates: list[Row] = []
        row_payloads: list[tuple[str, Row | None, Row | None, Row | None, Row | None]] = []
        for scenario in SCENARIO_ORDER:
            scenario_rows = [r for r in unique_rows if r.dataset == dataset and r.scenario == scenario]
            z_row = rows_by_key.get((dataset, scenario, "z"))
            topo_only = best([r for r in scenario_rows if r.mode.startswith("topo_")])
            direct = best([r for r in scenario_rows if is_topology_direct(r.mode)])
            fusion = best([r for r in scenario_rows if r.mode in FUSE_TOPO_MODES])
            payload = (ENCODER_LABELS[scenario], z_row, topo_only, direct, fusion)
            row_payloads.append(payload)
            dataset_candidates.extend([r for r in [z_row, topo_only, direct, fusion] if r is not None])
        dataset_best = best(dataset_candidates)

        for enc_idx, (encoder, z_row, topo_only, direct, fusion) in enumerate(row_payloads):
            row_entries = [r for r in [z_row, topo_only, direct, fusion] if r is not None]
            row_best = best(row_entries)
            label = DATASET_LABELS[dataset] if enc_idx == 0 else ""
            cells = [
                fmt(z_row, digits=4, bold=z_row == row_best, star=z_row == dataset_best),
                fmt(topo_only, digits=4, bold=topo_only == row_best, star=topo_only == dataset_best),
                fmt(direct, digits=4, bold=direct == row_best, star=direct == dataset_best),
                fmt(fusion, digits=4, bold=fusion == row_best, star=fusion == dataset_best),
            ]
            lines.append(f"{label} & {encoder} & " + " & ".join(cells) + r" \\")
        if dataset_idx != len(DATASET_ORDER) - 1:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    (OUT_DIR / "final_table_main.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_pca_control(rows_by_key: dict[tuple[str, str, str], Row]) -> None:
    csv_rows: list[dict[str, object]] = []
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Linear and nonlinear latent-window controls after z-only hyperparameter selection. PCA and KPCA replace the persistence descriptor with features computed from the same latent-window point cloud. ``Best topology'' is the best persistence descriptor among the evaluated direct and fused topology variants. Lower MSE is better.}",
        r"\label{tab:pca_control}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Dataset & Encoder & $z$ & $z+$PCA & $z+$KPCA & Fuse $H_1$ & Fuse PCA & Fuse KPCA & Best topology \\",
        r"\midrule",
    ]

    for dataset_idx, dataset in enumerate(DATASET_ORDER):
        for enc_idx, scenario in enumerate(SCENARIO_ORDER):
            encoder = ENCODER_LABELS[scenario]
            z = rows_by_key.get((dataset, scenario, "z"))
            z_pca = rows_by_key.get((dataset, scenario, "z_pca_h1"))
            z_kpca = rows_by_key.get((dataset, scenario, "z_kpca_h1"))
            fuse_h1 = rows_by_key.get((dataset, scenario, "z_fuse_h1"))
            fuse_pca = rows_by_key.get((dataset, scenario, "z_fuse_pca_h1"))
            fuse_kpca = rows_by_key.get((dataset, scenario, "z_fuse_kpca_h1"))
            direct_modes = [mode for (_d, _s, mode) in rows_by_key if _d == dataset and _s == scenario and is_topology_direct(mode)]
            topo_modes = sorted(set(direct_modes) | set(FUSE_TOPO_MODES))
            topo_candidates = [(mode, rows_by_key.get((dataset, scenario, mode))) for mode in topo_modes]
            topo_candidates = [(mode, row) for mode, row in topo_candidates if row is not None]
            best_mode, best_topo = min(topo_candidates, key=lambda item: item[1].mse_mean)
            displayed = [r for r in [z, z_pca, z_kpca, fuse_h1, fuse_pca, fuse_kpca, best_topo] if r is not None]
            row_best = best(displayed)
            label = DATASET_LABELS[dataset] if enc_idx == 0 else ""
            best_suffix = f"({topology_label(best_mode)})"
            cells = [
                fmt(z, digits=4, bold=z == row_best),
                fmt(z_pca, digits=4, bold=z_pca == row_best),
                fmt(z_kpca, digits=4, bold=z_kpca == row_best),
                fmt(fuse_h1, digits=4, bold=fuse_h1 == row_best),
                fmt(fuse_pca, digits=4, bold=fuse_pca == row_best),
                fmt(fuse_kpca, digits=4, bold=fuse_kpca == row_best),
                fmt(best_topo, digits=4, bold=best_topo == row_best, suffix=best_suffix),
            ]
            lines.append(f"{label} & {encoder} & " + " & ".join(cells) + r" \\")

            csv_rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "encoder": encoder,
                    "z_mean": z.mse_mean if z else "",
                    "z_std": z.mse_std if z else "",
                    "z_pca_mean": z_pca.mse_mean if z_pca else "",
                    "z_pca_std": z_pca.mse_std if z_pca else "",
                    "z_kpca_mean": z_kpca.mse_mean if z_kpca else "",
                    "z_kpca_std": z_kpca.mse_std if z_kpca else "",
                    "fuse_h1_mean": fuse_h1.mse_mean if fuse_h1 else "",
                    "fuse_h1_std": fuse_h1.mse_std if fuse_h1 else "",
                    "fuse_pca_mean": fuse_pca.mse_mean if fuse_pca else "",
                    "fuse_pca_std": fuse_pca.mse_std if fuse_pca else "",
                    "fuse_kpca_mean": fuse_kpca.mse_mean if fuse_kpca else "",
                    "fuse_kpca_std": fuse_kpca.mse_std if fuse_kpca else "",
                    "best_topo_mean": best_topo.mse_mean,
                    "best_topo_std": best_topo.mse_std,
                    "best_topo_mode": best_mode,
                    "best_topo_label": topology_label(best_mode),
                }
            )
        if dataset_idx != len(DATASET_ORDER) - 1:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    (OUT_DIR / "final_table_pca_control.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (OUT_DIR / "pca_kpca_control_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def make_diagram_vectorization(rows_by_key: dict[tuple[str, str, str], Row]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Ablation over fused persistence-diagram vectorizations using the standard AE encoder. Lower MSE is better.}",
        r"\label{tab:diagram_vectorization}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Dataset & Betti $H_1$ & PI $H_1$ & Landscape $H_1$ & PersLay-style $H_1$ \\",
        r"\midrule",
    ]
    for dataset in DATASET_ORDER:
        rows = [rows_by_key.get((dataset, "latent_tda", mode)) for mode in DIAGRAM_MODES]
        present = [r for r in rows if r is not None]
        row_best = best(present)
        cells = [fmt(r, digits=4, bold=r == row_best) for r in rows]
        lines.append(f"{DATASET_LABELS[dataset]} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUT_DIR / "final_table_encoder_benchmark.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_perturbation_tables(rows_by_key: dict[tuple[str, str, str], Row]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{$H_1$ perturbation controls after z-only hyperparameter selection. ``Best corrupt'' is the best of zero, shuffle, noise, and temporal-shift controls. Corrupted controls can remain competitive, so this table is a diagnostic control rather than evidence that exact $H_1$ semantics alone explain the gains. Lower MSE is better.}",
        r"\label{tab:perturbation_sanity}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Encoder & $z$ & $z+H_1$ & $z+$best corrupt $H_1$ \\",
        r"\midrule",
    ]

    for dataset_idx, dataset in enumerate(DATASET_ORDER):
        for enc_idx, scenario in enumerate(SCENARIO_ORDER):
            z = rows_by_key.get((dataset, scenario, "z"))
            h1 = rows_by_key.get((dataset, scenario, "z_h1"))
            corrupt = best([rows_by_key[(dataset, scenario, mode)] for mode in CORRUPT_H1_MODES if (dataset, scenario, mode) in rows_by_key])
            row_best = best([r for r in [z, h1, corrupt] if r is not None])
            label = DATASET_LABELS[dataset] if enc_idx == 0 else ""
            cells = [
                fmt(z, digits=4, bold=z == row_best),
                fmt(h1, digits=4, bold=h1 == row_best),
                fmt(corrupt, digits=4, bold=corrupt == row_best),
            ]
            lines.append(f"{label} & {ENCODER_LABELS[scenario]} & " + " & ".join(cells) + r" \\")
        if dataset_idx != len(DATASET_ORDER) - 1:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    (OUT_DIR / "final_table_perturbation_sanity.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sweep_lines = [
        r"\begin{scriptsize}",
        r"\begin{longtable}{lllccc}",
        r"\caption{Perturbation sweep over $H_1$ topology controls after z-only hyperparameter selection. Lower MSE and higher latent $R^2$ are better.}",
        r"\label{tab:appendix_perturbation_sweep}\\",
        r"\toprule",
        r"Dataset & Scenario & Mode & $n$ & MSE & Latent $R^2$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Dataset & Scenario & Mode & $n$ & MSE & Latent $R^2$ \\",
        r"\midrule",
        r"\endhead",
    ]
    perturb_rows = [
        row
        for row in rows_by_key.values()
        if row.dataset in DATASET_ORDER and row.scenario in SCENARIO_ORDER and row.mode in PERTURBATION_MODES
    ]
    ordered = sorted(
        perturb_rows,
        key=lambda r: (
            DATASET_ORDER.index(r.dataset),
            SCENARIO_ORDER.index(r.scenario),
            r.mse_mean,
            r.mode,
        ),
    )
    last_dataset = None
    for row in ordered:
        if last_dataset is not None and row.dataset != last_dataset:
            sweep_lines.append(r"\midrule")
        last_dataset = row.dataset
        sweep_lines.append(
            f"{tex_escape(row.dataset)} & {tex_escape(row.scenario)} & {tex_escape(row.mode)} & {row.n_runs} & "
            f"${row.mse_mean:.4f}\\pm{row.mse_std:.4f}$ & ${row.r2_mean:.4f}\\pm{row.r2_std:.4f}$ \\\\"
        )
    sweep_lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{scriptsize}"])
    (OUT_DIR / "final_table_appendix_b.txt").write_text("\n".join(sweep_lines) + "\n", encoding="utf-8")


def make_full_sweep(rows: list[Row]) -> None:
    lines = [
        r"\begin{scriptsize}",
        r"\begin{longtable}{lllccc}",
        r"\caption{Full tuned sweep over datasets, encoder families, and topology/fusion modes. Lower MSE and higher latent $R^2$ are better.}",
        r"\label{tab:appendix_full_sweep}\\",
        r"\toprule",
        r"Dataset & Scenario & Mode & $n$ & MSE & Latent $R^2$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Dataset & Scenario & Mode & $n$ & MSE & Latent $R^2$ \\",
        r"\midrule",
        r"\endhead",
    ]
    ordered = sorted(
        rows,
        key=lambda r: (
            DATASET_ORDER.index(r.dataset) if r.dataset in DATASET_ORDER else len(DATASET_ORDER),
            SCENARIO_ORDER.index(r.scenario) if r.scenario in SCENARIO_ORDER else len(SCENARIO_ORDER),
            r.mse_mean,
            r.mode,
        ),
    )
    last_dataset = None
    for row in ordered:
        if last_dataset is not None and row.dataset != last_dataset:
            lines.append(r"\midrule")
        last_dataset = row.dataset
        lines.append(
            f"{tex_escape(row.dataset)} & {tex_escape(row.scenario)} & {tex_escape(row.mode)} & {row.n_runs} & "
            f"${row.mse_mean:.4f}\\pm{row.mse_std:.4f}$ & ${row.r2_mean:.4f}\\pm{row.r2_std:.4f}$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{scriptsize}"])
    (OUT_DIR / "final_table_appendix_full_sweep.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = parse_rows(INPUT_PATH)
    rows_by_key = row_map(rows)
    unique_rows = list(rows_by_key.values())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_main_table(rows_by_key, rows)
    make_pca_control(rows_by_key)
    make_diagram_vectorization(rows_by_key)
    make_perturbation_tables(rows_by_key)
    make_full_sweep(unique_rows)
    print(f"Parsed {len(rows)} rows from {INPUT_PATH} ({len(unique_rows)} unique dataset/scenario/mode rows)")
    print("Wrote final_table_main.txt")
    print("Wrote final_table_pca_control.txt")
    print("Wrote final_table_encoder_benchmark.txt")
    print("Wrote final_table_perturbation_sanity.txt")
    print("Wrote final_table_appendix_b.txt")
    print("Wrote final_table_appendix_full_sweep.txt")
    print("Wrote pca_kpca_control_summary.csv")


if __name__ == "__main__":
    main()
