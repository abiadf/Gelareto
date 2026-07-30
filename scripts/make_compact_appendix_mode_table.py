#!/usr/bin/env python3
"""Make a wide, encoder-grouped appendix table from the full mode summary."""

from __future__ import annotations

from pathlib import Path

from make_tuned_result_tables import (
    DATASET_LABELS,
    DATASET_ORDER,
    SCENARIO_ORDER,
    Row,
    parse_rows,
    tex_escape,
)


INPUT = Path("latex_tables/raw_appendix_a_summary.txt")
OUTPUT = Path("latex_tables/compact_appendix_modes_by_encoder.txt")


def metric(mean: float, std: float) -> str:
    return rf"${mean:.4f}\mathbin{{\pm}}{std:.4f}$"


def mode_label(mode: str) -> str:
    return rf"\texttt{{{tex_escape(mode)}}}"


def make_table(rows: list[Row]) -> str:
    lookup = {(row.dataset, row.scenario, row.mode): row for row in rows}
    modes = sorted({row.mode for row in rows})
    # Put the plain latent baseline first; keep the remaining complete mode
    # names alphabetical so every dataset block has an identical row order.
    modes = (["z"] if "z" in modes else []) + [mode for mode in modes if mode != "z"]

    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{0.88}",
        r"\begin{longtable}{llcccccc}",
        (
            r"\caption{Complete mode-level forecasting results in a compact encoder-grouped layout. "
            r"Entries are five-seed mean$\pm$SD. Lower MSE and higher $R^2$ are better.}"
            r"\label{tab:appendix_modes_by_encoder}\\"
        ),
        r"\toprule",
        r" & & \multicolumn{2}{c}{AE} & \multicolumn{2}{c}{GeoAE} & \multicolumn{2}{c}{TopoAE} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Dataset & Mode & MSE & $R^2$ & MSE & $R^2$ & MSE & $R^2$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{8}{c}{\tablename\ \thetable{} -- continued} \\",
        r"\toprule",
        r" & & \multicolumn{2}{c}{AE} & \multicolumn{2}{c}{GeoAE} & \multicolumn{2}{c}{TopoAE} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Dataset & Mode & MSE & $R^2$ & MSE & $R^2$ & MSE & $R^2$ \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{8}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for dataset_index, dataset in enumerate(DATASET_ORDER):
        available_modes = [
            mode
            for mode in modes
            if any((dataset, scenario, mode) in lookup for scenario in SCENARIO_ORDER)
        ]
        for mode_index, mode in enumerate(available_modes):
            cells = [
                DATASET_LABELS[dataset] if mode_index == 0 else "",
                mode_label(mode),
            ]
            for scenario in SCENARIO_ORDER:
                row = lookup.get((dataset, scenario, mode))
                if row is None:
                    cells.extend(["--", "--"])
                else:
                    cells.extend(
                        [
                            metric(row.mse_mean, row.mse_std),
                            metric(row.r2_mean, row.r2_std),
                        ]
                    )
            lines.append(" & ".join(cells) + r" \\")
        if dataset_index < len(DATASET_ORDER) - 1:
            lines.append(r"\midrule")

    lines.extend([r"\end{longtable}", r"\endgroup"])
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = parse_rows(INPUT)
    OUTPUT.write_text(make_table(rows), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
