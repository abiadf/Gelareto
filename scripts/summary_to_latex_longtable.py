#!/usr/bin/env python3
"""Convert whitespace-aligned summary output into a LaTeX longtable.

Usage:
    python scripts/summary_to_latex_longtable.py raw_summary.txt > final_table_appendix_full_sweep.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def parse_line(line: str) -> tuple[str, str, str, int, float, float, float, float] | None:
    parts = line.split()
    if len(parts) != 8:
        return None
    dataset, scenario, mode, n_runs, mse_mean, mse_std, r2_mean, r2_std = parts
    try:
        return (
            dataset,
            scenario,
            mode,
            int(n_runs),
            float(mse_mean),
            float(mse_std),
            float(r2_mean),
            float(r2_std),
        )
    except ValueError:
        return None


def read_rows(path: Path) -> list[tuple[str, str, str, int, float, float, float, float]]:
    text = path.read_text()

    # Terminal output is often soft-wrapped, so parse the full whitespace token
    # stream instead of assuming each row occupies exactly one line.
    # dataset scenario mode n_runs mse_mean mse_std r2_mean r2_std
    rows = []
    tokens = text.split()
    i = 0
    while i + 7 < len(tokens):
        if tokens[i] == "dataset":
            i += 1
            continue
        candidate = " ".join(tokens[i : i + 8])
        parsed = parse_line(candidate)
        if parsed is not None:
            rows.append(parsed)
            i += 8
        else:
            i += 1
    return rows


def emit_table(rows: list[tuple[str, str, str, int, float, float, float, float]], caption: str, label: str) -> str:
    out = [
        r"\begin{scriptsize}",
        r"\begin{longtable}{lllccc}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}\\",
        r"\toprule",
        r"Dataset & Scenario & Mode & $n$ & MSE & Latent $R^2$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Dataset & Scenario & Mode & $n$ & MSE & Latent $R^2$ \\",
        r"\midrule",
        r"\endhead",
    ]
    last_dataset = None
    for dataset, scenario, mode, n_runs, mse_mean, mse_std, r2_mean, r2_std in rows:
        if last_dataset is not None and dataset != last_dataset:
            out.append(r"\midrule")
        last_dataset = dataset
        out.append(
            f"{tex_escape(dataset)} & {tex_escape(scenario)} & {tex_escape(mode)} & {n_runs} & "
            f"${mse_mean:.4f}\\pm{mse_std:.4f}$ & ${r2_mean:.4f}\\pm{r2_std:.4f}$ \\\\"
        )
    out.extend([r"\bottomrule", r"\end{longtable}", r"\end{scriptsize}"])
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--caption", default="Full sweep over datasets, encoder families, and topology/fusion modes. Lower MSE and higher latent $R^2$ are better.")
    parser.add_argument("--label", default="tab:appendix_full_sweep")
    args = parser.parse_args()
    rows = read_rows(args.input)
    if not rows:
        text = args.input.read_text(errors="ignore")
        if r"\begin{" in text or r"\toprule" in text:
            print(
                f"error: {args.input} appears to contain LaTeX, not raw summary rows. "
                "Paste the raw whitespace summary into this file and rerun.",
                file=sys.stderr,
            )
        else:
            print(
                f"error: parsed 0 rows from {args.input}. Expected rows like: "
                "dataset scenario mode n_runs mse_mean mse_std r2_mean r2_std",
                file=sys.stderr,
            )
        raise SystemExit(2)
    print(emit_table(rows, args.caption, args.label), end="")


if __name__ == "__main__":
    main()
