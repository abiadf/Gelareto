#!/usr/bin/env python3
"""Merge representation fidelity with topology gain and report Spearman correlations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FIDELITY_METRICS = (
    "metric_distortion_median",
    "h0_bottleneck_median",
    "h1_bottleneck_median",
)


def _read_many(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("At least one input CSV is required")
    return pd.concat(frames, ignore_index=True, sort=False)


def _ensure_encoder(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "encoder" not in result:
        if "scenario" not in result:
            raise ValueError("Input needs an encoder or scenario column")
        mapping = {
            "latent_tda": "ae",
            "geo_latent_tda": "geo_ae",
            "topo_latent_tda": "topo_ae",
        }
        result["encoder"] = result["scenario"].map(mapping)
    if "encoder_variant" not in result:
        result["encoder_variant"] = result["encoder"].astype(str)
        geo_mask = result["encoder"].eq("geo_ae")
        if "geo_ae_lambda" in result:
            result.loc[geo_mask, "encoder_variant"] = (
                "geo_ae_lambda"
                + result.loc[geo_mask, "geo_ae_lambda"].astype(float).map(lambda value: f"{value:g}")
            )
        topo_mask = result["encoder"].eq("topo_ae")
        if "topo_ae_lambda" in result:
            topo_variant = (
                "topo_ae_lambda"
                + result.loc[topo_mask, "topo_ae_lambda"].astype(float).map(lambda value: f"{value:g}")
            )
            if "topo_ae_distance" in result:
                topo_variant = topo_variant + "_" + result.loc[topo_mask, "topo_ae_distance"].astype(str)
            result.loc[topo_mask, "encoder_variant"] = topo_variant
    return result


def topology_gain(forecast: pd.DataFrame, topology_mode: str, mse_column: str) -> pd.DataFrame:
    forecast = _ensure_encoder(forecast)
    required = {"dataset", "encoder", "encoder_variant", "seed", "mode", mse_column}
    missing = sorted(required - set(forecast.columns))
    if missing:
        raise ValueError(f"Forecast CSV is missing columns: {', '.join(missing)}")
    selected = forecast[forecast["mode"].isin(["z", topology_mode])].copy()
    key_cols = ["dataset", "encoder", "encoder_variant", "seed"]
    duplicated = selected.duplicated([*key_cols, "mode"], keep=False)
    if duplicated.any():
        keys = selected.loc[duplicated, [*key_cols, "mode"]].head()
        raise ValueError(f"Forecast input has duplicate paired rows, for example:\n{keys}")
    wide = selected.pivot(
        index=key_cols, columns="mode", values=mse_column
    ).reset_index()
    if "z" not in wide or topology_mode not in wide:
        raise ValueError(f"Forecast input must contain both mode=z and mode={topology_mode}")
    wide = wide.dropna(subset=["z", topology_mode]).copy()
    wide["mse_z"] = wide["z"].astype(float)
    wide["mse_z_plus_topology"] = wide[topology_mode].astype(float)
    wide["topology_mode"] = topology_mode
    wide["topology_gain"] = (
        wide["mse_z"] - wide["mse_z_plus_topology"]
    ) / wide["mse_z"].replace(0.0, np.nan)
    return wide[
        [
            "dataset",
            "encoder",
            "encoder_variant",
            "seed",
            "topology_mode",
            "mse_z",
            "mse_z_plus_topology",
            "topology_gain",
        ]
    ]


def correlation_table(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [
        ("metric_distortion_median", "h0_bottleneck_median"),
        ("metric_distortion_median", "h1_bottleneck_median"),
        ("metric_distortion_median", "topology_gain"),
        ("h0_bottleneck_median", "topology_gain"),
        ("h1_bottleneck_median", "topology_gain"),
    ]
    for x_name, y_name in pairs:
        pair = merged[[x_name, y_name]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 3:
            rho, p_value, status = "undefined", "undefined", "fewer than 3 matched points"
        elif pair[x_name].nunique() < 2:
            rho, p_value, status = "undefined", "undefined", f"{x_name} is constant"
        elif pair[y_name].nunique() < 2:
            rho, p_value, status = "undefined", "undefined", f"{y_name} is constant"
        else:
            rho, p_value = spearmanr(pair[x_name], pair[y_name])
            rho, p_value, status = float(rho), float(p_value), "ok"
        rows.append(
            {
                "x": x_name,
                "y": y_name,
                "n": int(len(pair)),
                "spearman_rho": rho,
                "p_value": p_value,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def compact_model_table(merged: pd.DataFrame) -> pd.DataFrame:
    """One readable row per dataset and encoder variant."""
    columns = [
        "metric_distortion_median",
        "h0_bottleneck_median",
        "h1_bottleneck_median",
        "topology_gain",
    ]
    return (
        merged.groupby(["dataset", "encoder_variant"], sort=False)[columns]
        .median()
        .reset_index()
        .rename(
            columns={
                "encoder_variant": "encoder",
                "metric_distortion_median": "metric_distortion",
                "h0_bottleneck_median": "h0_distortion",
                "h1_bottleneck_median": "h1_distortion",
                "topology_gain": "predictive_gain",
            }
        )
    )


def interpretation_text(correlations: pd.DataFrame) -> str:
    def describe(x_name, y_name, positive):
        row = correlations[(correlations["x"] == x_name) & (correlations["y"] == y_name)].iloc[0]
        if row["status"] != "ok":
            return f"- {x_name} vs {y_name}: not estimable ({row['status']})."
        rho = float(row["spearman_rho"])
        expected = rho > 0 if positive else rho < 0
        direction = "positive" if rho > 0 else "negative" if rho < 0 else "zero"
        verdict = "matches" if expected else "does not match"
        return f"- {x_name} vs {y_name}: rho={rho:.3f} ({direction}); {verdict} the expected direction."

    lines = [
        "Interpretation",
        "Lower distortion is better; positive predictive gain means z+H1 improved over z.",
        describe("metric_distortion_median", "h0_bottleneck_median", True),
        describe("metric_distortion_median", "h1_bottleneck_median", True),
        describe("metric_distortion_median", "topology_gain", False),
        describe("h0_bottleneck_median", "topology_gain", False),
        describe("h1_bottleneck_median", "topology_gain", False),
    ]
    return "\n".join(lines)


def write_text_report(merged: pd.DataFrame, correlations: pd.DataFrame, path: Path) -> None:
    compact = compact_model_table(merged)
    report = [
        "Representation fidelity and predictive gain",
        "",
        compact.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "Spearman correlations",
        correlations.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        interpretation_text(correlations),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fidelity", type=Path, nargs="+", required=True, help="Fidelity summary CSV(s).")
    parser.add_argument("--forecast", type=Path, nargs="+", required=True, help="Forecast per-seed results CSV(s).")
    parser.add_argument("--topology-mode", default="z_h1", help="Predeclared z+topology forecasting mode.")
    parser.add_argument("--mse-column", default="test_mse")
    parser.add_argument("--output-dir", type=Path, default=Path("results/representation_fidelity_analysis"))
    parser.add_argument("--report-file", type=Path, default=Path("latex_tables/fidelity_runs.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fidelity = _ensure_encoder(_read_many(args.fidelity))
    key_cols = ["dataset", "encoder", "encoder_variant", "seed"]
    required = {*key_cols, *FIDELITY_METRICS}
    missing = sorted(required - set(fidelity.columns))
    if missing:
        raise ValueError(f"Fidelity CSV is missing columns: {', '.join(missing)}")
    if fidelity.duplicated(key_cols).any():
        raise ValueError("Fidelity summaries must contain one row per dataset-encoder-variant-seed")

    gain = topology_gain(_read_many(args.forecast), args.topology_mode, args.mse_column)
    merged = fidelity.merge(gain, on=key_cols, how="inner", validate="one_to_one")
    correlations = correlation_table(merged)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "fidelity_with_topology_gain.csv", index=False)
    correlations.to_csv(args.output_dir / "spearman_correlations.csv", index=False)
    write_text_report(merged, correlations, args.report_file)
    print("\nMedian results by dataset and encoder:")
    print(compact_model_table(merged).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSpearman correlations:")
    print(correlations.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved analysis outputs to {args.output_dir}")
    print(f"Saved compact report to {args.report_file}")


if __name__ == "__main__":
    main()
