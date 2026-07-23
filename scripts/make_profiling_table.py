from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("latex_tables/profiling.txt")
OUTPUT_PATH = Path("latex_tables/profiling_table.txt")

COLUMNS = [
    "dataset",
    "scenario",
    "mode",
    "profile_size",
    "wall_time_sec",
    "peak_cpu_rss_mb",
    "peak_accelerator_mem_mb",
    "topo_time_sec",
    "topo_fraction",
    "throughput_frames_per_sec",
    "n_train_clips",
    "n_test_clips",
    "sequence_len",
    "frame_size",
    "n_processed_frames",
    "device",
    "profile_baseline_mode",
    "overhead_vs_baseline",
]

ENCODER_LABELS = {
    "latent_tda": "AE",
    "geo_latent_tda": "GeoAE",
    "topo_latent_tda": "TopoAE",
}

MODE_LABELS = {
    "z": r"\(z\)",
    "z_fuse_h1": r"\(z_{\mathrm{fuse}}+H_1\)",
    "z_fuse_perslay_h1": r"\(z_{\mathrm{fuse}}+\mathrm{PersLay}\)",
    "z_fuse_pca_h1": r"\(z_{\mathrm{fuse}}+\mathrm{PCA}\)",
}


def read_profile(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.startswith("dataset"):
            continue
        parts = stripped.split()
        if len(parts) == len(COLUMNS):
            rows.append(parts)
    if not rows:
        raise ValueError(f"No profiling rows found in {path}")

    df = pd.DataFrame(rows, columns=COLUMNS)
    numeric_cols = [
        "profile_size",
        "wall_time_sec",
        "peak_accelerator_mem_mb",
        "overhead_vs_baseline",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    return df


def make_table(df: pd.DataFrame) -> str:
    profile_size = int(df["profile_size"].max())
    subset = df[df["profile_size"] == profile_size].copy()
    subset = subset[subset["scenario"].isin(ENCODER_LABELS)]
    subset = subset[subset["mode"].isin(MODE_LABELS)]
    subset["encoder"] = subset["scenario"].map(ENCODER_LABELS)
    subset["mode_label"] = subset["mode"].map(MODE_LABELS)

    grouped = (
        subset.groupby(["encoder", "mode", "mode_label"], sort=False)
        .agg(
            runtime_mean=("wall_time_sec", "mean"),
            runtime_std=("wall_time_sec", "std"),
            gpu_max=("peak_accelerator_mem_mb", "max"),
            overhead_mean=("overhead_vs_baseline", "mean"),
            overhead_std=("overhead_vs_baseline", "std"),
        )
        .reset_index()
    )
    encoder_order = {name: idx for idx, name in enumerate(["AE", "GeoAE", "TopoAE"])}
    mode_order = {name: idx for idx, name in enumerate(MODE_LABELS)}
    grouped["encoder_order"] = grouped["encoder"].map(encoder_order)
    grouped["mode_order"] = grouped["mode"].map(mode_order)
    grouped = grouped.sort_values(["encoder_order", "mode_order"])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Profiling summary at profile size {profile_size}, averaged over datasets. Runtime is end-to-end wall-clock time for one profiled run. GPU memory is the maximum accelerator memory observed. Runtime/\(z\) is relative to the corresponding \(z\)-only run within the same dataset and encoder family.}}",
        r"\label{tab:profiling}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Encoder & Input mode & Runtime (s) & Peak GPU mem. (MB) & Runtime / \(z\) \\",
        r"\midrule",
    ]
    previous_encoder = None
    for row in grouped.itertuples(index=False):
        if previous_encoder is not None and row.encoder != previous_encoder:
            lines.append(r"\midrule")
        runtime = f"{row.runtime_mean:.2f}$\\pm${0.0 if pd.isna(row.runtime_std) else row.runtime_std:.2f}"
        overhead = f"{row.overhead_mean:.2f}$\\pm${0.0 if pd.isna(row.overhead_std) else row.overhead_std:.2f}"
        lines.append(
            f"{row.encoder} & {row.mode_label} & {runtime} & {row.gpu_max:.0f} & {overhead} " + r"\\"
        )
        previous_encoder = row.encoder
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    df = read_profile(INPUT_PATH)
    OUTPUT_PATH.write_text(make_table(df), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
