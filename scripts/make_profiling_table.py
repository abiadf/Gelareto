from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("latex_tables/profiling.txt")
OUTPUT_PATH = Path("latex_tables/profiling_table.txt")
INVALID_OUTPUT_PATH = Path("latex_tables/profiling_invalid_rows.txt")

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
    "encoding_time_sec",
    "encoder_train_time_sec",
    "persistence_time_sec",
    "fusion_time_sec",
    "predictor_time_sec",
    "encoding_peak_accelerator_mem_mb",
    "encoder_train_peak_accelerator_mem_mb",
    "persistence_peak_accelerator_mem_mb",
    "fusion_peak_accelerator_mem_mb",
    "predictor_peak_accelerator_mem_mb",
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

PERSISTENCE_MODES = {
    "z_fuse_h1",
    "z_fuse_perslay_h1",
}

OLD_COLUMNS = [
    col
    for col in COLUMNS
    if col
    not in {
        "encoding_time_sec",
        "encoder_train_time_sec",
        "persistence_time_sec",
        "fusion_time_sec",
        "predictor_time_sec",
        "encoding_peak_accelerator_mem_mb",
        "encoder_train_peak_accelerator_mem_mb",
        "persistence_peak_accelerator_mem_mb",
        "fusion_peak_accelerator_mem_mb",
        "predictor_peak_accelerator_mem_mb",
    }
]

PHASE_PEAK_COLUMNS = {
    "encoding": "encoding_peak_accelerator_mem_mb",
    "encoder_train": "encoder_train_peak_accelerator_mem_mb",
    "persistence": "persistence_peak_accelerator_mem_mb",
    "fusion": "fusion_peak_accelerator_mem_mb",
    "predictor": "predictor_peak_accelerator_mem_mb",
}


def read_profile(path: Path) -> pd.DataFrame:
    rows = []
    active_columns = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-"):
            continue
        if stripped.startswith("dataset"):
            header_parts = stripped.split()
            if "dataset" in header_parts and "scenario" in header_parts and "mode" in header_parts:
                active_columns = header_parts
            continue
        parts = stripped.split()
        if active_columns is not None and len(parts) == len(active_columns):
            row = dict(zip(active_columns, parts))
            for col in COLUMNS:
                row.setdefault(col, "nan")
            rows.append(row)
        elif len(parts) == len(COLUMNS):
            row = dict(zip(COLUMNS, parts))
            for col in COLUMNS:
                row.setdefault(col, "nan")
            rows.append(row)
        elif len(parts) == len(OLD_COLUMNS):
            row = dict(zip(OLD_COLUMNS, parts))
            for col in COLUMNS:
                row.setdefault(col, "nan")
            rows.append(row)
    if not rows:
        raise ValueError(f"No profiling rows found in {path}")

    df = pd.DataFrame(rows, columns=COLUMNS)
    numeric_cols = [
        "profile_size",
        "wall_time_sec",
        "peak_accelerator_mem_mb",
        "topo_time_sec",
        "topo_fraction",
        "overhead_vs_baseline",
        *PHASE_PEAK_COLUMNS.values(),
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def make_table(df: pd.DataFrame) -> str:
    profile_size = int(df["profile_size"].max())
    subset = df[df["profile_size"] == profile_size].copy()
    subset = subset[subset["scenario"].isin(ENCODER_LABELS)]
    subset = subset[subset["mode"].isin(MODE_LABELS)]
    bad = subset[
        (~subset["mode"].isin(PERSISTENCE_MODES))
        & (subset["topo_time_sec"].astype(float) > 1e-6)
    ].copy()
    if not bad.empty:
        INVALID_OUTPUT_PATH.write_text(
            bad.to_string(
                index=False,
                columns=["dataset", "scenario", "mode", "profile_size", "wall_time_sec", "topo_time_sec", "topo_fraction"],
            ),
            encoding="utf-8",
        )
        bad_keys = bad[["dataset", "scenario", "profile_size"]].drop_duplicates()
        subset = subset.merge(
            bad_keys.assign(_bad_profile_group=True),
            on=["dataset", "scenario", "profile_size"],
            how="left",
        )
        subset = subset[subset["_bad_profile_group"].isna()].drop(columns=["_bad_profile_group"])
    else:
        INVALID_OUTPUT_PATH.write_text("", encoding="utf-8")

    subset["encoder"] = subset["scenario"].map(ENCODER_LABELS)
    subset["mode_label"] = subset["mode"].map(MODE_LABELS)
    subset["topo_percent"] = 100.0 * subset["topo_fraction"]

    grouped = (
        subset.groupby(["encoder", "mode", "mode_label"], sort=False)
        .agg(
            runtime_mean=("wall_time_sec", "mean"),
            runtime_std=("wall_time_sec", "std"),
            gpu_mean=("peak_accelerator_mem_mb", "mean"),
            gpu_std=("peak_accelerator_mem_mb", "std"),
            overhead_mean=("overhead_vs_baseline", "mean"),
            overhead_std=("overhead_vs_baseline", "std"),
            topo_mean=("topo_percent", "mean"),
            topo_std=("topo_percent", "std"),
            encoding_gpu_mean=("encoding_peak_accelerator_mem_mb", "mean"),
            encoder_train_gpu_mean=("encoder_train_peak_accelerator_mem_mb", "mean"),
            persistence_gpu_mean=("persistence_peak_accelerator_mem_mb", "mean"),
            fusion_gpu_mean=("fusion_peak_accelerator_mem_mb", "mean"),
            predictor_gpu_mean=("predictor_peak_accelerator_mem_mb", "mean"),
            n_datasets=("dataset", "nunique"),
        )
        .reset_index()
    )
    encoder_order = {name: idx for idx, name in enumerate(["AE", "GeoAE", "TopoAE"])}
    mode_order = {name: idx for idx, name in enumerate(MODE_LABELS)}
    grouped["encoder_order"] = grouped["encoder"].map(encoder_order)
    grouped["mode_order"] = grouped["mode"].map(mode_order)
    grouped = grouped.sort_values(["encoder_order", "mode_order"])
    phase_cols = {
        "Enc.": "encoding_gpu_mean",
        "Train": "encoder_train_gpu_mean",
        "TDA": "persistence_gpu_mean",
        "Fuse": "fusion_gpu_mean",
        "Pred.": "predictor_gpu_mean",
    }

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Profiling summary at profile size {profile_size}, averaged over datasets. Runtime and peak GPU memory are reported as mean$\pm$std across datasets. Runtime/\(z\) is relative to the corresponding \(z\)-only run within the same dataset and encoder family. TDA time is the fraction of runtime spent in persistence computations; it is zero for \(z\)-only and PCA-control modes. The GPU-heavy phase is the phase with the largest measured peak GPU allocation.}}",
        r"\label{tab:profiling}",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Encoder & Input mode & Runtime (s) & Peak GPU mem. (MB) & Runtime / \(z\) & TDA time (\%) & GPU-heavy phase \\",
        r"\midrule",
    ]
    previous_encoder = None
    for row in grouped.itertuples(index=False):
        if previous_encoder is not None and row.encoder != previous_encoder:
            lines.append(r"\midrule")
        runtime = f"{row.runtime_mean:.2f}$\\pm${0.0 if pd.isna(row.runtime_std) else row.runtime_std:.2f}"
        gpu = f"{row.gpu_mean:.0f}$\\pm${0.0 if pd.isna(row.gpu_std) else row.gpu_std:.0f}"
        overhead = f"{row.overhead_mean:.2f}$\\pm${0.0 if pd.isna(row.overhead_std) else row.overhead_std:.2f}"
        topo = f"{row.topo_mean:.1f}$\\pm${0.0 if pd.isna(row.topo_std) else row.topo_std:.1f}"
        phase_values = {
            label: getattr(row, col)
            for label, col in phase_cols.items()
            if not pd.isna(getattr(row, col))
        }
        gpu_heavy_phase = max(phase_values, key=phase_values.get) if phase_values else "--"
        lines.append(
            f"{row.encoder} & {row.mode_label} & {runtime} & {gpu} & {overhead} & {topo} & {gpu_heavy_phase} " + r"\\"
        )
        previous_encoder = row.encoder
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
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
