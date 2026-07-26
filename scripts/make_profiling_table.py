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
    "profile_repeat",
    "profile_is_warmup",
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
    "encoding_peak_cpu_rss_mb",
    "encoder_train_peak_cpu_rss_mb",
    "persistence_peak_cpu_rss_mb",
    "fusion_peak_cpu_rss_mb",
    "predictor_peak_cpu_rss_mb",
    "throughput_frames_per_sec",
    "n_train_clips",
    "n_test_clips",
    "sequence_len",
    "frame_size",
    "n_processed_frames",
    "device",
    "profile_baseline_mode",
    "overhead_vs_baseline",
    "added_time_vs_baseline_sec",
]

ENCODER_LABELS = {
    "latent_tda": "AE",
    "geo_latent_tda": "GeoAE",
    "topo_latent_tda": "TopoAE",
}

MODE_LABELS = {
    "z": r"\(z\)",
    "z_pca_h1": r"\(z+\mathrm{PCA}\)",
    "z_kpca_h1": r"\(z+\mathrm{KPCA}\)",
    "z_fuse_h1": r"\(z_{\mathrm{fuse}}+H_1\)",
    "z_fuse_pi_h1": r"\(z_{\mathrm{fuse}}+\mathrm{PI}\)",
    "z_fuse_landscape_h1": r"\(z_{\mathrm{fuse}}+\mathrm{Landscape}\)",
    "z_fuse_perslay_h1": r"\(z_{\mathrm{fuse}}+\mathrm{PersLay}\)",
    "z_fuse_pca_h1": r"\(z_{\mathrm{fuse}}+\mathrm{PCA}\)",
    "z_fuse_kpca_h1": r"\(z_{\mathrm{fuse}}+\mathrm{KPCA}\)",
}

PERSISTENCE_MODES = {
    "z_fuse_h1",
    "z_fuse_pi_h1",
    "z_fuse_landscape_h1",
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
        "encoding_peak_cpu_rss_mb",
        "encoder_train_peak_cpu_rss_mb",
        "persistence_peak_cpu_rss_mb",
        "fusion_peak_cpu_rss_mb",
        "predictor_peak_cpu_rss_mb",
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
        "profile_repeat",
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
        "overhead_vs_baseline",
        "added_time_vs_baseline_sec",
        *PHASE_PEAK_COLUMNS.values(),
        "encoding_peak_cpu_rss_mb",
        "encoder_train_peak_cpu_rss_mb",
        "persistence_peak_cpu_rss_mb",
        "fusion_peak_cpu_rss_mb",
        "predictor_peak_cpu_rss_mb",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def make_table(df: pd.DataFrame) -> str:
    profile_size = int(df["profile_size"].max())
    subset = df[df["profile_size"] == profile_size].copy()
    subset = subset[~subset["profile_is_warmup"].astype(str).str.lower().isin({"true", "1"})]
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
        bad_group_cols = [
            col
            for col in ["dataset", "scenario", "profile_size", "profile_repeat"]
            if col in bad.columns
        ]
        bad_keys = bad[bad_group_cols].drop_duplicates()
        subset = subset.merge(
            bad_keys.assign(_bad_profile_group=True),
            on=bad_group_cols,
            how="left",
        )
        subset = subset[subset["_bad_profile_group"].isna()].drop(columns=["_bad_profile_group"])
    else:
        INVALID_OUTPUT_PATH.write_text("", encoding="utf-8")

    subset["encoder"] = subset["scenario"].map(ENCODER_LABELS)
    subset["mode_label"] = subset["mode"].map(MODE_LABELS)
    subset["topo_percent"] = 100.0 * subset["topo_fraction"]
    if subset["added_time_vs_baseline_sec"].isna().all():
        subset["added_time_vs_baseline_sec"] = pd.NA
        group_cols = [
            col
            for col in ["dataset", "scenario", "profile_size", "profile_repeat", "n_train_clips", "n_test_clips", "sequence_len"]
            if col in subset.columns
        ]
        for _, idx in subset.groupby(group_cols, sort=False, dropna=False).groups.items():
            group = subset.loc[idx]
            if not (group["mode"] == "z").any():
                continue
            baseline_time = float(group.loc[group["mode"] == "z", "wall_time_sec"].iloc[0])
            if baseline_time <= 0:
                continue
            subset.loc[idx, "added_time_vs_baseline_sec"] = subset.loc[idx, "wall_time_sec"] - baseline_time
            missing_overhead = subset.loc[idx, "overhead_vs_baseline"].isna()
            subset.loc[missing_overhead[missing_overhead].index, "overhead_vs_baseline"] = (
                subset.loc[missing_overhead[missing_overhead].index, "wall_time_sec"] / baseline_time
            )

    grouped = (
        subset.groupby(["encoder", "mode", "mode_label"], sort=False)
        .agg(
            runtime_mean=("wall_time_sec", "mean"),
            runtime_std=("wall_time_sec", "std"),
            cpu_mean=("peak_cpu_rss_mb", "mean"),
            cpu_std=("peak_cpu_rss_mb", "std"),
            accel_mean=("peak_accelerator_mem_mb", "mean"),
            accel_std=("peak_accelerator_mem_mb", "std"),
            added_mean=("added_time_vs_baseline_sec", "mean"),
            added_std=("added_time_vs_baseline_sec", "std"),
            overhead_mean=("overhead_vs_baseline", "mean"),
            overhead_std=("overhead_vs_baseline", "std"),
            encoding_time_mean=("encoding_time_sec", "mean"),
            encoding_time_std=("encoding_time_sec", "std"),
            persistence_time_mean=("persistence_time_sec", "mean"),
            persistence_time_std=("persistence_time_sec", "std"),
            fusion_time_mean=("fusion_time_sec", "mean"),
            fusion_time_std=("fusion_time_sec", "std"),
            predictor_time_mean=("predictor_time_sec", "mean"),
            predictor_time_std=("predictor_time_sec", "std"),
            n_datasets=("dataset", "nunique"),
        )
        .reset_index()
    )
    encoder_order = {name: idx for idx, name in enumerate(["AE", "GeoAE", "TopoAE"])}
    mode_order = {name: idx for idx, name in enumerate(MODE_LABELS)}
    grouped["encoder_order"] = grouped["encoder"].map(encoder_order)
    grouped["mode_order"] = grouped["mode"].map(mode_order)
    grouped = grouped.sort_values(["encoder_order", "mode_order"])

    def mean_std(mean: float, std: float, digits: int = 2) -> str:
        if pd.isna(mean):
            return "--"
        std = 0.0 if pd.isna(std) else std
        return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"

    def phase_value(mean: float, std: float) -> str:
        if pd.isna(mean) or mean <= 1e-6:
            return "--"
        return mean_std(mean, std, digits=2)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Profiling summary at profile size {profile_size}, averaged over datasets after discarding profiling warmup runs. Runtime, added time, peak CPU RSS, peak accelerator memory, and phase times are reported as mean$\pm$std across datasets and measured repeats. Added time and Runtime/\(z\) are relative to the corresponding \(z\)-only run within the same dataset, encoder family, and profiling repeat. Persistence is CPU-side in our implementation, so its overhead is reflected primarily in wall time and persistence phase time rather than accelerator memory.}}",
        r"\label{tab:profiling}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r" & & & & & \multicolumn{2}{c}{Peak memory (MB)} & \multicolumn{3}{c}{Phase time (s)} \\",
        r"\cmidrule(lr){6-7}\cmidrule(lr){8-10}",
        r"Encoder & Input mode & Runtime (s) & Added time (s) & Runtime/\(z\) & CPU RSS & Accel. & Enc. & Persist. & Pred. \\",
        r"\midrule",
    ]
    previous_encoder = None
    for row in grouped.itertuples(index=False):
        if previous_encoder is not None and row.encoder != previous_encoder:
            lines.append(r"\midrule")
        runtime = mean_std(row.runtime_mean, row.runtime_std, digits=2)
        added = mean_std(row.added_mean, row.added_std, digits=2)
        overhead = mean_std(row.overhead_mean, row.overhead_std, digits=2)
        cpu = mean_std(row.cpu_mean, row.cpu_std, digits=0)
        accel = "--" if pd.isna(row.accel_mean) else mean_std(row.accel_mean, row.accel_std, digits=0)
        encoding = phase_value(row.encoding_time_mean, row.encoding_time_std)
        persistence = phase_value(row.persistence_time_mean, row.persistence_time_std)
        predictor = phase_value(row.predictor_time_mean, row.predictor_time_std)
        lines.append(
            f"{row.encoder} & {row.mode_label} & {runtime} & {added} & {overhead} & {cpu} & {accel} & "
            f"{encoding} & {persistence} & {predictor} " + r"\\"
        )
        previous_encoder = row.encoder
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
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
