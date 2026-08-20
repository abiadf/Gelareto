#!/usr/bin/env python3
"""Download one or more YouTube videos, chop them into clips, and report E/S/H factors.

Examples:
  uv run python scripts/prepare_youtube_sequences.py \
    --video colony=https://youtu.be/VIDEO_ID --sequence-length 64 --num-sequences 30
  uv run python scripts/prepare_youtube_sequences.py --manifest videos.txt \
    --sequence-length 64 --num-sequences 30

Manifest lines use ``name<TAB>url`` or
``name<TAB>url<TAB>start<TAB>end``. Timestamps accept seconds, ``MM:SS``, or
``HH:MM:SS``. Blank/comment lines are ignored. Every prepared dataset is
available to the main runner as ``youtube_<name>`` in a new Python process.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from topo.ml_tda_youtube import (
    DEFAULT_ROOT,
    YouTubeSequenceBuilder,
    YouTubeSequenceSpec,
    diagnose_youtube_dataset,
    parse_timestamp,
)


def _parse_video(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--video must be NAME=URL")
    name, url = value.split("=", 1)
    return name.strip(), url.strip()


def _manifest_videos(path: Path) -> list[tuple[str, str, float | None, float | None]]:
    videos = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) not in (2, 4):
            raise ValueError(
                f"{path}:{line_number}: expected NAME<TAB>URL optionally followed by "
                "<TAB>START<TAB>END"
            )
        start = parse_timestamp(parts[2].strip()) if len(parts) == 4 else None
        end = parse_timestamp(parts[3].strip()) if len(parts) == 4 else None
        videos.append((parts[0].strip(), parts[1].strip(), start, end))
    return videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", type=_parse_video, default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--num-sequences", type=int, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--start", type=parse_timestamp,
        help="First timestamp to use: seconds, MM:SS, or HH:MM:SS",
    )
    parser.add_argument(
        "--end", type=parse_timestamp,
        help="Last timestamp to use: seconds, MM:SS, or HH:MM:SS",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--knn", default="4,8,12")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--triangle-samples", type=int, default=10_000)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--results", type=Path, default=Path("results/youtube_curvature.csv"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = [(name, source, args.start, args.end) for name, source in args.video]
    if args.manifest:
        videos.extend(_manifest_videos(args.manifest))
    if not videos:
        raise SystemExit("Provide at least one --video NAME=URL or --manifest")
    builder = YouTubeSequenceBuilder(args.output_root)
    frames = []
    for name, source, manifest_start, manifest_end in videos:
        start = manifest_start if manifest_start is not None else args.start
        end = manifest_end if manifest_end is not None else args.end
        interval = f" [{start or 0:g}s:{end if end is not None else 'end'}]"
        print(f"\nPreparing {name}: {source}{interval}")
        output_dir = builder.prepare(
            YouTubeSequenceSpec(
                name=name,
                source=source,
                sequence_length=args.sequence_length,
                num_sequences=args.num_sequences,
                sample_fps=args.sample_fps,
                image_size=args.image_size,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                start_seconds=start,
                end_seconds=end,
            ),
            overwrite=args.overwrite,
        )
        result = diagnose_youtube_dataset(
            output_dir,
            latent_dim=args.latent_dim,
            knn_values=[int(value) for value in args.knn.split(",")],
            seeds=[int(value) for value in args.seeds.split(",")],
            n_samples=args.triangle_samples,
        )
        frames.append(result)
        columns = [
            "dataset", "seed", "knn", "k_mean", "negative_fraction",
            "flat_fraction", "positive_fraction", "largest_component_fraction",
            "suggested_signature",
        ]
        print(result[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    combined = pd.concat(frames, ignore_index=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.results, index=False, float_format="%.6f")
    summary = combined.groupby("dataset").agg(
        hyperbolic_fraction=("negative_fraction", "mean"),
        spherical_fraction=("positive_fraction", "mean"),
        euclidean_fraction=("flat_fraction", "mean"),
        connectivity=("largest_component_fraction", "mean"),
    ).sort_values("hyperbolic_fraction", ascending=False)
    print("\nRanked summary (exploratory diagnostic):")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved {args.results}")


if __name__ == "__main__":
    main()
