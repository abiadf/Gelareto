"""Prepare video links as leak-free sequence datasets and diagnose curvature.

The downloader is deliberately separate from the dataset loader: downloading is
an explicit preprocessing step, while experiments consume only a small,
reproducible ``dataset.npz`` plus metadata.  Curvature is computed on an
independent kNN graph per sequence so artificial edges are never introduced
between distant clips from the same (or different) source videos.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from topo.ml_tda_triangle import sequence_triangle_curvature_diagnostic


DEFAULT_ROOT = Path("datasets/2D/youtube_sequences")


def safe_dataset_name(value: str) -> str:
    """Return a stable lowercase identifier suitable for paths and CLI keys."""
    name = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not name:
        raise ValueError("Dataset name must contain a letter or number")
    return name


def parse_timestamp(value: str | float | int | None) -> float | None:
    """Convert seconds, ``MM:SS``, or ``HH:MM:SS`` to seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, (float, int)):
        seconds = float(value)
    else:
        parts = str(value).strip().split(":")
        if len(parts) > 3:
            raise ValueError(f"Invalid timestamp: {value!r}")
        try:
            numbers = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp: {value!r}") from exc
        seconds = sum(number * (60 ** power) for power, number in enumerate(reversed(numbers)))
    if seconds < 0:
        raise ValueError("Timestamps cannot be negative")
    return seconds


@dataclass(frozen=True)
class YouTubeSequenceSpec:
    """Parameters controlling temporal sampling and non-overlapping windows."""

    name: str
    source: str
    sequence_length: int
    num_sequences: int
    sample_fps: float = 2.0
    image_size: int = 64
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    start_seconds: float | None = None
    end_seconds: float | None = None

    def validate(self) -> None:
        if self.sequence_length < 3:
            raise ValueError("sequence_length must be at least 3")
        if self.num_sequences < 3:
            raise ValueError("num_sequences must be at least 3 for train/val/test splits")
        if self.sample_fps <= 0 or self.image_size <= 0:
            raise ValueError("sample_fps and image_size must be positive")
        if not 0 < self.train_fraction < 1 or not 0 <= self.val_fraction < 1:
            raise ValueError("Invalid train/validation fractions")
        if self.train_fraction + self.val_fraction >= 1:
            raise ValueError("train_fraction + val_fraction must be below 1")
        if self.end_seconds is not None and self.start_seconds is not None:
            if self.end_seconds <= self.start_seconds:
                raise ValueError("end_seconds must be greater than start_seconds")


class YouTubeSequenceBuilder:
    """Download one video and convert it to chronological, disjoint clips."""

    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self.root = Path(root)

    def prepare(self, spec: YouTubeSequenceSpec, *, overwrite: bool = False) -> Path:
        """Create ``dataset.npz`` and ``metadata.json`` and return their directory."""
        spec.validate()
        name = safe_dataset_name(spec.name)
        output_dir = self.root / name
        dataset_path = output_dir / "dataset.npz"
        if dataset_path.exists() and not overwrite:
            raise FileExistsError(f"{dataset_path} exists; pass --overwrite to replace it")
        output_dir.mkdir(parents=True, exist_ok=True)

        video_path, source_metadata = self._resolve_source(spec.source, output_dir, overwrite)
        frames, video_metadata = self._decode_frames(video_path, spec)
        clips, starts = self._make_disjoint_clips(frames, spec)
        train, val, test = self._chronological_split(clips, spec)
        np.savez_compressed(dataset_path, train=train, val=val, test=test)

        metadata = {
            "dataset_key": f"youtube_{name}",
            "name": name,
            "source": spec.source,
            "sequence_length": spec.sequence_length,
            "num_sequences": spec.num_sequences,
            "sample_fps": spec.sample_fps,
            "image_size": [spec.image_size, spec.image_size],
            "train_fraction": spec.train_fraction,
            "val_fraction": spec.val_fraction,
            "clip_start_sample_indices": starts.tolist(),
            "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
            **source_metadata,
            **video_metadata,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return output_dir

    @staticmethod
    def _resolve_source(source: str, output_dir: Path, overwrite: bool) -> tuple[Path, dict]:
        local = Path(source).expanduser()
        if local.exists():
            return local, {"source_type": "local", "source_path": str(local.resolve())}
        if not source.startswith(("http://", "https://")):
            raise ValueError(f"Source is neither an existing file nor an HTTP(S) URL: {source}")
        try:
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError("Install yt-dlp first: uv add yt-dlp") from exc

        target = output_dir / "source.mp4"
        if target.exists() and overwrite:
            target.unlink()
        options = {
            "outtmpl": str(target),
            "format": "best[ext=mp4][height<=720]/best[height<=720]/best",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "overwrites": overwrite,
        }
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(source, download=True)
        if not target.exists():
            candidates = sorted(output_dir.glob("source.*"))
            if not candidates:
                raise RuntimeError("yt-dlp completed but no video file was produced")
            target = candidates[0]
        return target, {
            "source_type": "youtube",
            "video_id": info.get("id"),
            "video_title": info.get("title"),
            "webpage_url": info.get("webpage_url", source),
            "license": info.get("license"),
        }

    @staticmethod
    def _decode_frames(path: Path, spec: YouTubeSequenceSpec) -> tuple[np.ndarray, dict]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open {path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
        step = max(1, int(round(source_fps / spec.sample_fps)))
        start_frame = int(round((spec.start_seconds or 0.0) * source_fps))
        end_frame = (
            int(round(spec.end_seconds * source_fps))
            if spec.end_seconds is not None else None
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames, frame_index = [], start_frame
        while True:
            ok, frame = capture.read()
            if not ok or (end_frame is not None and frame_index >= end_frame):
                break
            if (frame_index - start_frame) % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(
                    gray, (spec.image_size, spec.image_size), interpolation=cv2.INTER_AREA
                )
                frames.append(gray)
            frame_index += 1
        capture.release()
        required = spec.sequence_length * spec.num_sequences
        if len(frames) < required:
            available_seconds = len(frames) / spec.sample_fps
            raise ValueError(
                f"Only {len(frames)} sampled frames ({available_seconds:.1f}s) are available; "
                f"need {required}. Lower --sample-fps, --sequence-length, or --num-sequences."
            )
        array = np.asarray(frames, dtype=np.uint8)
        adjacent_change = np.mean(np.abs(np.diff(array.astype(np.float32), axis=0)), axis=(1, 2))
        return array, {
            "source_fps": source_fps,
            "effective_sample_fps": source_fps / step,
            "n_sampled_frames": len(array),
            "adjacent_change_median": float(np.median(adjacent_change)),
            "near_duplicate_fraction": float(np.mean(adjacent_change < 0.25)),
        }

    @staticmethod
    def _make_disjoint_clips(
        frames: np.ndarray, spec: YouTubeSequenceSpec
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select evenly distributed windows that never overlap."""
        length, count = spec.sequence_length, spec.num_sequences
        slack = len(frames) - count * length
        gaps = np.linspace(0, slack, count, dtype=int)
        starts = np.arange(count) * length + gaps
        clips = np.stack([frames[start : start + length] for start in starts])
        return clips, starts

    @staticmethod
    def _chronological_split(
        clips: np.ndarray, spec: YouTubeSequenceSpec
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(clips)
        train_end = max(1, int(np.floor(n * spec.train_fraction)))
        val_end = max(train_end + 1, int(np.floor(n * (spec.train_fraction + spec.val_fraction))))
        val_end = min(val_end, n - 1)
        return clips[:train_end], clips[train_end:val_end], clips[val_end:]


def load_youtube_sequences(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load prepared clips in the pipeline's ``(time, cases, H, W)`` convention."""
    path = Path(config["path"])
    with np.load(path) as payload:
        train = payload["train"].astype(np.float32) / 255.0
        # Validation remains isolated for later model selection; final pipeline
        # evaluation uses only the chronological test clips.
        test = payload["test"].astype(np.float32) / 255.0
    return train.transpose(1, 0, 2, 3), test.transpose(1, 0, 2, 3)


def diagnose_youtube_dataset(
    dataset_dir: str | Path,
    *,
    latent_dim: int = 16,
    knn_values: Iterable[int] = (4, 8, 12),
    seeds: Iterable[int] = (0, 1, 2),
    n_samples: int = 10_000,
) -> pd.DataFrame:
    """Return per-seed Gu-style E/S/H diagnostics for prepared training clips."""
    dataset_dir = Path(dataset_dir)
    with np.load(dataset_dir / "dataset.npz") as payload:
        train = payload["train"].astype(np.float32) / 255.0
    x = train.transpose(1, 0, 2, 3)
    rows = []
    for seed in seeds:
        for knn in knn_values:
            row = sequence_triangle_curvature_diagnostic(
                x,
                latent_dim=latent_dim,
                knn=int(knn),
                n_samples=n_samples,
                seed=int(seed),
            )
            row["dataset"] = dataset_dir.name
            rows.append(row)
    return pd.DataFrame(rows)


def discover_youtube_dataset_configs(root: str | Path = DEFAULT_ROOT) -> dict[str, dict]:
    """Discover prepared datasets so ``--dataset youtube_<name>`` works automatically."""
    root = Path(root)
    configs = {}
    if not root.exists():
        return configs
    for metadata_path in sorted(root.glob("*/metadata.json")):
        dataset_path = metadata_path.parent / "dataset.npz"
        if not dataset_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        key = metadata.get("dataset_key", f"youtube_{metadata_path.parent.name}")
        configs[key] = {
            "kind": "youtube_sequences",
            "path": str(dataset_path),
            "metadata_path": str(metadata_path),
            "image_size": tuple(metadata.get("image_size", [64, 64])),
            "LATENT_DIM": 16,
            "HIDDEN_DIM": 128,
            "PREDICTOR_EPOCHS": 40,
            "HORIZON": 5,
            "LATENT_TDA_WINDOW": min(15, int(metadata["sequence_length"])),
            "LATENT_TDA_BINS": 16,
            "RUN_SEEDS": list(range(3)),
        }
    return configs
