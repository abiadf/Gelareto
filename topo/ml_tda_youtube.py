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
from scipy.sparse import csr_matrix
from skimage.morphology import closing, disk, remove_small_objects, skeletonize

from topo.ml_tda_triangle import (
    _diagnostic_summary,
    gu_triangle_curvature_samples_from_adjacency,
    sequence_triangle_curvature_diagnostic,
)


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
            "start_seconds": spec.start_seconds,
            "end_seconds": spec.end_seconds,
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
            # Audio is unnecessary and many YouTube videos expose no combined
            # MP4 stream. Prefer an OpenCV-friendly video-only MP4, then fall
            # back to any video stream without requiring ffmpeg to merge audio.
            "format": (
                "bestvideo[ext=mp4][height<=480]/"
                "bestvideo[height<=480]/bestvideo/best"
            ),
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


def _branch_mask(frame: np.ndarray) -> tuple[np.ndarray, str]:
    """Extract a conservative foreground mask, automatically choosing polarity."""
    image = np.asarray(frame, dtype=np.uint8)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    low, background, high = np.percentile(blurred, (1, 50, 99))
    candidates = {
        "bright": blurred > background + 0.06 * max(1.0, high - background),
        "dark": blurred < background - 0.06 * max(1.0, background - low),
    }

    def score(mask: np.ndarray) -> float:
        fraction = float(mask.mean())
        border = np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1]))
        # Tree-like foregrounds are sparse and usually do not fill the border.
        invalid = 10.0 if not 0.005 <= fraction <= 0.55 else 0.0
        return invalid + abs(fraction - 0.15) + 0.75 * float(border.mean())

    polarity = min(candidates, key=lambda key: score(candidates[key]))
    mask = candidates[polarity].copy()
    # Suppress player chrome, captions, and hard video borders.
    margin = max(1, int(round(min(mask.shape) * 0.03)))
    mask[:margin] = mask[-margin:] = False
    mask[:, :margin] = mask[:, -margin:] = False
    mask = remove_small_objects(mask, max_size=7)
    mask = closing(mask, disk(1))
    return np.asarray(mask, dtype=bool), polarity


def _skeleton_adjacency(frame: np.ndarray, max_nodes: int = 700) -> tuple[csr_matrix, dict]:
    """Convert one image to an 8-neighbour skeleton graph with a safe node cap."""
    working = np.asarray(frame, dtype=np.uint8)
    polarity = "unknown"
    for _ in range(4):
        mask, polarity = _branch_mask(working)
        skeleton = skeletonize(mask)
        coordinates = np.argwhere(skeleton)
        if len(coordinates) <= max_nodes or min(working.shape) <= 24:
            break
        scale = max(0.6, np.sqrt(max_nodes / len(coordinates)))
        height = max(24, int(round(working.shape[0] * scale)))
        width = max(24, int(round(working.shape[1] * scale)))
        working = cv2.resize(working, (width, height), interpolation=cv2.INTER_AREA)
    if len(coordinates) < 3:
        raise ValueError("Fewer than three skeleton pixels survived segmentation")
    index = {tuple(point): i for i, point in enumerate(coordinates)}
    rows, cols = [], []
    for node, (row, col) in enumerate(coordinates):
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                if d_row == d_col == 0:
                    continue
                neighbour = index.get((row + d_row, col + d_col))
                if neighbour is not None and neighbour > node:
                    rows.extend((node, neighbour))
                    cols.extend((neighbour, node))
    pixel_adjacency = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(coordinates), len(coordinates)),
    )
    degrees = np.diff(pixel_adjacency.indptr)
    adjacency = _contract_degree_two(pixel_adjacency)
    return adjacency, {
        "skeleton_nodes": int(len(coordinates)),
        "contracted_nodes": int(adjacency.shape[0]),
        "skeleton_endpoints": int(np.sum(degrees == 1)),
        "skeleton_junctions": int(np.sum(degrees >= 3)),
        "foreground_polarity": polarity,
        "skeleton_height": int(working.shape[0]),
        "skeleton_width": int(working.shape[1]),
    }


def _contract_degree_two(adjacency: csr_matrix) -> csr_matrix:
    """Collapse pixel chains and connected junction blobs to morphological edges."""
    neighbours = [
        adjacency.indices[adjacency.indptr[i] : adjacency.indptr[i + 1]]
        for i in range(adjacency.shape[0])
    ]
    degrees = np.asarray([len(values) for values in neighbours])
    junctions = set(np.flatnonzero(degrees >= 3).tolist())
    node_to_super: dict[int, int] = {}
    # A thick rasterized junction often spans several adjacent pixels. Treat
    # each connected junction blob as one morphological branch point.
    unseen = set(junctions)
    super_count = 0
    while unseen:
        stack = [unseen.pop()]
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            linked = junctions.intersection(map(int, neighbours[node])) & unseen
            unseen.difference_update(linked)
            stack.extend(linked)
        for node in component:
            node_to_super[node] = super_count
        super_count += 1
    for node in np.flatnonzero(degrees <= 1):
        node_to_super[int(node)] = super_count
        super_count += 1
    if super_count < 3:
        return adjacency
    key_set = set(node_to_super)
    visited, edges, duplicates = set(), set(), []
    for start in key_set:
        for neighbour in neighbours[start]:
            if int(neighbour) in key_set and node_to_super[int(neighbour)] == node_to_super[start]:
                visited.add(tuple(sorted((start, int(neighbour)))))
                continue
            first_edge = tuple(sorted((start, int(neighbour))))
            if first_edge in visited:
                continue
            previous, current = start, int(neighbour)
            visited.add(first_edge)
            while current not in key_set:
                onward = [int(node) for node in neighbours[current] if int(node) != previous]
                if not onward:
                    break
                following = onward[0]
                visited.add(tuple(sorted((current, following))))
                previous, current = current, following
            if current in key_set and current != start:
                edge = tuple(sorted((node_to_super[start], node_to_super[current])))
                if edge[0] == edge[1]:
                    continue
                if edge in edges:
                    duplicates.append(edge)
                else:
                    edges.add(edge)
    if not edges:
        return adjacency
    rows, cols = [], []
    for left, right in edges:
        rows.extend((left, right))
        cols.extend((right, left))
    # Preserve genuine parallel paths (loops) with a midpoint instead of
    # silently collapsing them into one sparse-matrix edge.
    next_node = super_count
    for left, right in duplicates:
        rows.extend((left, next_node, next_node, right))
        cols.extend((next_node, left, right, next_node))
        next_node += 1
    return csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(next_node, next_node),
    )


def diagnose_youtube_skeletons(
    dataset_dir: str | Path,
    *,
    latent_dim: int = 16,
    seeds: Iterable[int] = (0, 1, 2),
    n_samples: int = 10_000,
    max_frames: int = 24,
    max_nodes: int = 700,
    source_resolution: int = 160,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    anchor_distance_quantile: float = 0.0,
) -> pd.DataFrame:
    """Diagnose spatial skeleton graphs from evenly sampled training frames.

    This is intentionally a separate diagnostic from raw-frame trajectory
    curvature. It asks whether visible morphology is tree-like, not whether the
    sequence of whole images follows a hyperbolic metric.
    """
    dataset_dir = Path(dataset_dir)
    source_path = dataset_dir / "source.mp4"
    if source_path.exists():
        metadata_path = dataset_dir / "metadata.json"
        stored = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        start = start_seconds if start_seconds is not None else stored.get("start_seconds")
        end = end_seconds if end_seconds is not None else stored.get("end_seconds")
        capture = cv2.VideoCapture(str(source_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        duration = float(capture.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
        start = float(start or 0.0)
        end = min(duration, float(end)) if end is not None else duration
        times = np.linspace(start, end, max_frames + 2)[1:-1]
        sampled = []
        for timestamp in times:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(timestamp * fps)))
            ok, frame = capture.read()
            if ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sampled.append(cv2.resize(gray, (source_resolution, source_resolution), interpolation=cv2.INTER_AREA))
        capture.release()
        frames = np.asarray(sampled, dtype=np.uint8)
    else:
        with np.load(dataset_dir / "dataset.npz") as payload:
            clips = payload["train"]
        frames = clips.reshape(-1, *clips.shape[-2:])
    indices = np.linspace(0, len(frames) - 1, min(max_frames, len(frames)), dtype=int)
    graphs, graph_metadata = [], []
    for index in np.unique(indices):
        try:
            adjacency, metadata = _skeleton_adjacency(frames[index], max_nodes=max_nodes)
        except ValueError:
            continue
        graphs.append(adjacency)
        graph_metadata.append(metadata)
    if not graphs:
        raise ValueError(f"No usable skeleton graphs were extracted from {dataset_dir.name}")

    rows = []
    samples_per_graph = max(1, int(np.ceil(n_samples / len(graphs))))
    for seed in seeds:
        values, retained = [], []
        for graph_index, adjacency in enumerate(graphs):
            try:
                graph_values, metadata = gu_triangle_curvature_samples_from_adjacency(
                    adjacency,
                    n_samples=samples_per_graph,
                    seed=int(seed) * 1009 + graph_index,
                    anchor_distance_quantile=anchor_distance_quantile,
                )
            except ValueError:
                continue
            values.append(graph_values)
            retained.append(metadata["largest_component_fraction"])
        if not values:
            raise ValueError(f"No skeleton graph contained a usable branching component: {dataset_dir.name}")
        pooled = np.concatenate(values)[:n_samples]
        row = _diagnostic_summary(
            pooled,
            {
                "n_input_points": int(sum(item["skeleton_nodes"] for item in graph_metadata)),
                "n_graph_points": int(sum(item["skeleton_nodes"] for item in graph_metadata)),
                "n_components": int(len(graphs)),
                "largest_component_fraction": float(np.mean(retained)),
            },
            latent_dim=latent_dim,
            flat_threshold=1e-6,
            seed=int(seed),
            knn=0,
        )
        row.update({
            "dataset": dataset_dir.name,
            "diagnostic_space": "spatial_skeleton_graph",
            "n_skeleton_frames": len(graphs),
            "median_skeleton_nodes": float(np.median([m["skeleton_nodes"] for m in graph_metadata])),
            "median_contracted_nodes": float(np.median([m["contracted_nodes"] for m in graph_metadata])),
            "median_skeleton_endpoints": float(np.median([m["skeleton_endpoints"] for m in graph_metadata])),
            "median_skeleton_junctions": float(np.median([m["skeleton_junctions"] for m in graph_metadata])),
            "anchor_distance_quantile": float(anchor_distance_quantile),
        })
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
