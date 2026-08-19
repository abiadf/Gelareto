"""Load HIRROS root-growth TIFF stacks as forecasting-ready video clips.

Each plant is one 29-frame sequence. Raw train/test plants remain separate,
frames are aspect-preservingly resized to 64x64, and intensity scaling is fit
only on the training split. Processed uint8 tensors are cached for inexpensive
CPU experiments on laptops.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch


def _resize_with_padding(frame: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize a frame without distorting root angles or branch lengths."""
    target_h, target_w = map(int, image_size)
    height, width = frame.shape
    scale = min(target_h / height, target_w / width)
    resized_h = max(1, round(height * scale))
    resized_w = max(1, round(width * scale))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    output = np.full((target_h, target_w), float(np.median(frame)), dtype=np.float32)
    y0, x0 = (target_h - resized_h) // 2, (target_w - resized_w) // 2
    output[y0:y0 + resized_h, x0:x0 + resized_w] = resized
    return output


def _read_split(directory: Path, image_size: tuple[int, int], expected_frames: int) -> np.ndarray:
    """Read every valid plant stack and fail clearly if the core TIFF is damaged."""
    clips = []
    for plant_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        stack_path = plant_dir / "22_registered_stack.tif"
        if not stack_path.exists():
            print(f"Skipping HIRROS plant without registered stack: {plant_dir.name}")
            continue
        try:
            stack = np.asarray(tifffile.imread(stack_path))
        except Exception as exc:
            print(f"Skipping unreadable HIRROS stack {plant_dir.name}: {exc}")
            continue
        if stack.ndim != 3 or stack.shape[0] != int(expected_frames):
            print(
                f"Skipping malformed HIRROS stack {plant_dir.name}: "
                f"expected ({expected_frames},H,W), got {stack.shape}"
            )
            continue
        clips.append(np.stack([_resize_with_padding(frame, image_size) for frame in stack]))
    if not clips:
        raise ValueError(f"No valid HIRROS stacks found in {directory}")
    # Repository video convention: (time, clips, height, width).
    return np.stack(clips, axis=1).astype(np.float32)


def load_hirros(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load/cache HIRROS using its official plant-level train/test split."""
    cache_dir = Path(config.get("cache_dir", "datasets/2D/hirros/processed"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_size = tuple(config.get("image_size", (64, 64)))
    expected_frames = int(config.get("expected_frames", 29))
    cache_path = cache_dir / f"hirros_T{expected_frames}_H{image_size[0]}_W{image_size[1]}.pt"
    if cache_path.exists() and not config.get("force_rebuild_cache", False):
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return payload["train"].numpy(), payload["test"].numpy()

    train = _read_split(Path(config["train_dir"]), image_size, expected_frames)
    test = _read_split(Path(config["test_dir"]), image_size, expected_frames)
    # Robust train-only scaling protects against a few extreme sensor pixels.
    low, high = np.quantile(train, [0.01, 0.99])
    scale = max(float(high - low), 1e-8)
    train = np.clip((train - low) / scale, 0.0, 1.0)
    test = np.clip((test - low) / scale, 0.0, 1.0)
    payload = {
        "train": torch.from_numpy(np.rint(train * 255).astype(np.uint8)),
        "test": torch.from_numpy(np.rint(test * 255).astype(np.uint8)),
        "train_quantiles": (float(low), float(high)),
    }
    torch.save(payload, cache_path)
    print(f"Saved HIRROS cache: {cache_path} train={train.shape} test={test.shape}")
    return payload["train"].numpy(), payload["test"].numpy()
