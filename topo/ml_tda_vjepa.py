"""Frozen V-JEPA feature extraction for latent-TDA predictor benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from topo.utils import tqdm_progress_bar


DEFAULT_VJEPA_REPO = "facebook/vjepa2-vitl-fpc64-256"
DEFAULT_VJEPA_CACHE_DIR = Path("models") / "hf_vjepa"


def _repo_tag(repo: str) -> str:
    return repo.replace("/", "__").replace(":", "_")


def _to_uint8_rgb_video(frames: torch.Tensor) -> np.ndarray:
    """Convert (T, 1, H, W) or (T, C, H, W) frames to uint8 RGB video."""
    frames = frames.detach().cpu().float()
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape (T, C, H, W), got {tuple(frames.shape)}")
    if frames.shape[1] == 1:
        frames = frames.repeat(1, 3, 1, 1)
    elif frames.shape[1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got shape {tuple(frames.shape)}")
    if frames.max() <= 1.0:
        frames = frames * 255.0
    frames = frames.clamp(0, 255).byte()
    return frames.permute(0, 2, 3, 1).numpy()


def _window_for_time(video: torch.Tensor, t: int, num_frames: int) -> torch.Tensor:
    """Return a fixed-length window ending at t, padding by repeating the first frame."""
    start = max(0, t - num_frames + 1)
    window = video[start : t + 1]
    if window.shape[0] < num_frames:
        pad = window[:1].repeat(num_frames - window.shape[0], 1, 1, 1)
        window = torch.cat([pad, window], dim=0)
    return window


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _pool_model_output(output: Any) -> torch.Tensor:
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state
    elif isinstance(output, (tuple, list)) and output:
        hidden = output[0]
    else:
        raise ValueError("Could not find pooler_output or last_hidden_state in V-JEPA output.")
    if hidden.ndim < 2:
        raise ValueError(f"Unexpected V-JEPA hidden state shape: {tuple(hidden.shape)}")
    if hidden.ndim == 2:
        return hidden
    return hidden.reshape(hidden.shape[0], -1, hidden.shape[-1]).mean(dim=1)


def load_vjepa_model(repo: str = DEFAULT_VJEPA_REPO, device: str | torch.device = "auto"):
    """Load a frozen V-JEPA model and processor from the repo-local models cache."""
    try:
        from transformers import AutoModel, AutoVideoProcessor
    except ImportError as exc:
        raise ImportError(
            "V-JEPA benchmark requires transformers. Install it with `pip install transformers accelerate`."
        ) from exc

    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    cache_dir = DEFAULT_VJEPA_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    if device_obj.type == "cuda":
        model = AutoModel.from_pretrained(repo, device_map="auto", cache_dir=cache_dir)
    else:
        model = AutoModel.from_pretrained(repo, cache_dir=cache_dir)
        model.to(device_obj)
    processor = AutoVideoProcessor.from_pretrained(repo, cache_dir=cache_dir)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, processor, device_obj


@torch.no_grad()
def encode_video_tensor_with_vjepa(
    video_tensor: torch.Tensor,
    *,
    repo: str = DEFAULT_VJEPA_REPO,
    device: str | torch.device = "auto",
    batch_size: int = 2,
    num_frames: int = 16,
) -> torch.Tensor:
    """Encode (T, B, C, H, W) clips into V-JEPA embeddings shaped (T, B, D)."""
    if video_tensor.ndim != 5:
        raise ValueError(f"Expected video tensor (T, B, C, H, W), got {tuple(video_tensor.shape)}")
    model, processor, device_obj = load_vjepa_model(repo=repo, device=device)
    T, B = video_tensor.shape[:2]
    embeddings = []
    pending_videos = []
    pending_positions = []
    total = T * B

    def flush() -> None:
        if not pending_videos:
            return
        inputs = processor(videos=pending_videos, return_tensors="pt")
        inputs = _move_batch_to_device(inputs, device_obj)
        output = model(**inputs)
        pooled = _pool_model_output(output).detach().float().cpu()
        for (t_idx, b_idx), emb in zip(pending_positions, pooled):
            embeddings.append((t_idx, b_idx, emb))
        pending_videos.clear()
        pending_positions.clear()

    steps = ((t, b) for t in range(T) for b in range(B))
    for t, b in tqdm_progress_bar(steps, desc="Encode V-JEPA windows", total=total):
        window = _window_for_time(video_tensor[:, b], t, int(num_frames))
        pending_videos.append(_to_uint8_rgb_video(window))
        pending_positions.append((t, b))
        if len(pending_videos) >= int(batch_size):
            flush()
    flush()

    if not embeddings:
        raise ValueError("V-JEPA produced no embeddings.")
    dim = int(embeddings[0][2].numel())
    z = torch.zeros((T, B, dim), dtype=torch.float32)
    for t, b, emb in embeddings:
        z[t, b] = emb.reshape(-1)
    return z


def vjepa_feature_cache_path(
    *,
    namespace: str,
    repo: str,
    seed: int,
    split_name: str,
    video_tensor: torch.Tensor,
    num_frames: int,
) -> Path:
    cache_dir = Path("models") / namespace / "vjepa_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"{_repo_tag(repo)}_seed{seed}_{split_name}_T{video_tensor.shape[0]}_B{video_tensor.shape[1]}_"
        f"H{video_tensor.shape[-2]}_W{video_tensor.shape[-1]}_frames{num_frames}"
    )
    return cache_dir / f"{tag}.pt"
