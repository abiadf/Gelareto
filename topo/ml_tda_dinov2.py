"""Frozen DINOv2 frame encoder for latent-TDA experiments."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


DEFAULT_DINOV2_REPO = "facebook/dinov2-small"


def _load_transformers_model(repo: str, device: str | torch.device):
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise ImportError(
            "Install transformers to use dinov2_latent_tda, e.g. `uv add transformers`."
        ) from exc

    device_obj = torch.device(device)
    config = AutoConfig.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).to(device_obj).eval()
    return model, config, device_obj


def _normalization_tensors(device: torch.device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return mean, std


def _prepare_frames(frames: torch.Tensor, *, image_size: int, device: torch.device) -> torch.Tensor:
    x = frames.to(device=device, dtype=torch.float32)
    if x.max() > 1.5:
        x = x / 255.0
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] != 3:
        raise ValueError(f"DINOv2 expects 1 or 3 channels, got shape {tuple(x.shape)}")
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean, std = _normalization_tensors(device)
    return (x.clamp(0.0, 1.0) - mean) / std


@torch.no_grad()
def encode_video_tensor_with_dinov2(
    video_tensor: torch.Tensor,
    *,
    repo: str = DEFAULT_DINOV2_REPO,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    image_size: int = 224,
) -> torch.Tensor:
    """Encode each frame independently and return z with shape (T, B, D)."""

    model, _, device_obj = _load_transformers_model(repo, device)
    if video_tensor.ndim != 5:
        raise ValueError(f"Expected video tensor (T, B, C, H, W), got {tuple(video_tensor.shape)}")
    t_steps, batch = int(video_tensor.shape[0]), int(video_tensor.shape[1])
    frames = video_tensor.permute(1, 0, 2, 3, 4).reshape(batch * t_steps, *video_tensor.shape[2:])
    features = []
    for start in range(0, len(frames), int(batch_size)):
        pixel_values = _prepare_frames(frames[start:start + int(batch_size)], image_size=image_size, device=device_obj)
        outputs = model(pixel_values=pixel_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            z = outputs.pooler_output
        else:
            z = outputs.last_hidden_state[:, 0]
        features.append(z.detach().cpu())
    flat_z = torch.cat(features, dim=0)
    return flat_z.reshape(batch, t_steps, -1).permute(1, 0, 2).contiguous()


def dinov2_feature_cache_path(
    *,
    namespace: str,
    repo: str,
    seed: int,
    split_name: str,
    video_tensor: torch.Tensor,
    image_size: int,
) -> Path:
    cache_dir = Path("models") / namespace / "dinov2_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_tag = repo.replace("/", "__")
    tag = (
        f"{repo_tag}_seed{seed}_{split_name}_T{video_tensor.shape[0]}_B{video_tensor.shape[1]}_"
        f"H{video_tensor.shape[-2]}_W{video_tensor.shape[-1]}_img{image_size}"
    )
    return cache_dir / f"{tag}.pt"
