"""Frozen CLIP vision encoder for latent-TDA experiments."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


DEFAULT_CLIP_REPO = "openai/clip-vit-base-patch32"


def _load_clip_model(repo: str, device: str | torch.device):
    try:
        from transformers import CLIPVisionModel
    except ImportError as exc:
        raise ImportError("Install transformers to use clip_latent_tda, e.g. `uv add transformers`.") from exc

    device_obj = torch.device(device)
    model = CLIPVisionModel.from_pretrained(repo).to(device_obj).eval()
    return model, device_obj


def _clip_normalization_tensors(device: torch.device):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
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
        raise ValueError(f"CLIP expects 1 or 3 channels, got shape {tuple(x.shape)}")
    x = F.interpolate(x, size=(image_size, image_size), mode="bicubic", align_corners=False)
    mean, std = _clip_normalization_tensors(device)
    return (x.clamp(0.0, 1.0) - mean) / std


@torch.no_grad()
def encode_video_tensor_with_clip(
    video_tensor: torch.Tensor,
    *,
    repo: str = DEFAULT_CLIP_REPO,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    image_size: int = 224,
) -> torch.Tensor:
    """Encode each frame independently and return z with shape (T, B, D)."""

    model, device_obj = _load_clip_model(repo, device)
    if video_tensor.ndim != 5:
        raise ValueError(f"Expected video tensor (T, B, C, H, W), got {tuple(video_tensor.shape)}")
    t_steps, batch = int(video_tensor.shape[0]), int(video_tensor.shape[1])
    frames = video_tensor.permute(1, 0, 2, 3, 4).reshape(batch * t_steps, *video_tensor.shape[2:])
    features = []
    for start in range(0, len(frames), int(batch_size)):
        pixel_values = _prepare_frames(frames[start:start + int(batch_size)], image_size=image_size, device=device_obj)
        outputs = model(pixel_values=pixel_values)
        z = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
        features.append(z.detach().cpu())
    flat_z = torch.cat(features, dim=0)
    return flat_z.reshape(batch, t_steps, -1).permute(1, 0, 2).contiguous()


def clip_feature_cache_path(
    *,
    namespace: str,
    repo: str,
    seed: int,
    split_name: str,
    video_tensor: torch.Tensor,
    image_size: int,
) -> Path:
    cache_dir = Path("models") / namespace / "clip_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_tag = repo.replace("/", "__")
    tag = (
        f"{repo_tag}_seed{seed}_{split_name}_T{video_tensor.shape[0]}_B{video_tensor.shape[1]}_"
        f"H{video_tensor.shape[-2]}_W{video_tensor.shape[-1]}_img{image_size}"
    )
    return cache_dir / f"{tag}.pt"
