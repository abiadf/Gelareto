"""Frozen DINOv2 frame encoder for latent-TDA experiments."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


DEFAULT_DINOV2_REPO = "facebook/dinov2-small"


class Dinov2FramePredictor(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=False)
        self.head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(z_seq)
        return self.head(out)


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


def _set_trainable_last_blocks(model: nn.Module, trainable_blocks: int) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if trainable_blocks <= 0:
        return
    layers = getattr(getattr(model, "encoder", None), "layer", None)
    if layers is None:
        raise AttributeError("Could not find transformer blocks at model.encoder.layer for DINOv2.")
    for layer in layers[-int(trainable_blocks):]:
        for param in layer.parameters():
            param.requires_grad = True
    for name in ("layernorm", "norm"):
        module = getattr(model, name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


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


def _encode_video_tensor_with_model(
    model: nn.Module,
    video_tensor: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    image_size: int,
    detach_cpu: bool,
) -> torch.Tensor:
    if video_tensor.ndim != 5:
        raise ValueError(f"Expected video tensor (T, B, C, H, W), got {tuple(video_tensor.shape)}")
    t_steps, batch = int(video_tensor.shape[0]), int(video_tensor.shape[1])
    frames = video_tensor.permute(1, 0, 2, 3, 4).reshape(batch * t_steps, *video_tensor.shape[2:])
    features = []
    for start in range(0, len(frames), int(batch_size)):
        pixel_values = _prepare_frames(frames[start:start + int(batch_size)], image_size=image_size, device=device)
        outputs = model(pixel_values=pixel_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            z = outputs.pooler_output
        else:
            z = outputs.last_hidden_state[:, 0]
        features.append(z.detach().cpu() if detach_cpu else z)
    flat_z = torch.cat(features, dim=0)
    return flat_z.reshape(batch, t_steps, -1).permute(1, 0, 2).contiguous()


def finetuned_encoder_cache_path(
    *,
    namespace: str,
    repo: str,
    seed: int,
    video_tensor: torch.Tensor,
    image_size: int,
    trainable_blocks: int,
    epochs: int,
    encoder_lr: float,
) -> Path:
    cache_dir = Path("models") / namespace / "dinov2_finetuned"
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_tag = repo.replace("/", "__")
    lr_tag = f"{encoder_lr:g}".replace(".", "p").replace("-", "m")
    tag = (
        f"{repo_tag}_seed{seed}_T{video_tensor.shape[0]}_B{video_tensor.shape[1]}_"
        f"H{video_tensor.shape[-2]}_W{video_tensor.shape[-1]}_img{image_size}_"
        f"blocks{trainable_blocks}_epochs{epochs}_elr{lr_tag}"
    )
    return cache_dir / f"encoder_{tag}.pt"


def load_or_finetune_dinov2_model(
    train_tensor: torch.Tensor,
    *,
    namespace: str,
    repo: str,
    seed: int,
    device: str | torch.device,
    batch_size: int,
    image_size: int,
    trainable_blocks: int,
    epochs: int,
    encoder_lr: float,
    predictor_lr: float,
    hidden_dim: int,
    horizon: int,
    clip_batch_size: int = 8,
    retrain: bool = False,
) -> tuple[nn.Module, Path]:
    model, _, device_obj = _load_transformers_model(repo, device)
    path = finetuned_encoder_cache_path(
        namespace=namespace,
        repo=repo,
        seed=seed,
        video_tensor=train_tensor,
        image_size=image_size,
        trainable_blocks=trainable_blocks,
        epochs=epochs,
        encoder_lr=encoder_lr,
    )
    if path.exists() and not retrain:
        print(f"Loading fine-tuned DINOv2 encoder: {path}")
        model.load_state_dict(torch.load(path, map_location=device_obj))
        model.to(device_obj).eval()
        return model, path

    _set_trainable_last_blocks(model, trainable_blocks)
    model.train()
    latent_dim = int(getattr(model.config, "hidden_size", 384))
    predictor = Dinov2FramePredictor(latent_dim=latent_dim, hidden_dim=int(hidden_dim)).to(device_obj)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": trainable_params, "lr": float(encoder_lr)},
            {"params": predictor.parameters(), "lr": float(predictor_lr)},
        ],
        weight_decay=1e-4,
    )
    criterion = nn.MSELoss()
    n_clips = int(train_tensor.shape[1])
    clip_batch_size = max(1, min(int(clip_batch_size), n_clips))
    generator = torch.Generator().manual_seed(int(seed))
    print(
        f"Fine-tuning DINOv2 last {trainable_blocks} block(s): {path} | "
        f"epochs={epochs}, encoder_lr={encoder_lr:g}, predictor_lr={predictor_lr:g}"
    )
    for epoch in range(1, int(epochs) + 1):
        perm = torch.randperm(n_clips, generator=generator)
        total = 0.0
        total_seen = 0
        for start in range(0, n_clips, clip_batch_size):
            idx = perm[start:start + clip_batch_size]
            clips = train_tensor[:, idx]
            z = _encode_video_tensor_with_model(
                model,
                clips,
                device=device_obj,
                batch_size=batch_size,
                image_size=image_size,
                detach_cpu=False,
            )
            source = z[:-int(horizon)]
            target = z[int(horizon):].detach()
            pred = predictor(source)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * int(idx.numel())
            total_seen += int(idx.numel())
        print(f"DINOv2 fine-tune epoch {epoch}/{epochs} | z-only train MSE={total / max(total_seen, 1):.6f}")
    torch.save(model.state_dict(), path)
    model.eval()
    return model, path


@torch.no_grad()
def encode_video_tensor_with_dinov2_model(
    model: nn.Module,
    video_tensor: torch.Tensor,
    *,
    device: str | torch.device,
    batch_size: int,
    image_size: int,
) -> torch.Tensor:
    device_obj = torch.device(device)
    model.to(device_obj).eval()
    return _encode_video_tensor_with_model(
        model,
        video_tensor,
        device=device_obj,
        batch_size=batch_size,
        image_size=image_size,
        detach_cpu=True,
    )


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
