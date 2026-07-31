"""Clip-level classification from latent trajectories and persistence summaries."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


VALID_MODES = {
    "z", "h0", "h1", "both", "z_h0", "z_h1", "z_both",
    "z_h0_shuffle", "z_h1_shuffle",
}


def labels_from_sources(config: dict, split: str, task: str) -> torch.Tensor:
    direct_key = f"_classification_{split}_labels"
    if direct_key in config:
        return torch.tensor(config[direct_key], dtype=torch.long)
    if task not in {"motion", "object"}:
        raise ValueError(f"Unknown classification task: {task}")
    counts = config[f"_classification_{split}_counts"]
    labels = []
    for source, count in zip(config["sources"], counts):
        labels.extend([int(source[f"{task}_label"])] * int(count))
    return torch.tensor(labels, dtype=torch.long)


def _temporal_summary(x: torch.Tensor, start: int) -> torch.Tensor:
    """Fixed-width per-clip summary retaining level and temporal variation."""
    x = x[start:].float()
    if x.shape[0] == 0:
        raise ValueError("Persistence window is longer than the sequence.")
    delta = x[1:] - x[:-1] if x.shape[0] > 1 else torch.zeros_like(x)
    return torch.cat(
        [x.mean(0), x.std(0, unbiased=False), x.amax(0), delta.abs().mean(0)],
        dim=-1,
    )


def clip_features(payload: dict, mode: str, window: int, seed: int = 0) -> torch.Tensor:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown classification mode {mode!r}; valid={sorted(VALID_MODES)}")
    start = max(int(window) - 1, 0)
    z = _temporal_summary(payload["z"], start)
    h0 = _temporal_summary(payload["h0"], start)
    h1 = _temporal_summary(payload["h1"], start)
    if mode == "z":
        return z
    if mode == "h0":
        return h0
    if mode == "h1":
        return h1
    if mode == "both":
        return torch.cat([h0, h1], dim=-1)
    if mode == "z_h0":
        return torch.cat([z, h0], dim=-1)
    if mode == "z_h1":
        return torch.cat([z, h1], dim=-1)
    if mode == "z_both":
        return torch.cat([z, h0, h1], dim=-1)
    generator = torch.Generator().manual_seed(int(seed))
    if mode == "z_h0_shuffle":
        shuffled = h0[torch.randperm(len(h0), generator=generator)]
    else:
        shuffled = h1[torch.randperm(len(h1), generator=generator)]
    return torch.cat([z, shuffled], dim=-1)


def paired_signflip_test(differences) -> float:
    """Exact one-sided paired test for the alternative mean(difference) > 0."""
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return float("nan")
    observed = float(differences.mean())
    exceedances = 0
    total = 1 << int(differences.size)
    for mask in range(total):
        signs = np.asarray(
            [1.0 if mask & (1 << idx) else -1.0 for idx in range(differences.size)]
        )
        exceedances += float((differences * signs).mean()) >= observed - 1e-15
    return float(exceedances / total)


class ClipClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_evaluate_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    seed: int,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(int(seed))
    mean = train_x.mean(0, keepdim=True)
    std = train_x.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train_x = ((train_x - mean) / std).to(device)
    test_x = ((test_x - mean) / std).to(device)
    train_y = train_y.to(device)
    test_y = test_y.to(device)
    model = ClipClassifier(train_x.shape[-1], int(hidden_dim), int(train_y.max()) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(int(seed) + 917)
    batch_size = min(64, len(train_x))
    for _ in range(int(epochs)):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(train_x[idx]), train_y[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(test_x).argmax(-1)
    accuracy = (pred == test_y).float().mean().item()
    recalls = []
    f1s = []
    for label in range(int(test_y.max()) + 1):
        true = test_y == label
        guessed = pred == label
        tp = (true & guessed).sum().float()
        recall = tp / true.sum().clamp_min(1)
        precision = tp / guessed.sum().clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(torch.stack(recalls).mean()),
        "macro_f1": float(torch.stack(f1s).mean()),
    }
