"""Export Pechstre clips for official OpenSTL/SimVP benchmarks.

This module deliberately does not vendor OpenSTL. It prepares fixed horizon
train/test arrays and command templates so the official OpenSTL checkout can be
used as an external pixel-space benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from topo import ml_tda


@dataclass
class OpenSTLExport:
    export_dir: Path
    train_path: Path
    test_path: Path
    metadata_path: Path
    readme_path: Path
    train_script_path: Path
    n_train_samples: int
    n_test_samples: int
    sequence_shape: tuple[int, ...]


def _take_clip_subset(video_tensor: torch.Tensor, max_clips: int | None, seed: int) -> torch.Tensor:
    if max_clips is None or max_clips >= video_tensor.shape[1]:
        return video_tensor
    generator = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(video_tensor.shape[1], generator=generator)[: int(max_clips)]
    return video_tensor[:, idx]


def _make_horizon_sequences(
    video_tensor: torch.Tensor,
    *,
    input_frames: int,
    horizon: int,
) -> np.ndarray:
    """Return [N, input_frames + 1, H, W, C] with final frame at t+horizon."""
    video = ml_tda.tensor_to_model_float(video_tensor).detach().cpu()
    if video.ndim != 5:
        raise ValueError(f"Expected video tensor (T, B, C, H, W), got {tuple(video.shape)}")
    n_times, n_clips, channels, height, width = video.shape
    last_source = n_times - int(horizon) - 1
    if last_source < int(input_frames) - 1:
        raise ValueError(
            f"Need at least input_frames+horizon frames; got T={n_times}, "
            f"input_frames={input_frames}, horizon={horizon}"
        )
    sequences = []
    for clip_idx in range(n_clips):
        for source_t in range(int(input_frames) - 1, last_source + 1):
            past = video[source_t - int(input_frames) + 1 : source_t + 1, clip_idx]
            target = video[source_t + int(horizon) : source_t + int(horizon) + 1, clip_idx]
            seq = torch.cat([past, target], dim=0)
            sequences.append(seq.permute(0, 2, 3, 1).numpy())
    return np.asarray(sequences, dtype=np.float32)


def export_openstl_dataset(
    *,
    dataset: str,
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    seed: int,
    input_frames: int,
    horizon: int,
    max_train_clips: int | None,
    max_test_clips: int | None,
    output_root: Path = Path("datasets/openstl_exports"),
    method: str = "SimVP",
    config_file: str = "configs/mmnist/simvp/SimVP_gSTA.py",
) -> OpenSTLExport:
    train_subset = _take_clip_subset(x_train, max_train_clips, seed=seed)
    test_subset = _take_clip_subset(x_test, max_test_clips, seed=seed + 1)
    train_sequences = _make_horizon_sequences(train_subset, input_frames=input_frames, horizon=horizon)
    test_sequences = _make_horizon_sequences(test_subset, input_frames=input_frames, horizon=horizon)

    export_dir = (
        Path(output_root)
        / f"{dataset}_seed{seed}_in{int(input_frames)}_h{int(horizon)}"
        / f"train{train_subset.shape[1]}_test{test_subset.shape[1]}"
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    train_path = export_dir / "train_nthwc.npy"
    test_path = export_dir / "test_nthwc.npy"
    metadata_path = export_dir / "metadata.json"
    readme_path = export_dir / "README_openstl.md"
    train_script_path = export_dir / "run_openstl_train.sh"
    dataset_helper_path = export_dir / "pechstre_npy_dataset.py"
    np.save(train_path, train_sequences)
    np.save(test_path, test_sequences)

    metadata = {
        "dataset": dataset,
        "seed": int(seed),
        "input_frames": int(input_frames),
        "pred_frames": 1,
        "horizon": int(horizon),
        "layout": "N,T,H,W,C",
        "value_range": "[0, 1]",
        "train_path": str(train_path),
        "test_path": str(test_path),
        "n_train_clips": int(train_subset.shape[1]),
        "n_test_clips": int(test_subset.shape[1]),
        "n_train_samples": int(train_sequences.shape[0]),
        "n_test_samples": int(test_sequences.shape[0]),
        "sequence_shape": tuple(int(v) for v in train_sequences.shape[1:]),
        "openstl_method": method,
        "openstl_config_file": config_file,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    readme = f"""# OpenSTL export for {dataset}

This folder contains fixed-horizon Pechstre clips for running official OpenSTL
as an external pixel-space benchmark.

Arrays:
- `train_nthwc.npy`: shape `{tuple(train_sequences.shape)}`, layout `N,T,H,W,C`
- `test_nthwc.npy`: shape `{tuple(test_sequences.shape)}`, layout `N,T,H,W,C`

Sequence convention:
- The first `{int(input_frames)}` frames are the model input.
- The final frame is the target at horizon `{int(horizon)}`.
- Values are float32 in `[0, 1]`.

Recommended OpenSTL model:
- method: `{method}`
- config: `{config_file}`

OpenSTL does not natively know this Pechstre dataset name. Add/register a
custom dataloader in the OpenSTL checkout that reads these `.npy` files and
splits the first `{int(input_frames)}` frames as input and the final frame as
target. A minimal PyTorch dataset helper is written to
`pechstre_npy_dataset.py`. Then use the command template in
`run_openstl_train.sh`.
"""
    readme_path.write_text(readme, encoding="utf-8")

    dataset_helper = f'''"""Minimal PyTorch dataset for Pechstre OpenSTL exports.

Copy/adapt this into the official OpenSTL checkout and register it as a custom
dataset named "pechstre". Arrays are expected in N,T,H,W,C layout.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class PechstreNpyDataset(Dataset):
    def __init__(self, path: str, input_frames: int = {int(input_frames)}):
        self.data = np.load(path, mmap_mode="r")
        self.input_frames = int(input_frames)

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, idx):
        seq = torch.from_numpy(np.asarray(self.data[idx])).float()
        # N,T,H,W,C -> T,C,H,W
        seq = seq.permute(0, 3, 1, 2).contiguous()
        x = seq[: self.input_frames]
        y = seq[self.input_frames :]
        return x, y
'''
    dataset_helper_path.write_text(dataset_helper, encoding="utf-8")

    script = f"""#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   OPENSTL_ROOT=/path/to/OpenSTL bash {train_script_path.name}
#
# The official OpenSTL checkout must provide a custom dataloader for:
#   train: {train_path}
#   test:  {test_path}
# See the minimal helper: {dataset_helper_path}

: "${{OPENSTL_ROOT:?Set OPENSTL_ROOT to the official OpenSTL checkout}}"
cd "$OPENSTL_ROOT"

python tools/train.py \\
  --dataname pechstre \\
  --method {method} \\
  --config_file {config_file} \\
  --overwrite \\
  --ex_name pechstre_{dataset}_in{int(input_frames)}_h{int(horizon)}_{method}
"""
    train_script_path.write_text(script, encoding="utf-8")
    train_script_path.chmod(0o755)

    return OpenSTLExport(
        export_dir=export_dir,
        train_path=train_path,
        test_path=test_path,
        metadata_path=metadata_path,
        readme_path=readme_path,
        train_script_path=train_script_path,
        n_train_samples=int(train_sequences.shape[0]),
        n_test_samples=int(test_sequences.shape[0]),
        sequence_shape=tuple(int(v) for v in train_sequences.shape[1:]),
    )
