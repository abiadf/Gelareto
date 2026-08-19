"""Select lightweight Wikispeedia predictor settings using validation paths only."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from topo.config import DATASET_CONFIGS
from topo.ml_runner import parse_args
from topo.ml_tda_wikispeedia_forecast import run_wikispeedia_next_node


def _cfg(*, mode: str, hidden: int, epochs: int, lr: float):
    return parse_args([
        "--scenario", "wikispeedia_next_node", "--dataset", "wikispeedia",
        "--seeds", "0", "--modes", mode, "--latent-dim", "16",
        "--hidden-dim", str(hidden), "--predictor-epochs", str(epochs),
        "--predictor-learning-rate", str(lr), "--horizon", "1",
        "--latent-tda-window", "5", "--device", "cpu", "--no-save",
    ])


def main():
    base = dict(DATASET_CONFIGS["wikispeedia"])
    base.update({
        "MAX_TRAIN_WINDOWS": 50_000,
        "SKIP_TEST_EVALUATION": True,
        "MODEL_DIR": "models/wikispeedia_selection",
    })
    hparam_rows = []
    for lr in (0.001, 0.003, 0.005):
        for hidden in (32, 64):
            for epochs in (5, 8):
                results, _ = run_wikispeedia_next_node(
                    _cfg(mode="z", hidden=hidden, epochs=epochs, lr=lr), base
                )
                row = results.iloc[0]
                hparam_rows.append({
                    "predictor_learning_rate": lr, "hidden_dim": hidden,
                    "predictor_epochs": epochs, "val_cross_entropy": row.val_cross_entropy,
                    "val_top1_accuracy": row.val_top1_accuracy,
                    "val_top5_accuracy": row.val_top5_accuracy, "val_mrr": row.val_mrr,
                })
    hparams = pd.DataFrame(hparam_rows).sort_values("val_cross_entropy")
    best = hparams.iloc[0]

    signature_rows = []
    for signature in ("e16", "h1_s7_e8", "h2_s11_e3", "h2_s14"):
        candidate = dict(base)
        candidate["MANIFOLD_SIGNATURE"] = signature
        results, _ = run_wikispeedia_next_node(_cfg(
            mode="graph_mixed", hidden=int(best.hidden_dim),
            epochs=int(best.predictor_epochs), lr=float(best.predictor_learning_rate),
        ), candidate)
        row = results.iloc[0]
        signature_rows.append({
            "manifold_signature": signature, "val_cross_entropy": row.val_cross_entropy,
            "val_top1_accuracy": row.val_top1_accuracy,
            "val_top5_accuracy": row.val_top5_accuracy, "val_mrr": row.val_mrr,
        })
    signatures = pd.DataFrame(signature_rows).sort_values("val_cross_entropy")

    hparam_dir = Path("results/hparam_search")
    signature_dir = Path("results/signature_selection")
    hparam_dir.mkdir(parents=True, exist_ok=True)
    signature_dir.mkdir(parents=True, exist_ok=True)
    hparams.to_csv(hparam_dir / "wikispeedia_hparam_grid.csv", index=False)
    signatures.to_csv(signature_dir / "wikispeedia_signature_validation.csv", index=False)
    (hparam_dir / "wikispeedia_best_hparams.json").write_text(json.dumps({
        "predictor_learning_rate": float(best.predictor_learning_rate),
        "hidden_dim": int(best.hidden_dim), "predictor_epochs": int(best.predictor_epochs),
        "selection": "lowest validation cross-entropy; test paths untouched",
    }, indent=2) + "\n")
    selected = signatures.iloc[0]
    (signature_dir / "wikispeedia_selected_signature.json").write_text(json.dumps({
        "manifold_signature": selected.manifold_signature,
        "val_cross_entropy": float(selected.val_cross_entropy),
        "selection": "lowest validation cross-entropy among triangle-softened candidates; test paths untouched",
    }, indent=2) + "\n")
    print("\nHyperparameters:\n", hparams.to_string(index=False))
    print("\nSignatures:\n", signatures.to_string(index=False))


if __name__ == "__main__":
    main()
