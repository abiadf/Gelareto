"""Select Diginetica next-item hyperparameters and manifold signature on validation sessions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from topo.config import DATASET_CONFIGS
from topo.ml_runner import parse_args
from topo.ml_tda_wikispeedia_forecast import run_graph_next_node


def _cfg(mode, seeds, hidden, epochs, lr):
    return parse_args([
        "--scenario", "graph_next_node", "--dataset", "diginetica",
        "--seeds", seeds, "--modes", mode, "--latent-dim", "16",
        "--hidden-dim", str(hidden), "--predictor-epochs", str(epochs),
        "--predictor-learning-rate", str(lr), "--horizon", "1",
        "--latent-tda-window", "5", "--device", "cpu", "--no-save",
    ])


def main():
    """Tune z-only prediction first, then compare diagnostic-guided signatures."""
    base = dict(DATASET_CONFIGS["diginetica"])
    base.update({
        "SKIP_TEST_EVALUATION": True,
        "MODEL_DIR": "models/diginetica_selection",
        "MAX_TRAIN_WINDOWS": 50_000,
    })
    hparam_rows = []
    for lr in (0.001, 0.003, 0.005):
        for hidden in (64, 128):
            for epochs in (5, 8):
                result, _ = run_graph_next_node(_cfg("z", "0", hidden, epochs, lr), base)
                row = result.iloc[0]
                hparam_rows.append({
                    "predictor_learning_rate": lr, "hidden_dim": hidden,
                    "predictor_epochs": epochs, "val_cross_entropy": row.val_cross_entropy,
                    "val_top1_accuracy": row.val_top1_accuracy,
                    "val_top5_accuracy": row.val_top5_accuracy, "val_mrr": row.val_mrr,
                })
    hparams = pd.DataFrame(hparam_rows).sort_values("val_cross_entropy")
    best = hparams.iloc[0]

    signature_runs = []
    # e16 is the control; the others progressively approach the stable h3_s10_e3 diagnosis.
    for signature in ("e16", "h2_s7_e7", "h3_s8_e5", "h3_s10_e3"):
        candidate = dict(base); candidate["MANIFOLD_SIGNATURE"] = signature
        result, _ = run_graph_next_node(_cfg(
            "graph_mixed", "0,1,2", int(best.hidden_dim),
            int(best.predictor_epochs), float(best.predictor_learning_rate),
        ), candidate)
        signature_runs.append(result)
    raw = pd.concat(signature_runs, ignore_index=True)
    signatures = raw.groupby("manifold_signature", as_index=False).agg(
        val_cross_entropy=("val_cross_entropy", "mean"),
        val_cross_entropy_std=("val_cross_entropy", "std"),
        val_top1_accuracy=("val_top1_accuracy", "mean"),
        val_top5_accuracy=("val_top5_accuracy", "mean"),
        val_mrr=("val_mrr", "mean"),
    ).sort_values("val_cross_entropy")

    hp_dir, sig_dir = Path("results/hparam_search"), Path("results/signature_selection")
    hp_dir.mkdir(parents=True, exist_ok=True); sig_dir.mkdir(parents=True, exist_ok=True)
    hparams.to_csv(hp_dir / "diginetica_hparam_grid.csv", index=False)
    raw.to_csv(sig_dir / "diginetica_signature_runs.csv", index=False)
    signatures.to_csv(sig_dir / "diginetica_signature_validation.csv", index=False)
    (hp_dir / "diginetica_best_hparams.json").write_text(json.dumps({
        "predictor_learning_rate": float(best.predictor_learning_rate),
        "hidden_dim": int(best.hidden_dim), "predictor_epochs": int(best.predictor_epochs),
        "selection": "lowest chronological-validation cross-entropy; test sessions untouched",
    }, indent=2) + "\n")
    selected = signatures.iloc[0]
    (sig_dir / "diginetica_selected_signature.json").write_text(json.dumps({
        "manifold_signature": selected.manifold_signature,
        "val_cross_entropy": float(selected.val_cross_entropy),
        "val_cross_entropy_std": float(selected.val_cross_entropy_std),
        "selection": "lowest 3-seed validation cross-entropy among diagnostic-guided candidates",
    }, indent=2) + "\n")
    print("\nHyperparameters:\n", hparams.to_string(index=False))
    print("\nSignatures:\n", signatures.to_string(index=False))


if __name__ == "__main__":
    main()
