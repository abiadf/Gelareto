import torch
import numpy as np

from topo.ml_tda import _resample_multivariate_trajectory
from topo.ml_tda_classification import (
    clip_features,
    labels_from_sources,
    paired_signflip_test,
    train_evaluate_classifier,
)


def test_classification_labels_follow_source_blocks():
    config = {
        "sources": [
            {"motion_label": 0, "object_label": 1},
            {"motion_label": 1, "object_label": 0},
        ],
        "_classification_train_counts": [2, 3],
    }
    assert labels_from_sources(config, "train", "motion").tolist() == [0, 0, 1, 1, 1]
    assert labels_from_sources(config, "train", "object").tolist() == [1, 1, 0, 0, 0]


def test_direct_archive_labels_and_variable_length_resampling():
    config = {"_classification_test_labels": [2, 0, 1]}
    assert labels_from_sources(config, "test", "character").tolist() == [2, 0, 1]
    case = np.asarray([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    resampled = _resample_multivariate_trajectory(case, 5)
    assert resampled.shape == (2, 5)
    assert np.allclose(resampled[:, [0, -1]], case)


def test_clip_features_have_fixed_per_clip_width_and_shuffle_control():
    payload = {
        "z": torch.randn(8, 5, 4),
        "h0": torch.randn(8, 5, 3),
        "h1": torch.randn(8, 5, 3),
    }
    assert clip_features(payload, "z", window=3).shape == (5, 16)
    assert clip_features(payload, "h1", window=3).shape == (5, 12)
    assert clip_features(payload, "z_h1", window=3).shape == (5, 28)
    assert clip_features(payload, "z_h1_shuffle", window=3).shape == (5, 28)
    assert clip_features(payload, "z_h0_shuffle", window=3).shape == (5, 28)


def test_exact_paired_signflip_test():
    assert paired_signflip_test([1, 1, 1, 1, 1]) == 1 / 32
    assert paired_signflip_test([-1, -1, -1, -1, -1]) == 1.0


def test_classifier_learns_simple_separable_problem():
    train_x = torch.cat([torch.randn(30, 2) - 3, torch.randn(30, 2) + 3])
    test_x = torch.cat([torch.randn(10, 2) - 3, torch.randn(10, 2) + 3])
    train_y = torch.tensor([0] * 30 + [1] * 30)
    test_y = torch.tensor([0] * 10 + [1] * 10)
    metrics = train_evaluate_classifier(
        train_x,
        train_y,
        test_x,
        test_y,
        seed=0,
        hidden_dim=8,
        epochs=50,
        learning_rate=1e-2,
        device=torch.device("cpu"),
    )
    assert metrics["accuracy"] > 0.9
