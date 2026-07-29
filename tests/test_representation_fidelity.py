import numpy as np
import pandas as pd

from scripts.analyze_representation_fidelity import _ensure_encoder
from topo.ml_tda_latent import (
    mean_normalized_distance_matrix,
    metric_distortion,
    representation_fidelity_for_window,
)


def test_metric_distortion_is_zero_for_scaled_copy():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
    distance_x, _ = mean_normalized_distance_matrix(points)
    distance_z, _ = mean_normalized_distance_matrix(7.5 * points)
    assert metric_distortion(distance_x, distance_z) < 1e-7


def test_fidelity_is_zero_for_same_geometry_up_to_scale_and_rotation():
    frames = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.25]],
        dtype=np.float32,
    )
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    latents = 3.0 * frames @ rotation
    metrics = representation_fidelity_for_window(frames, latents)
    assert metrics["metric_distortion"] < 1e-7
    assert metrics["h0_bottleneck"] < 1e-7
    assert metrics["h1_bottleneck"] < 1e-7


def test_distorted_geometry_has_positive_distortion():
    points = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    distorted = np.array([[0.0], [1.0], [4.0], [9.0]], dtype=np.float32)
    distance_x, _ = mean_normalized_distance_matrix(points)
    distance_z, _ = mean_normalized_distance_matrix(distorted)
    assert metric_distortion(distance_x, distance_z) > 0.1


def test_encoder_is_filled_for_baseline_rows_in_mixed_results():
    mixed = pd.DataFrame(
        {
            "scenario": ["latent_tda", "geo_latent_tda", "topo_latent_tda"],
            "encoder": [np.nan, "geo_ae", "topo_ae"],
        }
    )
    fixed = _ensure_encoder(mixed)
    assert fixed["encoder"].tolist() == ["ae", "geo_ae", "topo_ae"]
    assert fixed["encoder_variant"].tolist() == ["ae", "geo_ae", "topo_ae"]
