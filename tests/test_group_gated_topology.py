import pytest
import torch

from topo import ml_tda_latent


def test_group_gate_has_two_bounded_weights_and_preserves_output_shape():
    model = ml_tda_latent.GroupGatedTopologicalPredictor(
        latent_dim=4,
        h0_dim=3,
        h1_dim=3,
        gate_hidden_dim=5,
        hidden_dim=8,
    )
    inputs = torch.randn(6, 2, 10)

    model.eval()
    outputs = model(inputs)

    assert outputs.shape == (6, 2, ml_tda_latent.ml_tda.LATENT_DIM)
    assert model.last_gate_weights.shape == (6, 2, 2)
    assert torch.all(model.last_gate_weights >= 0)
    assert torch.all(model.last_gate_weights <= 1)


def test_group_gate_initializes_near_direct_concatenation():
    model = ml_tda_latent.GroupGatedTopologicalPredictor(
        latent_dim=4,
        h0_dim=3,
        h1_dim=3,
        gate_hidden_dim=5,
        hidden_dim=8,
    )
    model.eval()
    model(torch.randn(6, 2, 10))

    assert torch.allclose(
        model.last_gate_weights,
        torch.full((6, 2, 2), 0.9),
        atol=1e-6,
    )


def test_group_gate_applies_one_scalar_to_each_homology_group():
    model = ml_tda_latent.GroupGatedTopologicalPredictor(
        latent_dim=2,
        h0_dim=2,
        h1_dim=2,
        gate_hidden_dim=3,
        hidden_dim=4,
    )
    with torch.no_grad():
        model.gate[0].weight.zero_()
        model.gate[0].bias.zero_()
        model.gate[2].weight.zero_()
        model.gate[2].bias.copy_(torch.tensor([0.0, 2.0]))

    inputs = torch.randn(3, 1, 6)
    expected = torch.sigmoid(torch.tensor([0.0, 2.0]))
    model.eval()
    model(inputs)

    assert torch.allclose(
        model.last_gate_weights,
        expected.expand(3, 1, 2),
    )


def test_z_gate_both_builds_z_h0_h1_features_in_that_order():
    train_payload = {
        "z": torch.randn(4, 2, 5),
        "h0": torch.full((4, 2, 3), 1.0),
        "h1": torch.full((4, 2, 3), 2.0),
    }
    test_payload = {
        "z": torch.randn(4, 2, 5),
        "h0": torch.full((4, 2, 3), 3.0),
        "h1": torch.full((4, 2, 3), 4.0),
    }

    train, test = ml_tda_latent.features_for_latent_tda_mode_pair(
        train_payload,
        test_payload,
        "z_gate_both",
    )

    assert train.shape == (4, 2, 11)
    assert test.shape == (4, 2, 11)
    assert torch.equal(train[..., :5], train_payload["z"])
    assert torch.equal(train[..., 5:8], train_payload["h0"])
    assert torch.equal(train[..., 8:], train_payload["h1"])


def test_group_gate_rejects_single_homology_mode():
    payload = {
        "z": torch.randn(4, 2, 5),
        "h0": torch.randn(4, 2, 3),
        "h1": torch.randn(4, 2, 3),
    }

    with pytest.raises(ValueError, match="requires both homology groups"):
        ml_tda_latent.features_for_latent_tda_mode_pair(
            payload,
            payload,
            "z_gate_h1",
        )


def test_z_h0_h1_blocks_are_standardized_independently():
    z = torch.tensor([0.0, 2.0]).reshape(2, 1, 1).expand(-1, -1, 2)
    h0 = torch.tensor([100.0, 104.0]).reshape(2, 1, 1).expand(-1, -1, 3)
    h1 = torch.tensor([-20.0, 0.0]).reshape(2, 1, 1).expand(-1, -1, 3)
    features = torch.cat([z, h0, h1], dim=-1)

    mean, std = ml_tda_latent._fit_z_h0_h1_standardizer(features, latent_dim=2)
    standardized = ml_tda_latent._standardize(features, mean, std)

    for block in (standardized[..., :2], standardized[..., 2:5], standardized[..., 5:]):
        assert torch.allclose(block.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(block.std(), torch.tensor(1.0), atol=1e-6)
