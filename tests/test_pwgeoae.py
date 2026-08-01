import torch

from topo.ml_tda_pwgeoae import (
    blended_pwgeo_loss,
    persistence_edge_weights,
    persistence_weighted_geometry_loss,
)


def test_pwgeo_weights_include_symmetric_mst_and_local_edges():
    points = torch.tensor([[0.0], [1.0], [3.0], [7.0]])
    weights = persistence_edge_weights(torch.cdist(points, points), knn=1)
    assert torch.allclose(weights, weights.T)
    assert torch.all(weights.diag() == 0)
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) >= len(points) - 1
    assert weights[0, 1] > 1 and weights[1, 2] > 1 and weights[2, 3] > 1


def test_pwgeo_loss_is_scale_invariant_and_differentiable():
    frames = torch.tensor([[[[0.0]]], [[[1.0]]], [[[3.0]]], [[[7.0]]]])
    latents = (frames.flatten(start_dim=1) * 4).clone().requires_grad_(True)
    loss = persistence_weighted_geometry_loss(frames, latents, knn=2)
    assert float(loss.detach()) < 1e-10
    loss.backward()
    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()


def test_zero_blend_exactly_matches_geoae_loss():
    torch.manual_seed(4)
    frames = torch.randn(8, 1, 4, 4)
    latents = torch.randn(8, 3)
    from topo.ml_tda_geoae import geometry_proxy_loss
    expected = geometry_proxy_loss(frames, latents)
    actual = blended_pwgeo_loss(frames, latents, blend=0.0)
    assert torch.allclose(actual, expected)
