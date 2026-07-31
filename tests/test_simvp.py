import torch

from topo.ml_tda_simvp import SimVPFramePredictor, _batch_from_indices, _sample_indices


def test_simvp_predicts_one_frame_with_and_without_conditioning():
    frames = torch.rand(3, 5, 1, 32, 32)
    plain = SimVPFramePredictor(in_channels=1, hidden_dim=8, output_size=(32, 32))
    conditioned = SimVPFramePredictor(
        in_channels=1, hidden_dim=8, output_size=(32, 32), cond_dim=6
    )

    assert plain(frames).shape == (3, 1, 32, 32)
    assert conditioned(frames, torch.rand(3, 6)).shape == (3, 1, 32, 32)


def test_simvp_windows_and_condition_use_the_source_time():
    video = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2, 1, 1, 1)
    condition = torch.arange(8 * 2 * 3, dtype=torch.float32).reshape(8, 2, 3)
    indices = _sample_indices(video, input_frames=3, horizon=2)
    selected = indices[:1]
    x, y, cond = _batch_from_indices(video, selected, 3, 2, condition)
    source_t, clip = map(int, selected[0])

    assert torch.equal(x[0], video[source_t - 2 : source_t + 1, clip])
    assert torch.equal(y[0], video[source_t + 2, clip])
    assert torch.equal(cond[0], condition[source_t, clip])
