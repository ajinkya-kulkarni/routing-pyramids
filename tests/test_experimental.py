from __future__ import annotations

import torch

from routing_pyramids.experimental import (
    ExperimentalPyramidFlowSystem,
    RandomZoomCrop,
    label_center_components,
)
from routing_pyramids.pyramid_flow_system import (
    PyramidFlowDecoder2d,
    PyramidFlowEncoder2d,
)


def test_peak_split_separates_two_peaks_inside_one_threshold_component() -> None:
    center = torch.zeros(5, 9)
    center[2, 1:8] = 0.60
    center[2, 2] = 0.95
    center[2, 6] = 0.90

    connected = label_center_components(
        center,
        threshold=0.5,
        mode="connected",
        peak_min_distance=2,
    )
    peaks = label_center_components(
        center,
        threshold=0.5,
        mode="peaks",
        peak_min_distance=2,
    )

    assert int(connected.max()) == 1
    assert int(peaks.max()) == 2
    assert int(peaks[2, 2]) != int(peaks[2, 6])


def test_peak_split_keeps_flat_plateau_as_one_seed() -> None:
    center = torch.zeros(6, 6)
    center[1:5, 1:5] = 0.8

    peaks = label_center_components(
        center,
        threshold=0.5,
        mode="peaks",
        peak_min_distance=2,
    )

    assert int(peaks.max()) == 1


def test_random_zoom_crop_preserves_requested_shape() -> None:
    transform = RandomZoomCrop(32, scale_range=(0.75, 1.25))
    image = torch.rand(3, 64, 80)

    output = transform(image)

    assert output.shape == (3, 32, 32)


def test_stride4_system_reconstructs_input_shape() -> None:
    latent_dim = 8
    encoder = PyramidFlowEncoder2d(
        in_channels=1,
        channels=(4, 8, 12, 16),
        strides=(2, 2, 1),
        down_blocks=(1, 1, 1, 1),
        norm=("GROUP", {"num_groups": 2}),
    )
    decoder = PyramidFlowDecoder2d(
        in_channels=latent_dim,
        out_channels=1,
        channels=(latent_dim, 12, 8, 4),
        strides=(1, 2, 2),
        feature_stride=encoder.feature_stride,
        stage_blocks=(1, 1, 1, 1),
        transport_predictor="conv",
    )
    system = ExperimentalPyramidFlowSystem(
        encoder=encoder,
        decoder=decoder,
        foreground_latent_hidden_dim=16,
        background_latent_hidden_dim=8,
        background_latent_pool_stride=4,
        patch_size=4,
        latent_dim=latent_dim,
        latent_head_norm=("GROUP", {"num_groups": 2}),
        log_images_every_n_epochs=0,
    )
    system.eval()

    with torch.no_grad():
        output = system(torch.rand(1, 1, 32, 32))

    assert encoder.feature_stride == 4
    assert output["recon"].shape == (1, 1, 32, 32)
