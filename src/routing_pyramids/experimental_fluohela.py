"""Shared Fluo-N2DL-HeLa configuration for routing-v2 ablations."""

from __future__ import annotations

from dataclasses import dataclass

from routing_pyramids.experimental import (
    CenterSeedMode,
    ExperimentalPyramidFlowSystem,
)
from routing_pyramids.pyramid_flow_system import (
    LossWeightSchedule,
    PyramidFlowDecoder2d,
    PyramidFlowEncoder2d,
    PyramidFlowSegmentationTestConfig,
)


@dataclass(frozen=True)
class FluoHeLaVariant:
    name: str
    encoder_strides: tuple[int, int, int]
    background_latent_pool_stride: int
    spatial_scale_range: tuple[float, float] | None
    default_batch_size: int


VARIANTS: dict[str, FluoHeLaVariant] = {
    "baseline": FluoHeLaVariant(
        name="baseline",
        encoder_strides=(2, 2, 2),
        background_latent_pool_stride=8,
        spatial_scale_range=None,
        default_batch_size=64,
    ),
    "stride4": FluoHeLaVariant(
        name="stride4",
        encoder_strides=(2, 2, 1),
        background_latent_pool_stride=16,
        spatial_scale_range=None,
        default_batch_size=16,
    ),
    "stride4-scale": FluoHeLaVariant(
        name="stride4-scale",
        encoder_strides=(2, 2, 1),
        background_latent_pool_stride=16,
        spatial_scale_range=(0.75, 1.25),
        default_batch_size=16,
    ),
}


def get_variant(name: str) -> FluoHeLaVariant:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown variant {name!r}; choose from {sorted(VARIANTS)}"
        ) from exc


def build_system(
    variant: FluoHeLaVariant,
    *,
    center_seed_mode: CenterSeedMode = "connected",
    center_peak_min_distance: int = 2,
    segmentation_test_config: PyramidFlowSegmentationTestConfig | None = None,
) -> ExperimentalPyramidFlowSystem:
    latent_dim = 64
    encoder_channels = (32, 64, 128, 256)
    encoder = PyramidFlowEncoder2d(
        in_channels=1,
        channels=encoder_channels,
        strides=variant.encoder_strides,
        down_blocks=(2, 2, 2, 2),
        norm="GROUP",
    )
    decoder = PyramidFlowDecoder2d(
        in_channels=latent_dim,
        out_channels=1,
        channels=(latent_dim, 128, 64, 32),
        strides=tuple(reversed(variant.encoder_strides)),
        feature_stride=encoder.feature_stride,
        transport_predictor="conv",
        stage_blocks=(1, 4, 2, 1),
        normalize_latent_blend=False,
        dual_stream=False,
        value_modulation=False,
    )
    system = ExperimentalPyramidFlowSystem(
        encoder=encoder,
        decoder=decoder,
        foreground_latent_hidden_dim=256,
        background_latent_hidden_dim=32,
        background_latent_pool_stride=variant.background_latent_pool_stride,
        patch_size=encoder.feature_stride,
        latent_dim=latent_dim,
        foreground_sparsity_alpha=0.5,
        recon_loss_weight=LossWeightSchedule(1.0, 1.0, 0),
        foreground_kl_loss_weight=LossWeightSchedule(0, 1e-2, 10),
        background_kl_loss_weight=LossWeightSchedule(0, 5e-2, 10),
        flow_l2_loss_weight=LossWeightSchedule(0.0, 5e-3, 10),
        foreground_sparsity_loss_weight=LossWeightSchedule(0, 2e-1, 10),
        loss_type="l1",
        lr=1e-4,
        warmup_epochs=10,
        log_images_every_n_epochs=1,
        log_image_samples=3,
        log_decoder_ablation_every_n_epochs=10,
        log_decoder_ablation_max_batch_size=8,
        segmentation_test_config=segmentation_test_config,
    )
    system.configure_center_seeds(
        mode=center_seed_mode,
        peak_min_distance=center_peak_min_distance,
    )
    return system


def effective_batch_accumulation(
    batch_size: int,
    *,
    target_batch_size: int = 64,
) -> int:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if target_batch_size <= 0:
        raise ValueError(
            f"target_batch_size must be positive, got {target_batch_size}"
        )
    return max(1, round(target_batch_size / batch_size))


__all__ = [
    "VARIANTS",
    "FluoHeLaVariant",
    "build_system",
    "effective_batch_accumulation",
    "get_variant",
]
