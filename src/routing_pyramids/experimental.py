"""Opt-in helpers for routing-pyramid ablation experiments.

This module deliberately leaves the reference implementation unchanged. It
contains only the small experimental changes needed to test denser center grids,
peak-split center seeds, and random spatial scale augmentation.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TVF
from skimage.measure import label
from skimage.morphology import local_maxima
from torch import Tensor
from torchvision.transforms import InterpolationMode

from routing_pyramids.augmentation import PhotometricAugmentationConfig
from routing_pyramids.data.temporal_datamodule import VideoTemporalDataModule
from routing_pyramids.data.transforms import DetectionTransform, get_normalization
from routing_pyramids.pyramid_flow_system import PyramidFlowSystem, PyramidTransport

CenterSeedMode = Literal["connected", "peaks"]


def _validate_scale_range(scale_range: tuple[float, float]) -> tuple[float, float]:
    low, high = float(scale_range[0]), float(scale_range[1])
    if low <= 0.0 or high <= 0.0:
        raise ValueError(f"spatial scale bounds must be positive, got {scale_range}")
    if low > high:
        raise ValueError(f"spatial scale range must be ordered, got {scale_range}")
    return low, high


def _plateau_representative(
    center_map: np.ndarray,
    plateau_mask: np.ndarray,
) -> tuple[int, int, float]:
    coords = np.argwhere(plateau_mask)
    if coords.size == 0:
        raise ValueError("plateau_mask must contain at least one pixel")
    values = center_map[coords[:, 0], coords[:, 1]]
    max_value = float(values.max())
    maxima = coords[values == max_value]
    centroid = maxima.astype(np.float64).mean(axis=0)
    distances = ((maxima.astype(np.float64) - centroid) ** 2).sum(axis=1)
    y, x = maxima[int(np.argmin(distances))]
    return int(y), int(x), max_value


def _greedy_peak_nms(
    candidates: list[tuple[int, int, float]],
    *,
    min_distance: int,
) -> list[tuple[int, int, float]]:
    if not candidates:
        return []
    min_distance_sq = float(min_distance * min_distance)
    selected: list[tuple[int, int, float]] = []
    for candidate in sorted(candidates, key=lambda item: item[2], reverse=True):
        y, x, _ = candidate
        if all(
            float((y - sy) ** 2 + (x - sx) ** 2) >= min_distance_sq
            for sy, sx, _ in selected
        ):
            selected.append(candidate)
    return selected


def label_center_components(
    center_map: Tensor,
    *,
    threshold: float,
    mode: CenterSeedMode = "connected",
    peak_min_distance: int = 2,
) -> Tensor:
    """Label coarse object seeds, optionally splitting connected multi-peak regions.

    ``connected`` reproduces the reference implementation. ``peaks`` finds
    regional-maxima plateaus inside each thresholded connected component,
    performs a small greedy distance suppression, and partitions the support
    region by the nearest retained peak. The routing decoder still performs the
    pixel-level instance assignment.
    """
    if center_map.ndim != 2:
        raise ValueError(
            f"center_map must have shape (H, W), got {tuple(center_map.shape)}"
        )
    if mode not in ("connected", "peaks"):
        raise ValueError(f"mode must be 'connected' or 'peaks', got {mode!r}")
    if peak_min_distance < 1:
        raise ValueError(
            f"peak_min_distance must be >= 1, got {peak_min_distance}"
        )

    center_cpu = center_map.detach().float().cpu()
    center_np = center_cpu.numpy()
    support_np = center_np >= float(threshold)
    support_components = label(support_np).astype(np.int32)
    if mode == "connected" or int(support_components.max()) == 0:
        return torch.as_tensor(
            support_components,
            device=center_map.device,
            dtype=torch.int32,
        )

    maxima_np = local_maxima(center_np) & support_np
    plateau_components = label(maxima_np).astype(np.int32)

    split_components = np.zeros_like(support_components, dtype=np.int32)
    next_component_id = 1
    for support_id in range(1, int(support_components.max()) + 1):
        support_mask = support_components == support_id
        plateau_ids = np.unique(plateau_components[support_mask])
        plateau_ids = plateau_ids[plateau_ids > 0]
        candidates = [
            _plateau_representative(
                center_np,
                (plateau_components == int(plateau_id)) & support_mask,
            )
            for plateau_id in plateau_ids
        ]

        if not candidates:
            coords = np.argwhere(support_mask)
            values = center_np[coords[:, 0], coords[:, 1]]
            best = coords[int(np.argmax(values))]
            candidates = [(int(best[0]), int(best[1]), float(values.max()))]

        peaks = _greedy_peak_nms(candidates, min_distance=peak_min_distance)
        if not peaks:
            raise RuntimeError("peak suppression unexpectedly removed every center peak")

        support_coords = np.argwhere(support_mask)
        peak_coords = np.asarray([(y, x) for y, x, _ in peaks], dtype=np.float64)
        delta = support_coords[:, None, :].astype(np.float64) - peak_coords[None, :, :]
        nearest_peak = np.argmin((delta * delta).sum(axis=2), axis=1)
        assigned_ids = next_component_id + nearest_peak.astype(np.int32)
        split_components[support_coords[:, 0], support_coords[:, 1]] = assigned_ids
        next_component_id += len(peaks)

    return torch.as_tensor(
        split_components,
        device=center_map.device,
        dtype=torch.int32,
    )


class ExperimentalPyramidFlowSystem(PyramidFlowSystem):
    """Reference system with an opt-in coarse-center peak split at inference."""

    center_seed_mode: CenterSeedMode = "connected"
    center_peak_min_distance: int = 2

    def configure_center_seeds(
        self,
        *,
        mode: CenterSeedMode = "connected",
        peak_min_distance: int = 2,
    ) -> None:
        """Configure inference-only center extraction without changing checkpoints."""
        if mode not in ("connected", "peaks"):
            raise ValueError(
                f"center seed mode must be 'connected' or 'peaks', got {mode!r}"
            )
        if peak_min_distance < 1:
            raise ValueError(
                f"peak_min_distance must be >= 1, got {peak_min_distance}"
            )
        self.center_seed_mode = mode
        self.center_peak_min_distance = int(peak_min_distance)

    def _flow_induced_single_label_image(
        self,
        *,
        center_presence: Tensor,
        layer_transports: tuple[PyramidTransport, ...],
        center_threshold: float,
        pixel_mass_threshold: float,
        component_chunk_size: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        center_components = label_center_components(
            center_presence[0, 0],
            threshold=center_threshold,
            mode=self.center_seed_mode,
            peak_min_distance=self.center_peak_min_distance,
        )
        num_components = int(center_components.max().item())
        device = center_presence.device
        output_h, output_w = layer_transports[-1].lookup.fine_hw
        if num_components == 0:
            empty_mass = torch.zeros(
                output_h, output_w, dtype=torch.float32, device=device
            )
            empty_labels = torch.zeros(
                output_h, output_w, dtype=torch.int32, device=device
            )
            return center_components, empty_labels, empty_mass

        component_ids = torch.arange(
            1,
            num_components + 1,
            device=device,
            dtype=torch.int32,
        )
        component_map = center_components.view(1, 1, *center_components.shape)
        winning_mass: Tensor | None = None
        winning_label: Tensor | None = None
        for chunk_start in range(0, num_components, component_chunk_size):
            chunk_ids = component_ids[chunk_start : chunk_start + component_chunk_size]
            component_mask = component_map == chunk_ids.view(1, -1, 1, 1)
            mass = torch.where(
                component_mask,
                center_presence.float(),
                0.0,
            )
            for transport in layer_transports:
                mass = self._gather_scalar_map(mass, transport=transport)
            chunk_mass, chunk_index = mass[0].max(dim=0)
            chunk_label = chunk_ids[chunk_index]
            if winning_mass is None or winning_label is None:
                winning_mass = chunk_mass
                winning_label = chunk_label
                continue
            replace = chunk_mass > winning_mass
            winning_mass = torch.where(replace, chunk_mass, winning_mass)
            winning_label = torch.where(replace, chunk_label, winning_label)

        assert winning_mass is not None and winning_label is not None
        pred_labels = torch.where(
            winning_mass >= float(pixel_mass_threshold),
            winning_label,
            0,
        ).to(dtype=torch.int32)
        return center_components, pred_labels, winning_mass.float()


class RandomZoomCrop(torch.nn.Module):
    """Randomly change object scale while returning a fixed-size image crop."""

    def __init__(
        self,
        size: int,
        *,
        scale_range: tuple[float, float] = (0.75, 1.25),
    ) -> None:
        super().__init__()
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        self.size = int(size)
        self.scale_range = _validate_scale_range(scale_range)

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim < 3:
            raise ValueError(
                f"image must have at least 3 dimensions, got {tuple(image.shape)}"
            )
        height, width = int(image.shape[-2]), int(image.shape[-1])
        low, high = self.scale_range
        zoom = float(torch.empty((), device=image.device).uniform_(low, high).item())
        crop_h = min(height, max(1, round(self.size / zoom)))
        crop_w = min(width, max(1, round(self.size / zoom)))
        max_top = height - crop_h
        max_left = width - crop_w
        top = (
            0
            if max_top == 0
            else int(torch.randint(max_top + 1, size=(), device=image.device).item())
        )
        left = (
            0
            if max_left == 0
            else int(torch.randint(max_left + 1, size=(), device=image.device).item())
        )
        return TVF.resized_crop(
            image,
            top=top,
            left=left,
            height=crop_h,
            width=crop_w,
            size=[self.size, self.size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )


def _scale_augmented_transforms(
    *,
    crop_size: int,
    spatial_scale_range: tuple[float, float],
    photometric_augmentation: PhotometricAugmentationConfig | None,
) -> DetectionTransform:
    return DetectionTransform(
        T.Compose(
            [
                RandomZoomCrop(crop_size, scale_range=spatial_scale_range),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomChoice(
                    [
                        T.Identity(),
                        T.RandomRotation(degrees=(90, 90)),
                        T.RandomRotation(degrees=(180, 180)),
                        T.RandomRotation(degrees=(270, 270)),
                    ]
                ),
            ]
        ),
        photometric_augmentation=photometric_augmentation,
    )


class ExperimentalVideoTemporalDataModule(VideoTemporalDataModule):
    """Video data module with opt-in random object-scale augmentation."""

    def __init__(
        self,
        *,
        spatial_scale_range: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None:
        validated_scale_range = (
            None
            if spatial_scale_range is None
            else _validate_scale_range(spatial_scale_range)
        )
        super().__init__(**kwargs)
        self.spatial_scale_range = validated_scale_range

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        if self.spatial_scale_range is None or stage not in (None, "fit"):
            return
        if self.crop_size is None:
            raise ValueError("spatial scale augmentation requires a finite crop_size")
        if self.temporal_crop_shift_probability != 0.0:
            raise ValueError(
                "spatial scale augmentation currently requires "
                "temporal_crop_shift_probability=0"
            )

        normalization = get_normalization(
            clip_quantile_low=self.clip_quantile_low,
            norm_quantile_low=self.norm_quantile_low,
            norm_quantile_high=self.norm_quantile_high,
            clip_quantile_high=self.clip_quantile_high,
        )
        train_augment = _scale_augmented_transforms(
            crop_size=int(self.crop_size),
            spatial_scale_range=self.spatial_scale_range,
            photometric_augmentation=self.photometric_augmentation,
        )
        self.train_ds = self.dataset_class(
            root_dir=self.data_dir,
            split=self.train_split,
            normalization=normalization,
            augmentations=train_augment,
            sequence_length=self.sequence_length,
            temporal_source_length=self.temporal_source_length,
            temporal_frame_stride=self.temporal_frame_stride,
            scale_factor=self.input_scale_factor,
        )


__all__ = [
    "CenterSeedMode",
    "ExperimentalPyramidFlowSystem",
    "ExperimentalVideoTemporalDataModule",
    "RandomZoomCrop",
    "label_center_components",
]
