"""Run connected-seed or peak-split segmentation for a HeLa ablation checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import lightning as L
import torch

from routing_pyramids.data.temporal_datamodule import VideoTemporalDataModule
from routing_pyramids.data.temporal_video_dataset import CTCVideoDataset
from routing_pyramids.experimental import ExperimentalPyramidFlowSystem
from routing_pyramids.experimental_fluohela import VARIANTS, build_system, get_variant
from routing_pyramids.prediction import CTCSegmentationPredictionWriter
from routing_pyramids.pyramid_flow_system import PyramidFlowSegmentationTestConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--seed-mode",
        choices=("connected", "peaks"),
        default="connected",
    )
    parser.add_argument("--peak-min-distance-px", type=int, default=8)
    parser.add_argument("--center-threshold", type=float, default=0.5)
    parser.add_argument("--pixel-mass-threshold", type=float, default=0.1)
    parser.add_argument("--min-object-area", type=int, default=100)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/Fluo-N2DL-HeLa"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("predictions/hela_ablation"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant = get_variant(args.variant)
    torch.set_float32_matmul_precision("high")

    feature_stride = math.prod(variant.encoder_strides)
    peak_min_distance = max(1, round(args.peak_min_distance_px / feature_stride))
    segmentation_config = PyramidFlowSegmentationTestConfig(
        center_threshold=args.center_threshold,
        pixel_mass_threshold=args.pixel_mass_threshold,
        min_object_area=args.min_object_area,
        max_object_area=None,
        prediction_output_scale_factor=1.0,
    )
    template = build_system(
        variant,
        segmentation_test_config=segmentation_config,
    )
    system = ExperimentalPyramidFlowSystem.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        encoder=template.encoder,
        decoder=template.decoder,
        segmentation_test_config=segmentation_config,
    )
    system.configure_center_seeds(
        mode=args.seed_mode,
        peak_min_distance=peak_min_distance,
    )

    data = VideoTemporalDataModule(
        data_dir=str(args.data_dir),
        dataset_class=CTCVideoDataset,
        train_split="test",
        val_split="train",
        test_split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=None,
        sequence_length=1,
        temporal_source_length=1,
        temporal_frame_stride=1,
        input_scale_factor=1.0,
        pin_memory=True,
        drop_last=False,
        clip_quantile_low=0.05,
        clip_quantile_high=0.999,
        norm_quantile_low=0.50,
        norm_quantile_high=0.99,
    )
    output_dir = args.output_dir / args.split / f"{variant.name}-{args.seed_mode}"
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=1,
        precision=args.precision,
        callbacks=[CTCSegmentationPredictionWriter(output_dir)],
        logger=False,
    )
    print(
        f"variant={variant.name} split={args.split} seed_mode={args.seed_mode} "
        f"feature_stride={feature_stride} peak_min_distance={peak_min_distance} "
        f"({args.peak_min_distance_px} input px) output={output_dir}"
    )
    trainer.predict(model=system, datamodule=data, return_predictions=False)


if __name__ == "__main__":
    main()
