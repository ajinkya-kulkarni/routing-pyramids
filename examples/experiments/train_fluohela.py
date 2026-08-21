"""Train the minimal Routing Pyramids v2 ablation matrix on Fluo-N2DL-HeLa."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from routing_pyramids.augmentation import PhotometricAugmentationConfig
from routing_pyramids.data.temporal_video_dataset import CTCVideoDataset
from routing_pyramids.experimental import ExperimentalVideoTemporalDataModule
from routing_pyramids.experimental_fluohela import (
    VARIANTS,
    build_system,
    effective_batch_accumulation,
    get_variant,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/Fluo-N2DL-HeLa"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hela_ablation"),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant = get_variant(args.variant)
    batch_size = variant.default_batch_size if args.batch_size is None else args.batch_size
    accumulation = effective_batch_accumulation(batch_size)

    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    system = build_system(variant)
    if args.compile:
        system.compile()

    data = ExperimentalVideoTemporalDataModule(
        data_dir=str(args.data_dir),
        dataset_class=CTCVideoDataset,
        train_split="test",
        val_split="train",
        test_split="train",
        batch_size=batch_size,
        num_workers=args.num_workers,
        crop_size=256,
        sequence_length=1,
        temporal_frame_stride=1,
        temporal_crop_shift_probability=0.0,
        train_repeat_factor=50,
        spatial_scale_range=variant.spatial_scale_range,
        photometric_augmentation=PhotometricAugmentationConfig.from_ranges(
            scale=0.5,
            shift=0.2,
            gamma=(0.5, 2.0),
            noise_std=0.1,
            apply_prob=0.8,
        ),
        clip_quantile_low=0.05,
        clip_quantile_high=0.999,
        norm_quantile_low=0.50,
        norm_quantile_high=0.99,
    )

    output_dir = args.output_dir / variant.name
    callbacks: list[Callback] = [
        ModelCheckpoint(
            monitor="loss/val",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="{epoch}-{loss/val:.4f}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="pyramid_flow_vae",
        version="run",
    )
    trainer = L.Trainer(
        fast_dev_run=args.fast_dev_run,
        accelerator=args.accelerator,
        devices=1,
        num_nodes=1,
        strategy="auto",
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=accumulation,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    print(
        f"variant={variant.name} feature_stride={system.patch_size} "
        f"batch_size={batch_size} accumulate_grad_batches={accumulation} "
        f"spatial_scale_range={variant.spatial_scale_range}"
    )
    trainer.fit(system, datamodule=data)


if __name__ == "__main__":
    main()
