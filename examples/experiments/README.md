# Routing v2 ablations

This folder tests small changes to Routing Pyramids without modifying the reference implementation.

## Questions

1. Does a stride-4 center grid reduce merges compared with the reference stride-8 grid?
2. Does splitting thresholded center components by local peaks reduce merges without retraining?
3. Does mild random scale augmentation improve robustness once the center grid is denser?

Keep the segmentation thresholds fixed for the first comparison:

- `center_threshold = 0.5`
- `pixel_mass_threshold = 0.1`
- `min_object_area = 100`
- `peak_min_distance = 8` input pixels

The prediction script converts the 8-pixel peak distance to the corresponding coarse-grid distance for each architecture. Do not tune these values separately per variant until the ablation is complete.

## Training matrix

Only three checkpoints are required.

| checkpoint | center stride | peak split during training | spatial scale augmentation |
| --- | ---: | --- | --- |
| `baseline` | 8 | no | no |
| `stride4` | 4 | no | no |
| `stride4-scale` | 4 | no | 0.75–1.25x |

Peak splitting is inference-only, so each checkpoint can be evaluated with either connected-component seeds or peak-split seeds.

The stride-4 variants keep the same channel widths and routing block counts as the reference model. The only architecture changes are encoder strides `(2, 2, 1)`, decoder strides `(1, 2, 2)`, `patch_size=4`, and `background_latent_pool_stride=16`. The larger background pool preserves the reference model's 64-pixel effective background scale (`8 x 8 = 4 x 16`).

## Data

The training script expects the Cell Tracking Challenge Fluo-N2DL-HeLa dataset below `data/Fluo-N2DL-HeLa` by default, with both split directories present:

```text
data/Fluo-N2DL-HeLa/
├── train/
└── test/
```

Use `--data-dir PATH` if the dataset lives elsewhere. The architecture and peak-splitting unit tests do not require the dataset.

## Train

```bash
uv run python examples/experiments/train_fluohela.py baseline
uv run python examples/experiments/train_fluohela.py stride4
uv run python examples/experiments/train_fluohela.py stride4-scale
```

The default stride-4 batch size is 16 with gradient accumulation to keep the effective batch near 64. If that does not fit your GPU, use e.g. `--batch-size 8`; accumulation is adjusted automatically.

Smoke test a configuration before a long run:

```bash
uv run python examples/experiments/train_fluohela.py stride4 --fast-dev-run
```

## Predict on the labeled CTC training split

Assuming the default output locations, checkpoints are under:

```text
outputs/hela_ablation/<variant>/pyramid_flow_vae/run/checkpoints/last.ckpt
```

Run the six comparisons:

```bash
uv run python examples/experiments/predict_fluohela.py baseline \
  outputs/hela_ablation/baseline/pyramid_flow_vae/run/checkpoints/last.ckpt \
  --seed-mode connected

uv run python examples/experiments/predict_fluohela.py baseline \
  outputs/hela_ablation/baseline/pyramid_flow_vae/run/checkpoints/last.ckpt \
  --seed-mode peaks

uv run python examples/experiments/predict_fluohela.py stride4 \
  outputs/hela_ablation/stride4/pyramid_flow_vae/run/checkpoints/last.ckpt \
  --seed-mode connected

uv run python examples/experiments/predict_fluohela.py stride4 \
  outputs/hela_ablation/stride4/pyramid_flow_vae/run/checkpoints/last.ckpt \
  --seed-mode peaks

uv run python examples/experiments/predict_fluohela.py stride4-scale \
  outputs/hela_ablation/stride4-scale/pyramid_flow_vae/run/checkpoints/last.ckpt \
  --seed-mode connected

uv run python examples/experiments/predict_fluohela.py stride4-scale \
  outputs/hela_ablation/stride4-scale/pyramid_flow_vae/run/checkpoints/last.ckpt \
  --seed-mode peaks
```

`--split train` is the default so these outputs can be compared against the available CTC ground truth. Use `--split test` only for final unlabeled challenge-style prediction.

## Interpretation

Compare in this order:

1. `baseline-connected` vs `baseline-peaks`: pure postprocessing effect.
2. `baseline-connected` vs `stride4-connected`: pure center-resolution effect.
3. `stride4-connected` vs `stride4-peaks`: peak splitting after fixing center resolution.
4. `stride4-peaks` vs `stride4-scale-peaks`: scale augmentation effect.

Primary segmentation quality should be measured with the same CTC segmentation evaluation for every output. Also record instance count error, obvious merge/split failures on dense frames, inference time, and peak VRAM. A gain is not worth keeping if it is only obtained by variant-specific threshold tuning or an impractical memory increase.

## Checks

Focused experiment checks need only the default development environment:

```bash
uv run pytest tests/test_experimental.py
uv run ruff check \
  src/routing_pyramids/experimental.py \
  src/routing_pyramids/experimental_fluohela.py \
  examples/experiments \
  tests/test_experimental.py
```

The repository-wide type check includes visualization examples whose dependencies live in the optional `analysis` extra:

```bash
uv sync --extra analysis
uv run pyrefly check
```
