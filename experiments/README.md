# Experiments — candidate segmentation backbones (refs #11)

Quick, reproducible architecture comparison for the new-backbones issue:
SegFormer, DeepLabV3+ (ConvNeXt-Tiny) and U-Net (ConvNeXt-Tiny /
EfficientNet-B0) trained on BDAPPV positives, evaluated with the same
protocol so the numbers are directly comparable.

## Setup

```bash
pip install -r requirements-train.txt
```

The inference pipeline is not needed — `train.py` / `eval.py` only depend on
`requirements-train.txt`.

## Data (one-time download + prepare)

The Hub release ships **Parquet shards**, not the raw `img/` + `mask/`
directories: the `image` and `mask` columns carry the PNG bytes inline, and
there is no `annotations.csv` (the canonical split lives in the shards' `split`
column). `scripts/prepare_bdappv.py` materializes the raw layout the loader
expects. Download sizes for the `ign` provider:

| Split | Shards | Size |
|---|---|---|
| train | 7 | ~3.21 GB |
| validation | 2 | ~0.88 GB |
| test | 2 | ~0.69 GB |

```bash
# ign train + validation — enough for the quick runs (~4.1 GB)
hf download gabrielkasmi/bdappv --repo-type dataset --local-dir data/bdappv \
    ign/train-00000-of-00007.parquet ign/train-00001-of-00007.parquet \
    ign/train-00002-of-00007.parquet ign/train-00003-of-00007.parquet \
    ign/train-00004-of-00007.parquet ign/train-00005-of-00007.parquet \
    ign/train-00006-of-00007.parquet ign/validation-00000-of-00002.parquet \
    ign/validation-00001-of-00002.parquet

# everything (ign + google, ~8.1 GB)
hf download gabrielkasmi/bdappv --repo-type dataset --local-dir data/bdappv

# extract img/ + mask/ + annotations.csv from the shards (idempotent)
python scripts/prepare_bdappv.py --provider ign
```

Pass the shard paths explicitly as above — `--include "ign/*"` plus an
explicit filename in `--include "ign/*" "*.csv"` style can be misparsed
("Ignoring `--include` since filenames have been explicitly set"), and older
hub versions use `huggingface-cli download` with the same arguments.

Resulting layout (`data_dir` in `configs/train.yaml`):

```
data/bdappv/
├── ign/                          # or google/  (data_provider in config)
│   ├── *.parquet                 # as downloaded
│   ├── img/*.png                 # extracted positives, 400x400
│   └── mask/*.png                # binary PV masks, one per positive
├── annotations.csv               # image,split,provider  (split: train/val/test)
└── metadata.csv                  # not shipped; optional
```

Only mask-bearing images are extracted: segmentation trains on positives, and
`list_pairs()` ignores images without a mask. `prepare_bdappv.py
--include-negatives` writes those too (classification data, unused here).

## Environment variables

Only needed for `scripts/upload_results.sh` (HF Hub uploads) — copy
`.env.example` to `.env` and:

```bash
set -a; . ./.env; set +a
```

`HUGGINGFACE_HUB_TOKEN` is never required for training; anonymous downloads
cover the encoder pretraining weights and the dataset.

## Quick runs (5% subset, 3 epochs)

```bash
bash scripts/run_quick.sh segformer
bash scripts/run_quick.sh deeplab
bash scripts/run_quick.sh unet
# or all three:
bash scripts/run_quick.sh all
```

Each invocation:

1. trains `runs/<model>-mini/` (checkpoints, metrics.json, config dump,
   val_preds previews) — **not committed**, gitignored
2. evaluates the best checkpoint on the same validation subset and writes
   `experiments/results/<model>-mini.json` + preview PNGs under
   `experiments/results/previews/<model>-mini/` — **these are committed**

Equivalent direct commands:

```bash
python train.py --config configs/train.yaml --model segformer \
    --subset 0.05 --run-name segformer-mini
python eval.py --checkpoint runs/segformer-mini/checkpoints/best.pth \
    --out experiments/results/segformer-mini.json \
    --previews-dir experiments/results/previews/segformer-mini
```

`scripts/run_quick.sh` needs a real `bash` with coreutils. On Windows the
`bash` on PATH is usually the WSL launcher (`WindowsApps\bash.exe`), which
cannot use a Windows venv, and invoking MSYS2's `bash.exe` directly leaves
`/usr/bin` off PATH (`dirname: command not found`). Put MSYS2's bin on PATH
first:

```bat
set "PATH=C:\msys64\usr\bin;%PATH%"
set "PYTHON=%CD%\.venv\Scripts\python.exe"
bash scripts/run_quick.sh all
```

Otherwise just run the two commands above directly.

## Results (quick runs, `ign`, 5%)

254 train / 68 val positives, 3 epochs, batch 3, seed 42, RTX 3050 Laptop
(4 GB). Same data, seed and hyperparameters for all three models.

| Model | Params (M) | val_mIoU | val_pixelF1 | val_instanceF1 | FPS (512²) | s/epoch |
|---|---|---|---|---|---|---|
| SegFormer-B0 | 3.71 | 0.681 | 0.536 | 0.223 | 51.0 | 78 |
| DeepLabV3+ (ConvNeXt-Tiny) | 29.31 | 0.563 | 0.233 | 0.055 | 11.1 | 163 |
| U-Net (ConvNeXt-Tiny) | 31.93 | 0.646 | 0.458 | 0.033 | 6.1 | 187 |

`mIoU` is dominated by background here (foreground is only 0.2–0.5% of pixels
and every model scores `iouBG ≈ 0.994`), so `pixelF1` and `instanceF1` are the
discriminative numbers. Both ConvNeXt-Tiny decoders are still underfit at this
epoch budget.

## Full runs

Edit `configs/train.yaml` (or pass CLI overrides): `subset: 1.0`, more epochs,
possibly a larger `batch_size`. Use a distinct `--run-name`:

```bash
python train.py --model segformer --subset 1.0 --epochs 50 --run-name segformer-full
```

## Protocol notes

- **Split**: canonical department-based train/val/test split (dataset seed 42),
  carried by the Parquet shards' `split` column and written into
  `annotations.csv` by `scripts/prepare_bdappv.py` as `train`/`val`/`test`.
  Do not re-split — published BDAPPV results depend on it. `split_mode: random`
  exists only as a fallback and prints a warning.
- **Provider**: default `ign` — matches the pipeline's IGN 20 cm imagery and
  the Etalab 2.0 license. Switch to `google` (BDAPPV paper baselines) via
  `data_provider`. Google imagery carries Google Earth Engine ToS
  redistribution restrictions — relevant when publishing weights later.
- **Image size**: BDAPPV is 400×400; default `image_size: 512` upsamples.
  Keep multiples of 32 (SegFormer requirement).
- **Quick-run hyperparams** (held constant across models for comparability):
  5% subset, 3 epochs, batch 3, AdamW lr 1e-4, seed 42, basic augmentations
  (hflip/vflip/rot90). Same seed + `--subset` fraction ⇒ identical data
  subsets across models. On the `ign` split 5% is ~254 train / ~68 val images.
- **Batch size is VRAM-bound, not tuned**: batch 3 is the largest value that
  keeps the ConvNeXt-Tiny decoders in 4 GB without spilling to shared memory
  (measured peaks at 512²: 2430 MiB DeepLabV3+, 2469 MiB U-Net, 1239 MiB
  SegFormer-B0; batch 4 leaves only 64–112 MiB of headroom). Batch 1 is not
  possible at all — the decoders' BatchNorm layers raise
  `Expected more than 1 value per channel when training`. On a larger GPU raise
  `batch_size` uniformly across the models, or the comparison stops being
  apples-to-apples.

## Metrics

| key | definition |
|---|---|
| `mIoU` | mean of PV-class and background IoU |
| `pixelF1` | foreground (PV) pixel F1 |
| `instanceF1` | connected components matched one-to-one at IoU ≥ 0.5 (a proxy for per-installation detection) |
| `inference_fps` | single-image forward passes/s at `image_size` (eval.py) |

## Commit policy

- **Commit**: `experiments/results/*.json` and small preview PNGs under
  `experiments/results/previews/`
- **Never commit**: checkpoints (`*.pth`, `*.ckpt`), `runs/`, `.env`, tokens —
  all gitignored
- **Large artifacts**: `scripts/upload_results.sh` (private HF repo by default)

## Smoke test without the dataset

```bash
python train.py --dry-run
```

Synthetic data, 1 epoch, CPU, no downloads — verifies the whole
train → checkpoint → metrics → previews plumbing.
