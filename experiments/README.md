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

Edit `configs/train.yaml` (or pass CLI overrides): `subset: 1.0`, more epochs.
Use a distinct `--run-name`:

```bash
python train.py --model segformer --subset 1.0 --epochs 50 --run-name segformer-full
```

`batch_size` is a micro-batch (see the VRAM ceiling above), so raise the
*effective* batch by accumulating gradients rather than pushing `batch_size`
past the hardware limit:

```yaml
batch_size       : 4
grad_accum_steps : 8      # effective batch 32 — matches the production segmenter
amp              : true   # fp16 autocast + GradScaler (CUDA only)
```

`amp` applies to the training step only; validation stays fp32 so the reported
metrics are not perturbed by fp16 rounding. Accumulation does not change what
BatchNorm sees — it still normalises over `batch_size` samples per step.

### Full run result (SegFormer-B0, canonical split)

The one full run completed so far — `runs/segformer-full/`, SegFormer-B0 at
416², full canonical split (5089 train / 1357 val), `batch_size 4 ×
grad_accum_steps 8` (effective 32), AMP fp16, 50 epochs — selected epoch 47 on
`val_pixelF1` and was then evaluated on both splits with `eval.py`. Committed
summaries: [`experiments/results/segformer-full.json`](results/segformer-full.json)
(val) and [`segformer-full-test.json`](results/segformer-full-test.json) (test).

| Split | n | iouPV | pixelF1 | instanceF1 | mIoU | FPS (416²) |
|---|---|---|---|---|---|---|
| val | 1357 | 0.7037 | 0.8261 | 0.8135 | 0.8509 | 20.9 |
| test | 1239 | 0.7445 | 0.8535 | 0.8549 | 0.8714 | 42.7 |

The curve saturated around epoch 30–35 and then bounced inside a 0.81–0.83
`val_pixelF1` band; the extra epochs bought nothing. Wall-clock was dominated by
laptop sleep, not compute (awake epochs ~215 s).

### Benchmarking against the production baseline

Report on the **test** split (val is for model selection), using the
foreground-only `iouPV` / `pixelF1` keys — those are what line up with the
published baseline (DeepLabV3-ResNet101, `ign`: **IoU 0.735 / F1 0.844**,
n=1239). The test shards (`ign/test-*`, ~0.69 GB) are not part of the quick-run
download; fetch them and re-run `scripts/prepare_bdappv.py` first (the
`segformer-full-test.json` numbers above were produced that way).

```bash
python eval.py --checkpoint runs/segformer-full/checkpoints/best.pth \
    --split test --batch-size 8 \
    --out experiments/results/segformer-full-test.json \
    --previews-dir experiments/results/previews/segformer-full-test
```

On that split SegFormer-B0 lands **marginally ahead of the baseline** — iouPV
0.7445 / pixelF1 0.8535 against 0.735 / 0.844 — at 3.71 M parameters. Read that
as promising rather than conclusive; two caveats on comparability. The published
figures come from `train/evaluate.py`, which averages per-batch IoU/F1 ratios,
whereas `SegMetrics` pools TP/FP/FN across the whole split and computes each
ratio once — the pooled form is the statistically sounder one, and the
maintainers are recomputing the baseline rather than this scaffold adopting
mean-of-means. And the released segmenter is initialised from Mayer et al.
(2022) PV checkpoints, with cosine annealing, `weight_decay 1e-4` and early
stopping on val, none of which this scaffold does yet. One backbone, one seed,
no significance testing.

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
- **Image size**: BDAPPV is 400×400; default `image_size: 416`, the smallest
  multiple of 32 that covers it (SegFormer's encoder stages require a multiple
  of 32). 416 upsamples ~4% where 512 upsamples ~28% — cheaper in VRAM, and
  closer to the native 400 px the production baseline is evaluated at. The
  committed quick-run results below were produced at 512², before this default
  changed, so re-run them before comparing against 416² numbers.
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
  apples-to-apples. The effective batch must stay uniform too.

## Metrics

| key | definition |
|---|---|
| `mIoU` | mean of PV-class and background IoU |
| `iouPV` | foreground (PV) IoU — the key the published baseline reports as "IoU" |
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
