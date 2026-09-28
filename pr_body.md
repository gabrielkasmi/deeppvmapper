Refs #11

## Summary

Adds the training scaffold and the first real architecture comparison for the
new candidate segmentation backbones on BDAPPV. Quick runs on a 5% subset of
the canonical split give a clear early answer:

> **SegFormer-B0 wins on every metric at ~8× fewer parameters** than the
> ConvNeXt-Tiny candidates — mIoU 0.681 vs 0.646/0.563, instance F1 0.223 vs
> 0.033/0.055, and 5–8× faster inference.

This PR also unblocks the scaffold itself: as committed it could not open the
dataset at all (see *Findings*).

## What's in here

| Path | What it is |
|---|---|
| `train.py`, `eval.py` | training / evaluation entry points |
| `configs/{train,model}.yaml` | run config + model zoo (5 backbones) |
| `scripts/prepare_bdappv.py` | **new** — materializes the dataset layout the loader expects |
| `scripts/run_quick.sh`, `scripts/upload_results.sh` | quick-run wrapper, artifact upload |
| `experiments/README.md`, `experiments/results/` | protocol, metrics JSON, preview PNGs |
| `requirements-train.txt` | training-only deps, kept separate from the pipeline's |

**Reviewer's guide:** the code changes are `scripts/prepare_bdappv.py`,
`configs/train.yaml`, `train.py`, `eval.py`, `scripts/run_quick.sh` and
`requirements-train.txt`. Everything under `experiments/results/` is generated
output; `pr_body.md` and `CHANGELOG.md` are write-ups. No weights are committed
— `.gitignore` covers `runs/`, `models/`, `*.pth`, `*.ckpt`, `logs/`.

## Design notes

- **Dataset / provider**: BDAPPV via HF snapshot. Default is `ign` — matches the
  pipeline's IGN 20 cm imagery and the Etalab 2.0 license; Google imagery
  carries redistribution restrictions that matter when we later publish weights.
  Switchable via `configs/train.yaml`.
- **Split**: the canonical department-based train/val/test split, read from the
  Parquet shards' `split` column. The dataset card asks not to re-split; a
  seeded random fallback exists but warns loudly, since it breaks comparability
  with published results.
- **Checkpoints** are state-dict + model-spec dicts (no full-model pickles),
  ready for the loader in #7 to consume via `pretrained_hf_id` (placeholder in
  `configs/model.yaml`).
- **Metrics**: mIoU, pixel F1, and instance-level F1 (connected components
  matched one-to-one at IoU ≥ 0.5). The issue draft mentions `val_buildingF1`;
  BDAPPV masks are PV-only (no building channel), so this PR reports
  `val_instanceF1` as the object-level metric — happy to rename if maintainers
  prefer.

## Results

Quick runs, `ign` provider, 5% of the canonical split (254 train / 68 val
positives), 3 epochs, batch 3, seed 42, RTX 3050 Laptop (4 GB). Identical data,
seed and hyperparameters for all three models, so the columns are comparable.

| Model | Params (M) | val_mIoU | val_pixelF1 | val_instanceF1 | FPS (512²) | s/epoch |
|---|---|---|---|---|---|---|
| **SegFormer-B0** | **3.71** | **0.681** | **0.536** | **0.223** | **51.0** | **78** |
| DeepLabV3+ (ConvNeXt-Tiny) | 29.31 | 0.563 | 0.233 | 0.055 | 11.1 | 163 |
| U-Net (ConvNeXt-Tiny) | 31.93 | 0.646 | 0.458 | 0.033 | 6.1 | 187 |

Qualitative previews (input | ground truth | prediction) for all three models
are in `experiments/results/previews/`. SegFormer over-segments (~2× GT area),
DeepLabV3+ under-segments (~0.3–0.8×) — visible in both the previews and the
metrics. Full metrics are in `experiments/results/*.json`.

## Findings worth knowing

**1. The dataset card and the Hub disagree, and the card is the wrong one.**
BDAPPV is documented as `img/` + `mask/` + `annotations.csv`. The Hub actually
serves **Parquet shards with the PNG bytes inline** and no CSV at all, so
`list_pairs()` raised `FileNotFoundError` and `split_from_annotations()` had
nothing to read. `scripts/prepare_bdappv.py` extracts the documented layout from
the shards, taking the canonical department-based split from their `split`
column (`validation` → `val`). It is idempotent, so it can be re-run as more
shards arrive.

**2. `val_mIoU` is a poor `selection_metric` for this dataset.** PV installations
cover only **0.2–0.5% of pixels**, and every model scores `iouBG ≈ 0.994` — so
mIoU is mostly background, and it can hand `best.pth` to a near-empty
prediction. DeepLabV3+ is the visible case: its mIoU (0.563) looks far closer to
U-Net's (0.646) than its pixel F1 (0.233 vs 0.458) does, because it
under-segments. `pixelF1` / `instanceF1` are the discriminative metrics here.

**3. Batch size is a hardware floor, not a tuned hyperparameter.** Batch 8
overcommits 4 GB for the ConvNeXt-Tiny decoders — and Windows spills to shared
memory *silently* rather than raising OOM, so it looks like it worked. Measured
peaks at 512²: DeepLabV3+ 6250 MiB, U-Net 6336 MiB against ~3306 MiB actually
free. Batch 4 leaves only 64–112 MiB of headroom, and batch 1 is impossible
because the decoders' BatchNorm raises on a single sample. Hence 3, held
constant across models.

## Reproduce

```bash
# 1. snapshot (ign train + validation, ~4.1 GB) — shards named explicitly,
#    because `--include` with several patterns is easily misparsed
hf download gabrielkasmi/bdappv --repo-type dataset --local-dir data/bdappv \
    ign/train-0000{0..6}-of-00007.parquet \
    ign/validation-0000{0..1}-of-00002.parquet

# 2. materialize img/ + mask/ + annotations.csv  (-> train 5089 / val 1357)
python scripts/prepare_bdappv.py --provider ign

# 3. per model: train + evaluate
python train.py --config configs/train.yaml --model segformer \
    --subset 0.05 --run-name segformer-mini
python eval.py --checkpoint runs/segformer-mini/checkpoints/best.pth \
    --out experiments/results/segformer-mini.json \
    --previews-dir experiments/results/previews/segformer-mini
```

`bash scripts/run_quick.sh {segformer,deeplab,unet,all}` wraps step 3 for all
three models. (On Windows the `bash` on PATH is the WSL launcher, which cannot
use a Windows venv, and calling MSYS2's `bash.exe` directly leaves `/usr/bin`
off PATH — `experiments/README.md` documents the working invocation.)

## Scope and caveats

These are deliberately **quick** runs. 3 epochs over 254 images is enough to
rank architectures, **not** to report benchmark numbers, and they are labelled
as such throughout. Both ConvNeXt-Tiny decoders are still clearly underfit at
this budget, so the gap between them and SegFormer here is a data/epochs limit
at least as much as an architectural verdict — I would not conclude from this
that they are worse backbones. Full-scale training on the canonical split is the
next step, and I would re-run with a better `selection_metric` before quoting
any final figures.

## Next steps

- switch `selection_metric` off `val_mIoU` (to `val_pixelF1` or
  `val_instanceF1`) and re-run the three models,
- full-scale training for the most promising backbone(s) on the canonical
  department-based split, with an epoch budget large enough that the
  ConvNeXt-Tiny decoders actually converge,
- produce final benchmarks vs the Inception/DeepLab baselines on BDAPPV,
- upload final weights + a model card (limitations, intended use, privacy
  notice) to Hugging Face, pending license confirmation,
- happy to add a minimal CI workflow (flake8 + `train.py --dry-run` on
  synthetic data, no dataset download) once the approach is confirmed.

## Questions for maintainers

1. Any preference among the SegFormer / DeepLab / U-Net families to prioritize?
2. Should final benchmarks be reported on `val`, `test`, or both? Is the
   cross-provider shift protocol (train google → test ign) in scope?
3. Given the ~0.2–0.5% foreground coverage, is **object-level (instance) F1** the
   metric the project wants to optimize and report, rather than mIoU?
4. Is `ign` the right default provider (vs `google`, which the paper baselines
   use)? I picked `ign` for the Etalab 2.0 license — Google imagery carries
   redistribution restrictions that matter when we publish weights.
5. Preferred license for final weights/code — CC-BY, MIT, or both?
