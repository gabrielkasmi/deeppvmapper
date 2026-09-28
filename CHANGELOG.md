# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Training scaffold for candidate segmentation backbones on BDAPPV (refs #11):
  `train.py` / `eval.py`, `configs/`, quick-experiment scripts
  (`scripts/run_quick.sh`, `scripts/upload_results.sh`) and `experiments/`
  result summaries. Weights and runs are gitignored, never committed.
- `scripts/prepare_bdappv.py` — extracts the `img/` + `mask/` layout and
  `annotations.csv` from the Parquet shards the Hub actually serves now. The
  scaffold had been written against the dataset card's documented release
  (`img/`, `mask/`, `annotations.csv`), which no longer exists on the Hub, so
  `train.py` / `eval.py` could not open the data at all. The canonical split is
  taken from the shards' `split` column.
- First real quick-run results (`experiments/results/*.json` + previews) for
  SegFormer-B0, DeepLabV3+ (ConvNeXt-Tiny) and U-Net (ConvNeXt-Tiny), replacing
  the placeholder results table in the PR body.

### Changed
- `configs/train.yaml`: `batch_size` 8 → 3. Batch 8 overcommits 4 GB of VRAM for
  the ConvNeXt-Tiny decoders (measured 6250/6336 MiB peaks — Windows silently
  spills to shared memory rather than raising OOM), batch 4 leaves under 120 MiB
  of headroom, and batch 1 is impossible because the decoders' BatchNorm layers
  raise on a single sample.
- Quick-run protocol subset 1% → 5% (`scripts/run_quick.sh` default): on the
  `ign` split 1% is only ~51 train / ~14 val images, too small to rank
  architectures.
- `requirements-train.txt`: added `pyarrow`, required by `prepare_bdappv.py`.
- `experiments/README.md`: documented the Parquet download and prepare step,
  the corrected download sizes (~4.1 GB for ign train + validation, not the
  ~2.5 GB previously stated), the VRAM/batch-size constraint, and how to invoke
  `run_quick.sh` on Windows.

## [1.0] - 2026-06-17

First tagged stable release — pipeline validated end-to-end.

### Added
- Asynchronous tile decode/prefetch: a background process pool now decodes
  JP2 tiles while the GPU runs classification, instead of loading tiles
  sequentially before each batch. Closes #2.
- `decode_workers` / `decode_stagger_s` config options to tune the decode
  pool size and stagger initial submissions (avoids lockstep bursty waits).

### Changed
- Cleared the leftover single-tile test filter in `config.yml`'s default
  `tiles_list`.
