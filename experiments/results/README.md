# Results — what belongs here

Small, git-committable experiment outputs only:

- `<run>.json` — eval.py summaries (one per quick run, written by
  `scripts/run_quick.sh` via `--out`)
- `previews/<run>/*.png` — input | GT | prediction triplets (a handful per run)

Never put checkpoints, full logs, or the dataset here — see the commit policy
in [`experiments/README.md`](../README.md).

## Result JSON schema (written by eval.py)

| key | meaning |
|---|---|
| `model_name` | key from configs/model.yaml |
| `params_M` | model size in millions of parameters |
| `input_size` | square training/eval image size (px) |
| `avg_epoch_time_s` | mean epoch wall time during training |
| `val_mIoU`, `val_pixelF1`, `val_instanceF1` | validation metrics (prefix follows `--split`) |
| `inference_fps` | single-image forward passes/s at `input_size` |
| `checkpoint_path` | local path to the evaluated checkpoint (not committed) |

The draft PR table is filled from these files (see `pr_body.md`).
