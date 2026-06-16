# -*- coding: utf-8 -*-

"""
DETECTION

In-memory classification pipeline.

For each tile in the batch:
  1. Read the full jp2 into a numpy array (one GDAL call, no temp thumbnails)
  2. Build an InMemoryTileDataset and classify all patches in batches
  3. Save only positive patches as GeoTIFFs directly to temp/segmentation/
  4. Record results in raw_detection_results.json

This eliminates the thousands of disk writes/reads that dominated preprocessing time.
"""

import os
import json
import time
import shutil
import multiprocessing
import numpy as np
import torch
import tqdm
import torchvision
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from torch.nn import functional as F

try:
    from osgeo import gdal
except ImportError:
    import gdal


def _init_decode_worker():
    """
    Runs once per worker process when the decode pool starts.

    A single GDAL/OpenJPEG decode call only ever drives ~1.5-2 CPU cores for
    this JP2 profile, regardless of how many cores are available or how
    OPJ_NUM_THREADS / GDAL_NUM_THREADS are set (confirmed via mpstat: one
    thread pegged at ~97-98%, a second one partially loaded, everything else
    idle -- seen identically on a 48-core box and inside a 6-vCPU container).
    So instead of one decode trying and failing to spread across all cores,
    several decodes run concurrently as separate OS processes -- each capped
    to 1 internal thread here so N processes don't oversubscribe the CPU quota
    by each also trying to spawn their own internal thread pool.
    """
    os.environ['OPJ_NUM_THREADS']  = '1'
    os.environ['GDAL_NUM_THREADS'] = '1'


def _decode_tile_to_disk(jp2_path, out_path):
    """
    Decodes one JP2 tile and writes it back out as an uncompressed GeoTIFF.

    Runs in a separate OS process (see _init_decode_worker for why). Returns
    the cache file path rather than the decoded array: shipping a ~2GB array
    back through process IPC (pickling) would eat a meaningful chunk of the
    gain, whereas the path + small metadata is essentially free to pickle.
    """
    ds           = gdal.Open(jp2_path)
    geotransform = ds.GetGeoTransform()
    width        = ds.RasterXSize
    height       = ds.RasterYSize
    tile_arr     = ds.ReadAsArray()          # the expensive part: JP2 decode
    ds           = None
    if tile_arr.ndim == 2:                   # single-band edge case
        tile_arr = np.stack([tile_arr, tile_arr, tile_arr])

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_path, width, height, 3, gdal.GDT_Byte)
    out_ds.SetGeoTransform(geotransform)
    for b in range(3):
        out_ds.GetRasterBand(b + 1).WriteArray(tile_arr[b])
    out_ds.FlushCache()
    out_ds = None

    return out_path, geotransform, width, height


def _prepare_batch(dataset, idx_batch):
    """
    Builds one classification batch (stacked tensor + names + original indices).

    Runs in a background thread so batch b+1's slicing/padding/normalize work
    (pure CPU/numpy) overlaps with the GPU forward pass on batch b, instead of
    happening strictly before it (see Detection.run).
    """
    images, names, indices = [], [], []
    for idx in idx_batch:
        image, name, orig_idx = dataset[idx]
        images.append(image)
        names.append(name)
        indices.append(orig_idx)
    return torch.stack(images), names, indices


def _build_patch_meta(width, height, patch_size):
    """Returns list of (xOffset, yOffset, xNN, yNN) for every patch in the tile."""
    x_shifts = int(width  / patch_size) + 1
    y_shifts = int(height / patch_size) + 1
    x_max    = width  - patch_size / 2
    y_max    = height - patch_size / 2

    meta = []
    row  = -1
    for i in range(x_shifts * y_shifts):
        if i % x_shifts == 0:
            row += 1
        xNN = min(patch_size / 2 + patch_size * (i % x_shifts), x_max)
        yNN = min(patch_size / 2 + patch_size * row, y_max)
        meta.append((int(xNN - patch_size / 2), int(yNN - patch_size / 2), xNN, yNN))
    return meta


class Detection:
    """In-memory tile classification."""

    def __init__(self, configuration):
        self.temp_dir       = configuration.get('temp_dir')
        self.model_dir      = configuration.get('model_dir')
        self.device         = configuration.get('device')
        self.batch_size     = configuration.get('cls_batch_size')
        self.threshold      = configuration.get('cls_threshold')
        self.patch_size     = configuration.get('patch_size')
        self.model_name     = configuration.get('cls_model')
        # Number of JP2 tiles decoded concurrently (separate processes, see
        # _init_decode_worker). Size this to roughly match the CPU quota you
        # actually have, not the core count of the underlying host -- on a
        # 6 vCPU container where one decode drives ~1.5-2 cores, 3 concurrent
        # decodes is the sweet spot, not 8-16.
        self.decode_workers = configuration.get('decode_workers', 3)
        # Gap (seconds) between the n_workers initial decode submissions.
        # Without this, all n_workers start at t=0, finish together (same
        # duration each), and -- because each consumed result is replaced
        # immediately -- stay locked in step for the whole run: one big
        # decode_wait every n_workers tiles instead of a small one on every
        # tile (confirmed in production logs: wait=59.5s then 0.0s x3, repeat).
        # Staggering the initial submissions ~evenly across one decode's
        # duration breaks that lockstep so completions arrive one at a time.
        # Default of 35s approximates decode_time / decode_workers from
        # observed runs (~113s / 3) -- if logs still show big bursty waits,
        # nudge this up towards that ratio; if tiles are slower to start for
        # no benefit, nudge it down.
        self.decode_stagger_s = configuration.get('decode_stagger_s', 35)

    def initialization(self):
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        model  = torch.load(
            os.path.join(self.model_dir, self.model_name + '.pth'),
            map_location='cpu',      # load on CPU first to avoid CUDA/MPS mismatch
            weights_only=False,
        )
        model.to(device)
        model.eval()
        return model, device

    def run(self, tile_paths):
        """
        tile_paths : {tile_name: jp2_path} from PreProcessing.run()
        """
        model, device = self.initialization()

        # Import here to stay consistent with sys.path setup in main.py
        from dataset       import InMemoryTileDataset
        from tiles_processing import save_geotiff

        seg_dir  = os.path.join(self.temp_dir, 'segmentation')
        os.makedirs(seg_dir, exist_ok=True)

        # Rolling cache for decoded tiles, bounded to decode_workers files at a
        # time (~2GB each) -- wiped at the start of every run, files removed
        # as soon as they're consumed so disk usage never grows past the window.
        cache_dir = os.path.join(self.temp_dir, 'tile_cache')
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)

        transform = torchvision.transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std =(0.229, 0.224, 0.225),
        )

        model_outputs = {}
        print('Starting classification. {} tiles to process.'.format(len(tile_paths)))

        tile_names = list(tile_paths.keys())
        jp2_paths  = list(tile_paths.values())
        n_tiles    = len(jp2_paths)
        n_workers  = max(1, min(self.decode_workers, n_tiles)) if n_tiles else 1

        # Decode pool: up to n_workers tiles decoded concurrently in separate
        # OS processes (see _init_decode_worker/_decode_tile_to_disk for why
        # processes instead of one more thread). pending holds the lookahead
        # window of in-flight decode futures, depth n_workers -- as soon as one
        # is consumed, the next not-yet-submitted tile is queued behind it.
        # mp_context='spawn' (not the Linux default 'fork'): self.initialization()
        # has already loaded the model onto CUDA by this point, and forking a
        # process with an initialized CUDA context is a known source of flaky
        # crashes/hangs even when the child never touches CUDA itself. spawn
        # re-imports cleanly instead of copying the parent's memory; the extra
        # startup cost is one-time (workers are reused for the whole run).
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_decode_worker,
                                  mp_context=multiprocessing.get_context('spawn')) as tile_pool, \
             ThreadPoolExecutor(max_workers=1) as batch_executor:

            pending  = deque()
            next_idx = 0

            def _submit_next():
                nonlocal next_idx
                if next_idx < n_tiles:
                    cache_path = os.path.join(cache_dir, '{}.tif'.format(tile_names[next_idx]))
                    pending.append(tile_pool.submit(_decode_tile_to_disk, jp2_paths[next_idx], cache_path))
                    next_idx += 1

            # Stagger the initial fill instead of firing all n_workers at
            # once (see decode_stagger_s docstring in __init__): each sleep
            # offsets one worker's start time, so their completions land
            # spread out instead of bunched -- the offset persists on every
            # later cycle too, since each result is replaced immediately on
            # consumption, inheriting the same relative timing.
            for k in range(n_workers):
                _submit_next()
                if k < n_workers - 1:
                    time.sleep(self.decode_stagger_s)

            for i, tile_name in enumerate(tile_names):
                print('Processing tile {}...'.format(tile_name))

                # Time spent blocked here is decode time NOT hidden by the
                # lookahead window (expect ~0 once the window is warmed up and
                # the decode pool's throughput keeps pace with classification).
                t0 = time.perf_counter()
                cache_path, geotransform, width, height = pending.popleft().result()
                decode_wait_s = time.perf_counter() - t0

                _submit_next()

                # Cached file is an uncompressed GeoTIFF -- this is a plain
                # fread, not a JP2/wavelet decode.
                ds       = gdal.Open(cache_path)
                tile_arr = ds.ReadAsArray()
                ds       = None
                os.remove(cache_path)   # keep disk usage bounded to the window

                patch_meta = _build_patch_meta(width, height, self.patch_size)
                dataset    = InMemoryTileDataset(
                    tile_arr, patch_meta, self.patch_size, geotransform, transform=transform
                )

                n_patches      = len(dataset)
                batch_idx_list = [
                    list(range(b, min(b + self.batch_size, n_patches)))
                    for b in range(0, n_patches, self.batch_size)
                ]

                model_outputs[tile_name] = []

                # Prefetch: while the GPU classifies batch b, a background thread
                # slices/pads/normalizes batch b+1 from tile_arr, so CPU prep overlaps
                # with GPU compute instead of happening strictly before it.
                pending_batch = (
                    batch_executor.submit(_prepare_batch, dataset, batch_idx_list[0])
                    if batch_idx_list else None
                )

                t1 = time.perf_counter()
                for b, idx_batch in enumerate(tqdm.tqdm(batch_idx_list, desc=tile_name)):
                    inputs, patch_names, indices = pending_batch.result()

                    if b + 1 < len(batch_idx_list):
                        pending_batch = batch_executor.submit(
                            _prepare_batch, dataset, batch_idx_list[b + 1]
                        )

                    with torch.no_grad():
                        inputs  = inputs.to(device)
                        probs   = F.softmax(model(inputs), dim=1)[:, 1]
                        # Single GPU->CPU sync for the whole batch, instead of one
                        # sync per positive patch (see Detection.run docstring history).
                        pos_idx = torch.where(probs >= self.threshold)[0].cpu().tolist()

                    for j in pos_idx:
                        name     = patch_names[j]
                        orig_idx = indices[j]
                        xOffset, yOffset, xNN, yNN = patch_meta[orig_idx]

                        patch = tile_arr[:, yOffset:yOffset + self.patch_size,
                                          xOffset:xOffset + self.patch_size].copy()
                        # Pad if at tile boundary
                        ph = self.patch_size - patch.shape[1]
                        pw = self.patch_size - patch.shape[2]
                        if ph > 0 or pw > 0:
                            patch = np.pad(patch, [(0, 0), (0, ph), (0, pw)])

                        save_geotiff(
                            patch[0], patch[1], patch[2],
                            xNN, yNN, self.patch_size, geotransform,
                            os.path.join(seg_dir, name),
                        )
                        model_outputs[tile_name].append(name)

                classify_s = time.perf_counter() - t1
                # Threshold loosened from 1s to 5s now that submissions are
                # staggered: a steady few-second wait per tile is the expected
                # residual (decode_time/decode_workers is still slightly above
                # classify_s), not a sign the stagger isn't working.
                hidden = '' if i < n_workers else (
                    ' [fully hidden by decode pool]' if decode_wait_s < 5
                    else ' [decode NOT fully hidden -> still the bottleneck]'
                )
                print('  decode_wait={:.1f}s, classify={:.1f}s{}'.format(
                    decode_wait_s, classify_s, hidden
                ))

                del tile_arr   # free memory before next tile

        shutil.rmtree(cache_dir, ignore_errors=True)
        self._save_results(model_outputs)
        print('Classification complete.')

    def _save_results(self, model_outputs):
        raw_path = os.path.join(self.temp_dir, 'raw_detection_results.json')
        if not os.path.isfile(raw_path):
            with open(raw_path, 'w') as f:
                json.dump(model_outputs, f, indent=2)
        else:
            existing = json.load(open(raw_path))
            existing.update(model_outputs)
            with open(raw_path, 'w') as f:
                json.dump(existing, f, indent=2)
