#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DETECTION PIPELINE

Single entry point. Runs in three stages for a given department:

  1. Initialization  — build auxiliary files (buildings, plants, communes) into temp/
                       Skipped if temp/ already contains them (allows resuming after crash)
  2. Classification  — in-memory tile classification, positive patches saved to temp/segmentation/
  3. Segmentation    — segment positive patches → LAMB93 polygons → temp/segmentation/ deleted
  4. Aggregation     — pypvroof characteristics + building filter → outputs written to data/

On success : temp/ is deleted.
On crash   : temp/ is kept; rerun the same command to resume from where it stopped.
Force full rerun : pass --clean to wipe temp/ before starting.
"""

import sys
import os
import shutil
sys.path.append('scripts/pipeline_components/')
sys.path.append('scripts/src/')

# Patch pypvroof: its utils.py does a bare `from osgeo import gdal` at import time
# which fails when GDAL is installed as `gdal` (not `osgeo`) on some systems.
# Applied once at startup — no manual post-install step needed.
try:
    import pypvroof
    import pypvroof.utils as _pypvroof_utils

    # 1. GDAL patch
    if not hasattr(_pypvroof_utils, '_gdal_patched'):
        try:
            from osgeo import gdal as _gdal
        except ImportError:
            import gdal as _gdal
        _pypvroof_utils.gdal = _gdal
        _pypvroof_utils._gdal_patched = True

    # 2. Data file patch — copy bundled CSV into the install if missing
    _pypvroof_data_dir = os.path.join(os.path.dirname(pypvroof.__file__), 'data')
    _bundled_csv = os.path.join(
        os.path.dirname(__file__), 'pypvroof_data', 'bdappv-metadata.csv'
    )
    _target_csv = os.path.join(_pypvroof_data_dir, 'bdappv-metadata.csv')
    if not os.path.isfile(_target_csv) and os.path.isfile(_bundled_csv):
        os.makedirs(_pypvroof_data_dir, exist_ok=True)
        shutil.copy(_bundled_csv, _target_csv)

except Exception:
    pass  # pypvroof not installed yet — will fail later with a clear error

import warnings
import argparse
import json

import yaml
import torch

import preprocessing
import detection
import segmentation
import aggregation
import data_handlers

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Auxiliary initialization (merged from auxiliary.py)
# ---------------------------------------------------------------------------

def _initialize_auxiliary(configuration, dpt):
    """
    Generates per-department auxiliary files into temp_dir if not already present.
    Safe to call on every run — skips files that already exist.
    """
    temp_dir           = configuration.get('temp_dir')
    source_topo_dir    = configuration.get('source_topo_dir')
    source_commune_dir = configuration.get('source_commune_dir')

    buildings_path = os.path.join(temp_dir, 'buildings_locations_{}.json'.format(dpt))
    plants_path    = os.path.join(temp_dir, 'plants_locations_{}.json'.format(dpt))
    communes_path  = os.path.join(temp_dir, 'communes_{}.json'.format(dpt))

    if not os.path.exists(buildings_path):
        print('Building auxiliary: building locations...')
        buildings = data_handlers.get_buildings_locations(source_topo_dir)
        with open(buildings_path, 'w') as f:
            json.dump(buildings, f, indent=2)

    if not os.path.exists(plants_path):
        print('Building auxiliary: power plant locations...')
        plants = data_handlers.get_power_plants(source_topo_dir)
        with open(plants_path, 'w') as f:
            json.dump(plants, f, indent=2)

    if not os.path.exists(communes_path):
        print('Building auxiliary: communes...')
        communes = data_handlers.get_communes(source_commune_dir, dpt)
        with open(communes_path, 'w') as f:
            json.dump(communes, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # ── Arguments ─────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='Large-scale PV detection pipeline')
    parser.add_argument('--dpt',   required=True, type=int,
                        help='Department number to process')
    parser.add_argument('--count', default=16, type=int,
                        help='Tiles per classification batch (default: 16)')
    parser.add_argument('--clean', action='store_true',
                        help='Wipe temp/ before running (forces full rerun)')
    parser.add_argument('--config', default='config.yml',
                        help='Path to config file (default: config.yml)')
    args = parser.parse_args()

    dpt = args.dpt

    # ── Configuration ──────────────────────────────────────────────────────────
    with open(args.config, 'rb') as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)

    temp_dir    = configuration.get('temp_dir', 'temp')
    outputs_dir = configuration.get('outputs_dir', 'data')

    # Route aux file lookups to temp_dir so everything lives in one place
    configuration['aux_dir'] = temp_dir

    # ── Setup ─────────────────────────────────────────────────────────────────
    if args.clean and os.path.isdir(temp_dir):
        print('--clean: removing {}/'.format(temp_dir))
        shutil.rmtree(temp_dir)

    os.makedirs(temp_dir,    exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    run_classification = configuration.get('run_classification', True)
    run_segmentation   = configuration.get('run_segmentation',   True)
    run_aggregation    = configuration.get('run_aggregation',    True)

    # ── Run ───────────────────────────────────────────────────────────────────
    try:

        # Step 0 — auxiliary files (skipped if already in temp/)
        _initialize_auxiliary(configuration, dpt)

        # Step 1 — classification
        if run_classification:
            tiles_tracker = preprocessing.TilesTracker(configuration, dpt)
            print('Starting classification. Batches of {} tiles.'.format(args.count))
            while tiles_tracker.completed():
                pre  = preprocessing.PreProcessing(configuration, args.count, dpt)
                tile_paths = pre.run()
                det  = detection.Detection(configuration)
                det.run(tile_paths)
                tiles_tracker.update()
            print('Classification complete for department {}.'.format(dpt))

        # Step 2 — segmentation
        if run_segmentation:
            print('Starting segmentation...')
            segmenter = segmentation.Segmentation(configuration, dpt)
            segmenter.run()
            print('Segmentation complete for department {}.'.format(dpt))

        # Step 3 — aggregation
        if run_aggregation:
            print('Starting aggregation...')
            aggregator = aggregation.Aggregation(configuration, dpt)
            aggregator.run()
            print('Aggregation complete.')

        # ── Success: clean up temp/ ────────────────────────────────────────────
        print('Run successful. Cleaning up {}/ ...'.format(temp_dir))
        shutil.rmtree(temp_dir)
        print('Done.')

    except Exception as e:
        print('\nPipeline error: {}'.format(e))
        print('Temporary files kept in {}/ — rerun the same command to resume.'.format(temp_dir))
        raise


if __name__ == '__main__':
    torch.manual_seed(42)
    main()
