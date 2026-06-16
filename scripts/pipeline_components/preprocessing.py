# -*- coding: utf-8 -*-

"""
PREPROCESSING

Reads the tiles list for the current department and resolves jp2 paths
for the next batch of unprocessed tiles.

No thumbnail writing — classification runs fully in-memory (see detection.py).
"""

import os
import glob
import json
import tqdm
from fiona import collection


def initialize_tiles_list(source_dir, target_tiles, dpt):
    """
    Returns {tile_name: False} for every tile to process.
    target_tiles : optional list of tile names from config; None means all tiles.
    """
    tiles_list = {}

    if target_tiles is None:
        dnsSHP = glob.glob(source_dir + '/**/dalles.shp', recursive=True)
        if not dnsSHP:
            raise ValueError('dalles.shp not found in {}'.format(source_dir))
        with collection(dnsSHP[0], 'r') as shp:
            for rec in shp:
                name = rec['properties']['NOM'][2:-4]
                tiles_list[name] = False
    else:
        tiles_list = {tile: False for tile in target_tiles}

    print('There are {} tiles for department {}.'.format(len(tiles_list), dpt))
    return tiles_list


def _build_tile_index(source_dir):
    """
    Builds {tile_name: jp2_path} for every tile in dalles.shp.

    Walks the source directory exactly once (single recursive glob for all
    .jp2 files) instead of once per tile, then matches each shapefile record
    against that lookup. Cached to disk by PreProcessing so it only runs once
    per department, not once per batch.
    """
    dnsSHP = glob.glob(source_dir + '/**/dalles.shp', recursive=True)
    if not dnsSHP:
        raise ValueError('dalles.shp not found in {}'.format(source_dir))

    jp2_by_basename = {
        os.path.basename(p): p
        for p in glob.glob(source_dir + '/**/*.jp2', recursive=True)
    }

    index = {}
    with collection(dnsSHP[0], 'r') as shp:
        for rec in tqdm.tqdm(shp, desc='Indexing tile paths'):
            name = rec['properties']['NOM'][2:-4]
            dns  = rec['properties']['NOM'][2:]
            path = jp2_by_basename.get(dns)
            if path:
                index[name] = path

    return index


class TilesTracker:
    """
    Persists the processing state to disk so the pipeline can resume after a crash.
    """

    def __init__(self, configuration, dpt, force=False):
        self.source_dir = configuration.get('source_images_dir')
        self.temp_dir   = configuration.get('temp_dir')
        self.dpt        = dpt
        self._target    = configuration.get('tiles_list')   # optional subset from config

        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'segmentation'), exist_ok=True)

        tiles_file = os.path.join(self.temp_dir, 'tiles_list_{}.json'.format(dpt))

        if os.path.isfile(tiles_file) and not force:
            with open(tiles_file) as f:
                tiles_list = json.load(f)
        else:
            tiles_list = initialize_tiles_list(self.source_dir, self._target, dpt)
            with open(tiles_file, 'w') as f:
                json.dump(tiles_list, f, indent=2)
            raw = os.path.join(self.temp_dir, 'raw_detection_results.json')
            if os.path.exists(raw):
                os.remove(raw)

        self.tiles_list = tiles_list

    def update(self):
        """Mark processed tiles as done based on raw_detection_results.json."""
        raw_path = os.path.join(self.temp_dir, 'raw_detection_results.json')
        proceeded = json.load(open(raw_path))

        str_dpt  = ('0' + str(self.dpt)) if self.dpt < 10 else str(self.dpt)
        dpt_done = sum(1 for t in proceeded if t[:2] == str_dpt)
        print('{}/{} tiles completed for department {}'.format(
            dpt_done, len(self.tiles_list), self.dpt))

        for tile in proceeded:
            self.tiles_list[tile] = True

        tiles_file = os.path.join(self.temp_dir, 'tiles_list_{}.json'.format(self.dpt))
        with open(tiles_file, 'w') as f:
            json.dump(self.tiles_list, f, indent=2)

    def completed(self):
        """Returns True while there are still unprocessed tiles."""
        return any(not v for v in self.tiles_list.values())


class PreProcessing:
    """
    Resolves jp2 file paths for the next batch of unprocessed tiles.
    Returns {tile_name: jp2_path} consumed by Detection.
    """

    def __init__(self, configuration, count, dpt):
        self.source_dir = configuration.get('source_images_dir')
        self.temp_dir   = configuration.get('temp_dir')
        self.dpt        = dpt

        full_list = json.load(
            open(os.path.join(self.temp_dir, 'tiles_list_{}.json'.format(dpt)))
        )
        pending           = {k: v for k, v in full_list.items() if not v}
        self.tiles_batch  = list(pending.keys())[:min(len(pending), count)]

    def run(self):
        """Returns {tile_name: jp2_path} for the current batch."""
        if not self.tiles_batch:
            return {}

        # Index is built once per department and cached to disk — every
        # subsequent batch (and any resumed run) just loads the JSON instead
        # of re-walking the source directory.
        index_path = os.path.join(self.temp_dir, 'tile_index_{}.json'.format(self.dpt))
        if os.path.isfile(index_path):
            with open(index_path) as f:
                tile_index = json.load(f)
        else:
            tile_index = _build_tile_index(self.source_dir)
            with open(index_path, 'w') as f:
                json.dump(tile_index, f, indent=2)

        tile_paths = {
            name: tile_index[name] for name in self.tiles_batch if name in tile_index
        }

        missing = [name for name in self.tiles_batch if name not in tile_index]
        if missing:
            raise ValueError(
                "{} requested tile(s) not found in dalles.shp / jp2 files under {} "
                "(department {}): {}. Check that 'tiles_list' in config.yml actually "
                "belongs to this department/source_images_dir, or delete "
                "temp/tile_index_{}.json to force a reindex.".format(
                    len(missing), self.source_dir, self.dpt, missing, self.dpt
                )
            )

        print('Batch: {} tiles resolved.'.format(len(tile_paths)))
        return tile_paths
