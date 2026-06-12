# -*- coding: utf-8 -*-

"""
AGGREGATION

Takes arrays_{dpt}.geojson produced by segmentation, extracts panel
characteristics via pypvroof MetadataExtraction, filters implausible
detections, and writes the final CSV + enriched GeoJSON outputs.
"""

import sys
sys.path.append('../src')

import os
import json
import numpy as np
import geojson
import pandas as pd

import postprocessing_helpers

from pypvroof import MetadataExtraction


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save(outputs, target_directory, file_name):
    path = os.path.join(target_directory, file_name)
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            json.dump(outputs, f, indent=2)
    else:
        existing = json.load(open(path))
        existing.update(outputs)
        with open(path, 'w') as f:
            json.dump(existing, f, indent=2)


def save_geojson(outputs, target_directory, file_name):
    path = os.path.join(target_directory, file_name)
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            geojson.dump(outputs, f, indent=2)
    else:
        existing = geojson.load(open(path))
        existing.update(outputs)
        with open(path, 'w') as f:
            geojson.dump(existing, f, indent=2)


# ---------------------------------------------------------------------------
# Aggregation pipeline
# ---------------------------------------------------------------------------

class Aggregation:
    """
    Characterizes detected PV installations and filters them.

    Steps:
      1. initialize  — load arrays GeoJSON + communes lookup
      2. characterize — run pypvroof MetadataExtraction; add city/tile metadata
      3. filter_installations — remove implausible kWp; optionally filter to buildings
      4. export — write CSV, aggregated CSV, and enriched GeoJSON
    """

    def __init__(self, configuration, dpt):
        self.temp_dir    = configuration.get('temp_dir')
        self.aux_dir     = configuration.get('aux_dir')
        self.outputs_dir = configuration.get('outputs_dir')
        self.img_dir     = configuration.get('source_images_dir')
        self.dpt         = dpt

        self.building      = configuration.get('filter_building', True)
        self.method_tilt   = configuration.get('tilt_method',    'lut')
        self.method_az     = configuration.get('azimuth_method', 'bounding-box')
        self.method_ic     = configuration.get('ic_method',      'clustered')
        self.ic_clusters   = configuration.get('ic_clusters',    4)
        self.constant_tilt = configuration.get('constant_tilt',  27.)

    # ------------------------------------------------------------------

    def initialize(self):
        """Loads arrays GeoJSON and communes lookup."""
        print('Loading data for department {}...'.format(self.dpt))

        communes = json.load(
            open(os.path.join(self.aux_dir, 'communes_{}.json'.format(self.dpt)))
        )
        arrays = geojson.load(
            open(os.path.join(self.outputs_dir, 'arrays_{}.geojson'.format(self.dpt)))
        )
        return arrays, communes

    # ------------------------------------------------------------------

    def characterize(self, arrays, communes):
        """
        Runs pypvroof MetadataExtraction over the arrays GeoJSON, then
        appends city code, tile name, and installation_id.

        Returns a DataFrame with columns:
          surface, tilt, azimuth, kWp, city, lat, lon, tile_name, installation_id
        """
        print('Extracting characteristics (tilt={}, azimuth={}, ic={})...'.format(
            self.method_tilt, self.method_az, self.method_ic))

        p = {
            'tilt-method'        : self.method_tilt,
            'azimuth-method'     : self.method_az,
            'regression-type'    : self.method_ic,
            'regression-clusters': self.ic_clusters,
            'constant-tilt'      : self.constant_tilt,
        }
        extractor = MetadataExtraction(p=p)
        df = extractor.extract_all_characteristics(arrays)

        # pypvroof returns: lon, lat, tilt, azimuth, installed_capacity, surface
        df = df.rename(columns={'installed_capacity': 'kWp'})

        # Add per-installation metadata from the source GeoJSON
        cities, tile_names, inst_ids = [], [], []
        for i, feature in enumerate(arrays['features']):
            coords = np.array(feature['geometry']['coordinates']).squeeze(0)
            center = np.mean(coords, axis=0)          # (lon, lat) in WGS84
            cities.append(postprocessing_helpers.retrieve_city_code(center, communes))
            tile_names.append(feature['properties'].get('tile', ''))
            inst_ids.append(i)

        df['city']            = cities
        df['tile_name']       = tile_names
        df['installation_id'] = inst_ids

        # Guarantee expected column order; create missing ones with NaN
        expected = ['surface', 'tilt', 'azimuth', 'kWp', 'city', 'lat', 'lon',
                    'tile_name', 'installation_id']
        for col in expected:
            if col not in df.columns:
                df[col] = np.nan

        print('Characteristics extraction complete ({} installations).'.format(len(df)))
        return df[expected]

    # ------------------------------------------------------------------

    def filter_installations(self, df, communes):
        """
        1. Drops implausible kWp values (< 1.7 or > 36.1).
        2. Optionally filters to detections that fall on a building and
           merges multiple installations on the same building.
        """
        print('Filtering installations...')

        df = df[(df['kWp'] > 1.7) & (df['kWp'] < 36.1)].copy()

        if not self.building:
            print('Building filter disabled. {} installations kept.'.format(len(df)))
            return df

        annotations    = postprocessing_helpers.reshape_dataframe(df)
        tiles_list     = list(annotations.keys())
        buildings      = json.load(
            open(os.path.join(self.aux_dir, 'buildings_locations_{}.json'.format(self.dpt)))
        )
        sorted_buildings = postprocessing_helpers.assign_building_to_tiles(
            tiles_list, buildings, self.img_dir, self.temp_dir, self.dpt
        )
        df_out = postprocessing_helpers.filter_installations(
            df, annotations, sorted_buildings, communes
        )
        print('Filtering complete. {} installations kept.'.format(len(df_out)))
        return df_out

    # ------------------------------------------------------------------

    def export(self, filtered_installations):
        """
        Writes:
          - characteristics_{dpt}.csv       — full per-installation registry
          - aggregated_characteristics_{dpt}.csv — city-level aggregation
          - arrays_characteristics_{dpt}.geojson — enriched polygons
        """
        # De-duplicate and write registry
        out = postprocessing_helpers.merge_duplicates(filtered_installations)
        out.to_csv(
            os.path.join(self.outputs_dir, 'characteristics_{}.csv'.format(self.dpt)),
            index=False,
        )

        # City-level aggregation
        agg_cap   = out[['kWp', 'city']].groupby('city').sum()
        count     = out[['city', 'kWp']].groupby('city').count().rename(columns={'kWp': 'count'})
        means     = out[['surface', 'city', 'lat', 'lon', 'kWp']].groupby('city').mean()
        means.columns = ['avg_surface', 'lat', 'lon', 'avg_kWp']
        aggregated = pd.concat([agg_cap, count, means], axis=1)
        aggregated = aggregated[['count', 'kWp', 'avg_surface', 'avg_kWp', 'lat', 'lon']]
        aggregated.to_csv(
            os.path.join(self.outputs_dir, 'aggregated_characteristics_{}.csv'.format(self.dpt))
        )

        # Enriched GeoJSON
        characteristics = pd.read_csv(
            os.path.join(self.outputs_dir, 'characteristics_{}.csv'.format(self.dpt))
        )
        arrays = geojson.load(
            open(os.path.join(self.outputs_dir, 'arrays_{}.geojson'.format(self.dpt)))
        )
        postprocessing_helpers.associate_characteristics_to_pv_polygons(
            characteristics, arrays, self.outputs_dir, self.dpt
        )

    # ------------------------------------------------------------------

    def run(self):
        arrays, communes = self.initialize()
        characteristics  = self.characterize(arrays, communes)
        filtered         = self.filter_installations(characteristics, communes)
        self.export(filtered)
