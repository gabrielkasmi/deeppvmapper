# -*- coding: utf-8 -*-

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
from pyproj import Transformer
import tqdm
import glob
from fiona import collection
import pandas as pd
import os
import json
import geojson


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def convert(coordinates, to_gps=True):
    """Converts Lambert93 ↔ WGS84."""
    doubled = False
    transformer = Transformer.from_crs(2154, 4326, always_xy=True) if to_gps \
        else Transformer.from_crs(4326, 2154, always_xy=True)

    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    if coordinates.shape[0] == 1:
        coordinates = coordinates.squeeze(0)
    elif len(coordinates.shape) == 3:
        return np.empty((3, 3))
    if len(coordinates.shape) == 1:
        coordinates = np.vstack([coordinates, coordinates])
        doubled = True

    out = np.empty(coordinates.shape)
    for i, c in enumerate(transformer.itransform(coordinates)):
        out[i] = c

    return out.tolist()[0] if doubled else out.tolist()


def intersect_or_contain(p1, p2):
    return p1.contains(p2) or p1.intersects(p2)


# ---------------------------------------------------------------------------
# City / commune helpers
# ---------------------------------------------------------------------------

def retrieve_city_code(center, communes):
    """Returns the code_insee of the commune containing center (lon, lat)."""
    Center = Point(center)
    for commune in communes:
        raw_coords = communes[commune]['coordinates']
        if len(raw_coords) > 1:
            for rc in raw_coords:
                arr = np.array(rc)
                if arr.shape[0] == 1:
                    Rc = Polygon(arr.squeeze(0))
                    if Rc.contains(Center):
                        return communes[commune]['properties']['code_insee']
        else:
            arr = np.array(raw_coords).squeeze(0)
            if arr.shape[0] == 1:
                Coords = Polygon(arr.squeeze(0))
                if Coords.contains(Center):
                    return communes[commune]['properties']['code_insee']
    return None


# ---------------------------------------------------------------------------
# Tile geometry helpers
# ---------------------------------------------------------------------------

def compute_tiles_coordinates(tiles_dir):
    """Returns {tile_name: coordinates} from dalles.shp."""
    dnsSHP = glob.glob(tiles_dir + '/**/dalles.shp', recursive=True)
    if not dnsSHP:
        raise ValueError('dalles.shp not found in {}'.format(tiles_dir))
    items = {}
    with collection(dnsSHP[0], 'r') as shp:
        for rec in tqdm.tqdm(shp):
            name = rec['properties']['NOM'][2:-4]
            items[name] = rec['geometry']['coordinates']
    return items


def assign_building_to_tiles(tiles_list, buildings, tiles_dir, temp_dir, dpt):
    """
    Returns {tile: {building_id: building_data}}.
    Uses STRtree for O(tiles × log(buildings)) lookup.
    Result is cached to disk.
    """
    cache = os.path.join(temp_dir, 'sorted_buildings_{}.json'.format(dpt))
    if os.path.exists(cache):
        return json.load(open(cache))

    tiles_coordinates = compute_tiles_coordinates(tiles_dir)

    # Build STRtree over all buildings
    building_ids   = list(buildings.keys())
    building_polys = []
    for bid in building_ids:
        try:
            building_polys.append(Polygon(buildings[bid]['coordinates']))
        except Exception:
            building_polys.append(None)

    valid_ids   = [bid for bid, p in zip(building_ids, building_polys) if p is not None]
    valid_polys = [p   for p       in building_polys                   if p is not None]
    id_to_bid   = {id(p): valid_ids[k] for k, p in enumerate(valid_polys)}
    tree = STRtree(valid_polys)

    items = {}
    for tile in tqdm.tqdm(tiles_list):
        items[tile] = {}
        if tile not in tiles_coordinates:
            continue
        coords = np.array(tiles_coordinates[tile]).squeeze(0)
        Tile   = Polygon(coords)
        for idx in tree.query(Tile):           # shapely 2.x returns indices
            cand = valid_polys[idx]
            if Tile.intersects(cand):
                bid = id_to_bid[id(cand)]
                items[tile][bid] = buildings[bid]

    with open(cache, 'w') as f:
        json.dump(items, f)
    return items


# ---------------------------------------------------------------------------
# DataFrame / annotation helpers
# ---------------------------------------------------------------------------

def reshape_dataframe(df):
    """Converts a characteristics DataFrame to {tile: {installation_id: {WGS, LAMB93}}}."""
    tiles_list = np.unique(df['tile_name'].values).tolist()
    print('Annotations found on {} tiles.'.format(len(tiles_list)))
    annotations = {tile: {} for tile in tiles_list}
    for i in tqdm.tqdm(df.index):
        coordinates = df['lon'][i], df['lat'][i]
        tile        = df['tile_name'][i]
        iid         = df['installation_id'][i]
        annotations[tile][iid] = {
            'WGS'   : list(coordinates),
            'LAMB93': convert(np.array(coordinates), to_gps=False),
        }
    return annotations


def filter_installations(df, annotations, sorted_buildings, communes):
    """
    Keeps only installations that fall on a building; merges multiple
    installations on the same building into one row.
    """
    clustered = {}
    for tile in annotations:
        if tile not in sorted_buildings:
            continue
        for building_id, building in sorted_buildings[tile].items():
            Building = Polygon(building['coordinates'])
            for iid, inst in annotations[tile].items():
                pt = Point(inst['LAMB93'])
                if Building.contains(pt):
                    clustered.setdefault(building_id, []).append(iid)

    rows = []
    for building_id, iids in clustered.items():
        if len(iids) == 1:
            rows.append(df.loc[iids[0]].values.tolist())
        else:
            sub     = df.loc[iids]
            surface = sub['surface'].sum()
            kWp     = sub['kWp'].sum()
            lat, lon = sub[['lat', 'lon']].values.mean(axis=0)
            city    = sub.iloc[0]['city']
            if pd.isna(city):
                city = retrieve_city_code((lon, lat), communes)
            rows.append([
                surface,
                sub.iloc[0]['tilt'],
                sub.iloc[0]['azimuth'],
                kWp,
                city,
                lat, lon,
                sub.iloc[0]['tile_name'],
                sub.iloc[0]['installation_id'],
            ])

    return pd.DataFrame(rows, columns=[
        'surface', 'tilt', 'azimuth', 'kWp', 'city', 'lat', 'lon', 'tile_name', 'installation_id'
    ])


def merge_duplicates(characteristics):
    """Removes duplicate rows; merges rows sharing the same installation_id."""
    characteristics = characteristics.drop_duplicates()
    agg = (
        characteristics.groupby('installation_id', sort=False)
        .agg(
            surface  =('surface',   'sum'),
            tilt     =('tilt',      'first'),
            azimuth  =('azimuth',   'first'),
            kWp      =('kWp',       'sum'),
            city     =('city',      'first'),
            lat      =('lat',       'mean'),
            lon      =('lon',       'mean'),
            tile_name=('tile_name', 'first'),
        )
        .reset_index()
    )
    return agg[['surface', 'tilt', 'azimuth', 'kWp', 'city', 'lat', 'lon', 'tile_name', 'installation_id']]


def associate_characteristics_to_pv_polygons(characteristics, arrays, data_dir, dpt):
    """Writes arrays_characteristics_{dpt}.geojson with per-polygon properties."""
    features = []
    for installation_id in characteristics['installation_id'].values:
        array = arrays['features'][installation_id]
        row   = characteristics[characteristics['installation_id'] == installation_id].iloc[0]
        city  = row['city']
        props = {
            'city'   : 'nan' if pd.isna(city) else int(city),
            'surface': float(row['surface']),
            'tilt'   : float(row['tilt']),
            'azimuth': float(row['azimuth']),
            'kWp'    : float(row['kWp']),
        }
        features.append(geojson.Feature(
            geometry=geojson.Polygon(array['geometry']['coordinates']),
            properties=props,
        ))
    fc = geojson.FeatureCollection(features)
    with open(os.path.join(data_dir, 'arrays_characteristics_{}.geojson'.format(dpt)), 'w') as f:
        geojson.dump(fc, f)
