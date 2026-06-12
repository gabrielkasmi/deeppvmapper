# -*- coding: utf-8 -*-

"""
Tile-level geometry helpers.

Key functions:
  save_geotiff               — writes a patch as a geo-referenced GeoTIFF
  masks_to_coordinates       — converts segmentation masks to LAMB93 polygons
  sort_polygons              — assigns polygons to their parent tile (STRtree-accelerated)
  translate_thumbnail_point_to_geo — pixel → LAMB93 for a single point
"""

import os
import glob
import numpy as np
import cv2
from fiona import collection
from shapely.geometry import Polygon
from shapely.strtree import STRtree

try:
    from osgeo import gdal, osr
except ImportError:
    import gdal
    import osr


# ---------------------------------------------------------------------------
# GeoTIFF I/O
# ---------------------------------------------------------------------------

def save_geotiff(R, G, B, xNN, yNN, patch_size, geotransform, filename):
    """
    Saves a patch as a Lambert93 GeoTIFF.

    R, G, B      : (patch_size, patch_size) uint8 arrays
    xNN, yNN     : pixel-space centre of the patch (float)
    patch_size   : int
    geotransform : GDAL geotransform of the parent tile
    filename     : output path
    """
    ulx, xres, _, uly, _, yres = geotransform

    lons = [ulx + (xNN - patch_size / 2) * xres,
            ulx + (xNN + patch_size / 2) * xres]
    lats = [uly + (yNN - patch_size / 2) * yres,
            uly + (yNN + patch_size / 2) * yres]

    xmin, ymin = min(lons), min(lats)
    xmax, ymax = max(lons), max(lats)
    res_x = (xmax - xmin) / float(patch_size)
    res_y = (ymax - ymin) / float(patch_size)

    geotrans = (xmin, res_x, 0, ymax, 0, -res_y)

    dst = gdal.GetDriverByName('GTiff').Create(
        filename, patch_size, patch_size, 3, gdal.GDT_Byte
    )
    dst.SetGeoTransform(geotrans)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2154)
    dst.SetProjection(srs.ExportToWkt())
    dst.GetRasterBand(1).WriteArray(R)
    dst.GetRasterBand(2).WriteArray(G)
    dst.GetRasterBand(3).WriteArray(B)
    dst.FlushCache()
    dst = None


# ---------------------------------------------------------------------------
# Segmentation mask → polygon coordinates
# ---------------------------------------------------------------------------

def masks_to_coordinates(outputs, img_names, img_dir):
    """
    Converts binary segmentation masks to LAMB93 polygon coordinates.

    outputs   : np.ndarray (N, H, W) binary masks
    img_names : list of N GeoTIFF filenames
    img_dir   : directory containing those GeoTIFFs

    Returns {img_name: {polygon_id: ndarray (K, 2) LAMB93}}
    """
    polygons = {}
    _id = 0

    for i, name in enumerate(img_names):
        mask      = outputs[i]
        polygons[name] = {}

        thumbnail = gdal.Open(os.path.join(img_dir, name))
        ulx, xres, _, uly, _, yres = thumbnail.GetGeoTransform()

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            _id += 1
            pts          = contour.squeeze(1).astype(float)
            pts[:, 0]    = pts[:, 0] * xres + ulx
            pts[:, 1]    = pts[:, 1] * yres + uly
            polygons[name][_id] = pts

    return polygons


def sort_polygons(polygons, source_images_dir):
    """
    Assigns each detected polygon to its parent tile.
    Uses an STRtree for fast spatial lookup instead of O(polygons × tiles).

    Returns {tile_name: {polygon_id: {'LAMB93': [...], 'PX': [...]}}}
    """
    dnsSHP = glob.glob(source_images_dir + '/**/dalles.shp', recursive=True)
    if not dnsSHP:
        return {}

    # Build flat list of all annotation polygons
    all_items = []   # (img_name, polygon_id, Polygon object, raw ndarray)
    for img in polygons:
        for pid, ann in polygons[img].items():
            if ann.shape[0] > 2:
                all_items.append((img, pid, Polygon(ann), ann))

    if not all_items:
        return {}

    poly_objects = [item[2] for item in all_items]
    # id() is stable while poly_objects keeps them alive
    id_to_meta   = {id(p): (all_items[k][0], all_items[k][1], all_items[k][3])
                    for k, p in enumerate(poly_objects)}
    tree = STRtree(poly_objects)

    def lamb_to_px(coord, gt):
        ulx, xres, _, uly, _, yres = gt
        return int((coord[0] - ulx) / xres), int((coord[1] - uly) / yres)

    raw_polygons = {}

    with collection(dnsSHP[0], 'r') as shp:
        for rec in shp:
            tile = rec['properties']['NOM'][2:-4]
            path = glob.glob(
                source_images_dir + '/**/{}.jp2'.format(tile), recursive=True
            )
            if not path:
                continue

            Tile = Polygon(rec['geometry']['coordinates'][0])
            ds   = gdal.Open(path[0])
            gt   = ds.GetGeoTransform()

            raw_polygons[tile] = {}
            for idx in tree.query(Tile):           # shapely 2.x returns indices
                cand = poly_objects[idx]
                if Tile.contains(cand):
                    img, pid, ann = id_to_meta[id(cand)]
                    raw_polygons[tile][pid] = {
                        'LAMB93': ann.tolist(),
                        'PX':     [lamb_to_px(pt, gt) for pt in ann],
                    }

    return {t: v for t, v in raw_polygons.items() if v}


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def translate_thumbnail_point_to_geo(point, thumbnail):
    x, y = point
    ulx, xres, _, uly, _, yres = thumbnail.GetGeoTransform()
    return x * xres + ulx, y * yres + uly
