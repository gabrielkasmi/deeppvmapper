# -*- coding: utf-8 -*-

# fucntions that filter the locatinos to look for on the tile.

import json
import itertools
import glob
from fiona import collection
import gdal
from shapely.geometry import Point, Polygon
import geopandas as gpd
import numpy as np


def compute_centers(width, height, size, step):
    """
    computes the coordinates (in pixels) of the centers of the thumbnails

    args: 
    - width, height : the width and height of the tile
    - size : the size of the thumbnail
    - step : the step size between two centers

    returns:
    - centers : a list of tuples of locations (in pixels)
    """

    if width == height:
        
        # start and end point
        start = size // 2
        end = width - (size // 2)

        count = int((end - start) / step) + 1 # number of steps
        steps = [start + i * step for i in range(count)] # all intermediary points
        steps.append(end) # add the last point

        # return the centers
        return [p for p in itertools.product(steps, steps)]

def pixel_to_lambert(point, geotransform):
    """
    converts a pixel point into a Lambert93 coordinate
    specifies whether we are dealing with an x or y point
    with the argument 'type'.
    """

    ulx, xres, _, uly, _, yres  = geotransform

    x, y = point

    x_c = ulx + x * xres
    y_c =  uly + y * yres

    return x_c, y_c

def lambert_to_pixel(point, geotransform):

    ulx, xres, _, uly, _, yres  = geotransform

    x_c, y_c = point.xy

    x_c = x_c[0]
    y_c = y_c[0]

    x = int(round((x_c - ulx) / xres, 2))
    y = int(round((y_c - uly) / yres, 2))

    return x, y

def filter_locations(tile, tile_name, buildings, p):
    """
    returns list of coordinates (pixel) that should be proceeded

    args
    tile: the gdal tile
    tile_name:  the name of the tile
    buiddings : the dictionnary containing the buildings
    p : a dictionnary of parameters

    returns : 
    coordinates : a list of tuples of coordinates (in pixels)
    """

    # get the parameters:
    buffer = p['buffer']
    size = p['size']
    step = p['step']

    width, height = tile.RasterXSize, tile.RasterYSize
    geotransform = tile.GetGeoTransform()

    # compute the centers (in pixels)
    centers = compute_centers(width, height, size, step)
    # convert as coordinates
    coordinates = [
        pixel_to_lambert(p, geotransform) for p in centers
    ]

    # retrieve the buildings and convert it as a GeoSeries
    gpd_buildings = gpd.GeoSeries([
    Polygon(np.array(buildings[tile_name][k]['coordinates'])).buffer(buffer) for k in buildings[tile_name].keys()
    ])

    # convert the points as a series
    gpd_points = gpd.GeoSeries(
        [Point(gc) for gc in coordinates]
    )

    # 
    point_gdf = gpd.GeoDataFrame({'geometry': gpd_points})
    poly_gdf = gpd.GeoDataFrame({'geometry': gpd_buildings})

    filtered = gpd.overlay(point_gdf, poly_gdf, how='intersection')
    filtered.set_crs('epsg:2154') # set the coordinates system (Lambert 93)

    return [lambert_to_pixel(filtered['geometry'][i], geotransform) for i in range(filtered.shape[0])], filtered

def open_tile(folder, tile_name):

    dnsSHP = glob.glob(folder + "/**/dalles.shp", recursive = True)

    with collection(dnsSHP[0], "r") as input:
        for shapefile_record  in input:

            if shapefile_record['properties']['NOM'][2:-4] == tile_name:
                dns=shapefile_record['properties']['NOM'][2:] #open the corresponding tile

                dnsJP2=glob.glob(folder + "/**/" + dns,recursive = True)[0]

    return gdal.Open(dnsJP2)
