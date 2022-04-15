#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Automatically generates a map from a results file output.
Returns the results in the form of a html map.
"""

import argparse
import pandas as pd
import numpy as np
import os
import geojson
import json
from pyproj import Transformer
import folium


# Arguments
parser = argparse.ArgumentParser(description = 'Computation of the accuracy')

parser.add_argument('--dpt', default = None, help = "Department to proceed", type=int)
parser.add_argument('--maille', default = 'commune', help = "Geographical scale to consider : IRIS or commune", type=str)
parser.add_argument('--filter', default = True, help = "Whether building processing should be applied", type=bool)
parser.add_argument('--directory', default = 'validation',help = 'where the outputs are stored and processed', type = str)


args = parser.parse_args()


if args.dpt is not None:
    dpt = args.dpt
else:
    print('Please input a departement number to rune the script.')
    raise ValueError


# open the file
filename = "accuracy_{}_{}.json".format(args.maille, args.dpt)
results = json.load(open(os.path.join(args.directory, filename)))


"""
function that converts the coordinates to GPS
"""
def convert_to_gps(coordinates):
    """
    takes an array, returns an array
    """
    
    transformer = Transformer.from_crs(2154, 4326, always_xy = True)

    # array that stores the output coordinates
    out_coordinates = np.empty(coordinates.shape)

    # do the conversion and store it in the array
    converted_coords = transformer.itransform(coordinates)

    for i, c in enumerate(converted_coords):

        out_coordinates[i, :] = c

    
    return out_coordinates

"""
converts the results as a geojson file
"""
def create_geojson(results, to_gps = True):
    """
    generates a geojson from the results
    """
    
    features = []

    for item in results.keys():
        
        if to_gps :
            contour = convert_to_gps(np.array(results[item]['maille_coords']))
        else:
            contour = np.array(results[item]["maille_coords"])
            
        polygon = geojson.Polygon([[(c[0], c[1]) for c in contour]])       
        features.append(geojson.Feature(geometry=polygon, properties={"item" : item}))

    feature_collection = geojson.FeatureCollection(features)
    
    return feature_collection

"""
computes the dataframe with the merged results
"""
def accuracy_dataframe(results, filtered = True):
    """
    computes the accuracy by considering the absolute difference
    between the number of detections and target number
    """
    
    out = {}
    
    if filtered:
        key = "detections"
    else:
        key = "unfiltered_detections"
    
    for item in results.keys():
        
        detections = results[item][key]
        target = results[item]["targets"]
        
        discrepancy = abs(detections - target)
        
        out[item] = discrepancy
        
    # convert as a dataframe and return it
    output = pd.DataFrame.from_dict(out, orient = 'index').reset_index()
    output.columns = ['item', 'discrepancy']
    
    return output
    

def main():
    """
    main script, converts the results into a map
    """

    m = folium.Map(location=[45.818701443200275, 4.804099378122765], zoom_start=9)

    geodata = create_geojson(results)
    data = accuracy_dataframe(results, filtered = args.filter)

    folium.Choropleth(
        geo_data = geodata,#geojson.load(open("geodata_results.geojson")),                  #json
        name ='discrepancy',                  
        data = data,                     
        columns = ['item', 'discrepancy'], #columns to work on
        key_on ='properties.item',
        fill_color ='YlOrRd',     #I passed colors Yellow,Green,Blue
        fill_opacity = 0.7,
        line_opacity = 0.2,
    legend_name = "Detection gap"
    ).add_to(m)

    folium.LayerControl().add_to(m)

    name = "results_{}_{}.html".format(args.maille, args.dpt)
    m.save(os.path.join(args.directory, name))


if __name__ == '__main__':

    main()