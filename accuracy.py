#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('scripts/pipeline_components/')
sys.path.append('scripts/src/')


import data_handlers
import helpers
import yaml
import tqdm
import sys
import os
import geojson
from shapely.geometry import Polygon
import argparse
import numpy as np
import glob
from fiona import collection
import pandas as pd
import json
from pyproj import Transformer


"""
Script that computes the accuracy.
Uses the registry from RTE to do the computations
Returns a dictionnary with the number of (filtered) detections
per maille (iris/commune) and the overall averages.
"""


# Arguments
parser = argparse.ArgumentParser(description = 'Computation of the accuracy')

parser.add_argument('--dpt', default = None, help = "Department to proceed", type=int)
parser.add_argument('--maille', default = 'iris', help = "Geographical scale to consider : IRIS or commune", type=str)
parser.add_argument('--filter', default = True, help = "Whether building processing should be applied", type=bool)

parser.add_argument('--source_dir', default = '../ipes-fosphor/files',help = 'location of the ground truth registry', type = str)
parser.add_argument('--target_dir', default = 'validation',help = 'where the outputs are stored', type = str)


args = parser.parse_args()


if args.dpt is not None:
    dpt = args.dpt
else:
    print('Please input a departement number to rune the script.')
    raise ValueError

# Load the configuration file
config = 'config.yml'

with open(config, 'rb') as f:
    configuration = yaml.load(f, Loader=yaml.FullLoader)

# Get the folders from the configuration file
aux_dir = configuration.get('aux_dir')

# source directories for the auxiliary outputs
if args.maille == "iris":
    source_dir = configuration.get('source_iris_dir') 
elif args.maille == "commune":
    source_dir = configuration.get('source_commune_dir') 
else:
    print('Wrong maille imputed. Please input "iris" or "commune".')
    raise ValueError


source_topo_dir = configuration.get('source_topo_dir') 
source_images_dir = configuration.get('source_images_dir')

data_dir = configuration.get('outputs_dir')

"""
handles the conversions between lambert and gps
"""
def convert(coordinates, to_gps = True):
    "converts lambert to wgs coordinates"

    if to_gps:
        transformer = Transformer.from_crs(2154, 4326, always_xy = True)
    else:
        transformer = Transformer.from_crs(4326, 2154, always_xy = True)
        
    # reshape the coordinates
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    
    if coordinates.shape[0] == 1:         
        coordinates = np.array(coordinates).squeeze(0)
        
    elif len(coordinates.shape) == 3: 
        out_coordinates = np.empty((3,3))
        return out_coordinates
        
    # array that stores the output coordinates
    out_coordinates = np.empty(coordinates.shape)

    # do the conversion and store it in the array
    converted_coords = transformer.itransform(coordinates)

    for i, c in enumerate(converted_coords):

        out_coordinates[i, :] = c

    # new instance
    # out_coords.append(out_coordinates.tolist())
        
    return out_coordinates.tolist()


def intersect_or_contain(p1, p2):
    """
    check whether p1 contains or intersects p2
    """

    return p1.contains(p2) or p1.intersects(p2)


"""
Function that formats the .geojson if necessary
"""
def reshape_geojson(file):

    """
    takes a geojson output file as input and returns it in the form of a json file.
    notably, explicits the tiles that have been proceeded.
    """

    tiles_list = []

    # first loop to get the list of annotations
    for id_annotation in tqdm.tqdm(range(len(file['features']))):
        annotation = file["features"][id_annotation]
        
        tile = annotation['properties']['tile']

        tiles_list.append(tile)

    tiles_list = list(set(tiles_list))
    print("Annotations have been found on {} tiles.".format(len(tiles_list)))
    # tiles_list = ['69-2020-0810-6535-LA93-0M20-E080', '69-2020-0805-6525-LA93-0M20-E080', '69-2020-0815-6555-LA93-0M20-E080', '69-2020-0845-6510-LA93-0M20-E080', '69-2020-0815-6580-LA93-0M20-E080', '69-2020-0825-6560-LA93-0M20-E080', '69-2020-0810-6570-LA93-0M20-E080', '69-2020-0805-6560-LA93-0M20-E080', '69-2020-0855-6520-LA93-0M20-E080', '69-2020-0840-6530-LA93-0M20-E080', '69-2020-0815-6515-LA93-0M20-E080', '69-2020-0845-6515-LA93-0M20-E080', '69-2020-0810-6530-LA93-0M20-E080', '69-2020-0815-6520-LA93-0M20-E080', '69-2020-0805-6550-LA93-0M20-E080', '69-2020-0835-6520-LA93-0M20-E080', '69-2020-0835-6550-LA93-0M20-E080', '69-2020-0800-6565-LA93-0M20-E080', '69-2020-0830-6515-LA93-0M20-E080', '69-2020-0835-6505-LA93-0M20-E080']

    
    annotations = {tile : {} for tile in tiles_list}



    for id_annotation in tqdm.tqdm(range(len(file['features']))):
        annotation = file["features"][id_annotation]
        
        tile = annotation['properties']['tile']

        if tile in tiles_list:

        #print(tile)
                            
            reversed_coordinates = np.array(annotation['geometry']['coordinates']).squeeze(0) 
            coordinates = np.empty(reversed_coordinates.shape)
            coordinates = reversed_coordinates[:,[1, 0]]  
            
            annotations[tile][id_annotation] = {}

            annotations[tile][id_annotation]['WGS'] = coordinates.tolist()
            annotations[tile][id_annotation]['LAMB93'] = convert(reversed_coordinates, to_gps = False)    


    return annotations

"""
filters the geographical unit (iris or commune)
"""
def get_geographical_units(maille, tiles_list, source_dir, dpt):
    """
    gets the list of IRIS or communes to consider
    associates 
    """


    if maille == 'iris':
        # récupérer la liste des iris à considérer
        # Retrieve the geographical information on the communes and IRIS

        dnsSHP = glob.glob(source_dir + "/CONTOURS-IRIS.shp", recursive = True) 


        maille_list = []
        maille_coords = {}


        with collection(dnsSHP[0], 'r') as input: # look for the tile that contains the point
            for shapefile_record  in tqdm.tqdm(input):       
                
                source_coords = np.array(shapefile_record['geometry']['coordinates'])
                
                # consider only the conform coordinates.
                if source_coords.shape[0] == 1:
                    coordinates = source_coords.squeeze(0)
                    Coordinates = Polygon(coordinates)
                    
                    iris_name = shapefile_record['properties']['CODE_IRIS']
                    
                    for tile in tiles_list.keys():
                                        
                        Tile = Polygon(tiles_list[tile]['coordinates'])
                        
                        if Tile.contains(Coordinates):
                            
                            maille_list.append(float(iris_name))
                            maille_coords[float(iris_name)] = coordinates #convert(source_coords)

    elif maille == 'communes':
        # récupérer la liste des communes à considérer


        maille_list = []
        maille_coords = {}


        dnsSHP = glob.glob(source_dir + "/communes-20210101.shp", recursive = True) 


        with collection(dnsSHP[0], 'r') as input: # look for the tile that contains the point
            for shapefile_record  in tqdm.tqdm(input):
                
                code_postal = shapefile_record['properties']['insee']
                
                if code_postal[:2] == str(dpt): #filter in order to avoid proceding all communes

                    commune_shape = np.array(convert(np.array(shapefile_record['geometry']['coordinates']), to_gps = False))
                    if not commune_shape.shape == ((3,3)): # consider only conform shapes
                        Coordinates = Polygon(commune_shape)

                        for tile in tiles_list.keys():

                            Tile = Polygon(tiles_list[tile]['coordinates'])

                            if Tile.contains(Coordinates):

                                maille_list.append(int(code_postal))
                                maille_coords[int(code_postal)] = commune_shape #convert(source_coords)

    else:
        print('Input "iris" or "commune" as argument for --maille.')
        raise ValueError

    return maille_list, maille_coords

'''
main function of the script
'''
def main():

    # STEP 1 : get the list of tiles to proceed and the file to evaluate
    # open the file
    filename = "arrays_{}.geojson".format(args.dpt)

    raw_annotations = geojson.load(open(os.path.join(data_dir, filename)))
    annotations = reshape_geojson(raw_annotations) # reshape
    tiles_list = data_handlers.get_relevant_tiles(source_images_dir, list(annotations.keys())) # list of the tiles + coordinates



    # Step 2: retrive the list of iris or communes to consider. 
    _, maille_coords = get_geographical_units(args.maille, tiles_list, source_dir, args.dpt)

    # Step 3 : 
    # open the file of RTE 
    # and retrieve the list of IRIS/Communes to consider.
    print('Opening the reference registry ... This can take a while ...')
    installations_source = pd.read_excel(os.path.join(args.source_dir, 'INSTALLATIONS_DATES_032021.xlsx'))
    print('Loading complete.')

    if args.maille == 'iris':
        clef = 'Maille Iris'
    elif args.maille == "commune":
        clef = 'CodeINSEEProd'
    else:
        print('Wrong maille imputed. Please input "iris" or "commune".')
        raise ValueError

    # restricts to the target geograhpical units 
    # and until the end of the year of publication of the images.
    installations_cible = installations_source[installations_source[clef].isin(list(maille_coords.keys()))]

    date = source_images_dir[-4:] # retrieve the date from the source image dir name.

    end_date = np.datetime64('{}-12-31'.format(date))

    installations_cible = installations_cible[installations_cible['DateMiseEnServiceRaccordement'] <= end_date]    
    
    # STEP 4 - count the number of target installations
    count_installation = {}

    for code in maille_coords.keys():
        nb_inst = installations_cible[installations_cible[clef] == code].shape[0]
        
        count_installation[code] = nb_inst

    # STEP 5 - count the number of detected installations
    clustered_annotations = {}

    for iris in maille_coords.keys():
        
        clustered_annotations[iris] = {}
        Iris = Polygon(maille_coords[iris])

        for tile in annotations.keys():

            for annotation_id in annotations[tile].keys():

                coordinates = annotations[tile][annotation_id]['LAMB93']

                Array = Polygon(coordinates)


                if intersect_or_contain(Iris,Array):

                    if iris not in clustered_annotations.keys():

                        clustered_annotations[iris] = {}
                        clustered_annotations[iris][annotation_id] = coordinates

                    else:
                        clustered_annotations[iris][annotation_id] = coordinates
        
    # STEP 6 - count the number of detections
    count_detections = {}

    for iris in clustered_annotations.keys():
        
        count_detections[iris] = len(clustered_annotations[iris])

    # if we need to filter the detections by buildings
    if args.filter == True:

        # get the building list
        buildings = json.load(open(os.path.join(aux_dir, 'buildings_locations_{}.json'.format(dpt))))
        building_in_tiles = helpers.assign_building_to_tiles(tiles_list, buildings)

        # filter the detections
        print('Filtering the polygons by buildings...')
        merged_polygons = {}
        filtered_polygons = {}

        for tile in tqdm.tqdm(tiles_list.keys()):
            
            merged_polygons[tile] = {}
            
            for annotation_id in annotations[tile].keys():
                
                coordinates = annotations[tile][annotation_id]['LAMB93']
                
                Array = Polygon(coordinates)
                
                for building_id in building_in_tiles[tile].keys():
                    
                    building_coords = building_in_tiles[tile][building_id]['coordinates']
                    Building = Polygon(building_coords)
                    
                    if intersect_or_contain(Building, Array):
                        
                        merged_polygons[tile][building_id] = building_coords
                        filtered_polygons[annotation_id] = annotations[tile][annotation_id]['WGS']

        # cluster per geograhpical unit, but now considereing the merged_polygons (with the building coords)
        clustered_buildings = {}

        for iris in maille_coords.keys():
            
            clustered_buildings[iris] = {}
            Iris = Polygon(maille_coords[iris])

            for tile in annotations.keys():

                for annotation_id in merged_polygons[tile].keys():
                    coordinates = merged_polygons[tile][annotation_id]

                    Array = Polygon(coordinates)

                    if intersect_or_contain(Iris,Array):

                        if iris not in clustered_buildings.keys():

                            clustered_buildings[iris] = {}
                            clustered_buildings[iris][annotation_id] = coordinates

                        else:
                            clustered_buildings[iris][annotation_id] = coordinates
        # count
        count_buildings = {}

        for iris in clustered_buildings.keys():
            
            count_buildings[iris] = len(clustered_buildings[iris])

    
    # format the results
    results = {}
    for maille in maille_coords.keys():

        results[maille] = {}

        # keys : maille_coords, detections_coords, detections, targets

        # coordinates of the geographical unit
        results[maille]['maille_coords'] = maille_coords[maille].tolist()

        # add the keys to the dictoinnary
        # if the filtering is enabled, we also report the number of unfiltered detections
        if args.filter == True:
            results[maille]['detection_coords'] = {}
                       
            for _id in clustered_buildings[maille].keys():

                results[maille]["detection_coords"][_id] = clustered_buildings[maille][_id]

            results[maille]["detections"] = count_buildings[maille]
            results[maille]["unfiltered_detections"] = count_detections[maille]

        else:

            results[maille]['detection_coords'] = {}

            for _id in clustered_annotations[maille].keys():

                results[maille]["detection_coords"][_id] = clustered_annotations[maille][_id]            
            
            results[maille]["detections"] = count_detections[maille]

        # targets
        results[maille]["targets"] = count_installation[maille]


    # Format the files and export them
    print('Exporting the files...')

    with open(os.path.join(args.target_dir, 'accuracy_{}_{}.json'.format(args.maille, dpt)), 'w') as f :

        json.dump(results, f)

    print('Evaluation completed. Outputs are writted in /{}.'.format(args.target_dir))

if __name__ == '__main__':

    main()