#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from tokenize import PlainToken

sys.path.append('scripts/pipeline_components/')
sys.path.append('scripts/src/')


import data_handlers
import helpers
import yaml
import sys
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import wget

"""
Script that generates the auxiliary outputs needed throughout the pipeline. 
This script needs to be run first. It initializes the aux/ directory 
that contains these outputs. 

If the user does not run this script first, a warning is sent in the main script

These auxiliary outputs are :

Extracted from the BDTOPO
- buildings : geographic coordinates of the buidlings in a given departement.
- plants : geographic coordinates of the plants in a given departement.


Extracted from their own folder : 
- IRIS : statistical clusters of 2,000 inhabitants each
- Communes : lowest aggregation level in the validation registries
"""


# Arguments
parser = argparse.ArgumentParser(description = 'Auxiliary files for the large scale detection pipeline')

parser.add_argument('--dpt', default = None, help = "Department to proceed")

args = parser.parse_args()


if args.dpt is not None:
    dpt = args.dpt
    
    if isinstance(dpt, str):
        dpt = int(dpt)
    
else:
    print('Please input a departement number to run the pipeline.')
    raise ValueError

# Load the configuration file
config = 'config.yml'

with open(config, 'rb') as f:
    configuration = yaml.load(f, Loader=yaml.FullLoader)

# Get the folders from the configuration file
aux_dir = configuration.get('aux_dir')

# source directories for the auxiliary outputs
source_commune_dir = configuration.get('source_commune_dir') 
source_topo_dir = configuration.get('source_topo_dir') 
source_images_dir = configuration.get('source_images_dir')

# main script 

# intialize the aux/ directory
if not os.path.isdir(aux_dir):
    os.mkdir(aux_dir)


"""
helper functions for the extraction of the MNS
"""
#Dictionnary of Request Parameters
RP = {
    "MNS" : {
        "KEY" : "altimetrie/",
        "VERSION" : "1.3.0",
        "LAYERS" : "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES.MNS",
        "FORMAT" : "image/jpeg",
        "REQUEST" : "GetMap",
        "EXCEPTIONS" : "text/xml",
        "STYLES" : "",
        "CRS" : "EPSG:2154",
        "WIDTH" : "6250",
        "HEIGHT" : "6250"
    }}

def return_bbox(coord):
    
    left, top = coord
        
    x_shift, y_shift = (6250) * abs(0.2), (6250) * abs(-0.2)
        
    right, bottom = coord[0] + x_shift, coord[1] - y_shift
    
    BB = "{},{},{},{}".format(left, bottom, right, top)
        
    return BB


'''Returns the URL corresponding to the wanted MNS (altitude heatmap) image'''
def get_mns_url(coord):
    KEY = RP["MNS"]["KEY"]
    VERSION = RP["MNS"]["VERSION"]
    FORMAT = RP["MNS"]["FORMAT"]
    REQUEST = RP["MNS"]["REQUEST"]
    EXCEPTIONS = RP["MNS"]["EXCEPTIONS"]
    STYLES = RP["MNS"]["STYLES"]
    CRS = RP["MNS"]["CRS"]
    LAYERS = RP["MNS"]["LAYERS"]
    WIDTH = RP["MNS"]["WIDTH"]
    HEIGHT = RP["MNS"]["HEIGHT"]
    
    BB = return_bbox(coord)
    
    URL = 'https://wxs.ign.fr/'+KEY+'geoportail/r/wms/?'\
        +'LAYERS='+LAYERS+'&'\
        +'EXCEPTIONS='+EXCEPTIONS+'&'\
        +'VERSION='+VERSION+'&'\
        +'FORMAT='+FORMAT+'&'\
        +'REQUEST='+REQUEST+'&'\
        +'STYLES='+STYLES+'&'\
        +'CRS='+CRS+'&'\
        +'BBOX='+BB+'&'\
        +'WIDTH='+WIDTH+'&'\
        +'HEIGHT='+HEIGHT   
        
    return URL    
    
'''Plots the wanted url image.'''
def show_url_img(coord):
    # get the URL
    URL = get_mns_url(coord)
    
    # prepare and load the image
    dnsRDM = 'tmp_IGN_'+str(int((np.random.rand(1)[0])*10**9))+'.jpg'
    
    wget.download(URL,'./'+dnsRDM)
    A=plt.imread('./'+dnsRDM)
    os.remove('./'+dnsRDM)
    return A 


"""
function that requests the WMS server to extract the MNS of the tiles
"""
def get_associated_mns(aux_dir, source_images_dir):
    """
    retrieves the MNS of the departement.
    """

    # load the tile list

    # set up the directory
    mns_dir = os.path.join(aux_dir, 'mns')
    if not os.path.exists(mns_dir):
        os.mkdir(mns_dir)

    # check that the folder for the current departement exists
    dpt_mns_dir = os.path.join(mns_dir, str(dpt))
    if not os.path.exists(dpt_mns_dir):
        os.mkdir(dpt_mns_dir)

    completed_tiles = os.listdir(dpt_mns_dir)

    # get the tiles list of the departement
    tiles = helpers.compute_tiles_coordinates(source_images_dir)

    for tile in tiles.keys():
        # for each tile, check whether it has been competed or not
        if '{}.png'.format(tile) in completed_tiles:
            continue
        else:
            full_tile = compute_mns(tiles[tile])

            tile_name = '{}.png'.format(tile)
            target_directory = os.path.join(dpt_mns_dir, tile_name)
            plt.imsave(target_directory, full_tile)



"""
computes the MNS of a tile
"""
def compute_mns(tile_coords):
    """
    computes the MNS of a tile by requesting chunks on the 
    WMS serveur of the IGN. 

    args :
    -  tile_coords : a list coming from the compute_tiles_coordinates function.
    """

    # request to the MNS the 16 subparts of the tile
    images = {}

    row = -1
    for i in range(4 * 4):
        
        k = i%4
        
        if i%4 == 0:
            row += 1
            
        
        left, top = tile_coords[0][0] # initialize wrt to the upper left corner of the tile.

        x_shift, y_shift = k * (6250) * abs(0.2), row * (6250) * abs(-0.2)
        
        left, top =  left + x_shift, top - y_shift
        
        coord = left, top
                    
        A = show_url_img(coord)
        
        images[i] = A

    # aggregate the parts together into one large tile
    full_tile = np.empty((25000,25000))

    row = -1
    for i in range(4*4):
        
        k = i%4    
        if i%4 == 0:
            row += 1
        
        block = images[i]
        
        y_start, y_end = k * 6250, (k+1) * 6250
        x_start, x_end = row * 6250, (row + 1) * 6250
        
        full_tile[x_start:x_end , y_start:y_end] = block

    return full_tile


def main():

    # buildings
    if not os.path.exists(os.path.join(aux_dir, 'buildings_locations_{}.json'.format(args.dpt))):

        print('Computing the location of the buildings...')
        buildings_locations = data_handlers.get_buildings_locations(source_topo_dir)

        # save the file
        print('Computation complete. Saving the file.')

        with open(os.path.join(aux_dir, 'buildings_locations_{}.json'.format(args.dpt)), 'w') as f:
            json.dump(buildings_locations, f, indent=2)

        print('Done.')

    # power plants
    if not os.path.exists(os.path.join(aux_dir, 'plants_locations_{}.json'.format(args.dpt))):

        # List of power plants
        print('Extracting the localization of the power plants...')
        plants_locations = data_handlers.get_power_plants(source_topo_dir)

        # Saving the file

        with open(os.path.join(aux_dir, 'plants_locations_{}.json'.format(args.dpt)), 'w') as f:
            json.dump(plants_locations, f, indent = 2)

    # Computation for the IRIS : 
#    if not os.path.exists(os.path.join(aux_dir, 'iris_{}.json'.format(args.dpt))):

#        print('Filtering the IRIS attached to departement {}...'.format(args.dpt))
#        iris_location = data_handlers.get_iris(source_iris_dir, args.dpt)

        # save the file
#        print('Computation complete. Saving the file.')

#        with open(os.path.join(aux_dir, 'iris_{}.json'.format(args.dpt)), 'w') as f:
#            json.dump(iris_location, f, indent=2)

        print('Done.')

    # communes
    if not os.path.exists(os.path.join(aux_dir, 'communes_{}.json'.format(args.dpt))):

        print('Filtering the communes attached to departement {}...'.format(args.dpt))
        communes_location = data_handlers.get_communes(source_commune_dir, args.dpt)

        # save the file
        print('Computation complete. Saving the file.')

        with open(os.path.join(aux_dir, 'communes_{}.json'.format(args.dpt)), 'w') as f:
            json.dump(communes_location, f, indent=2)

        print('Done.')

    # MNS of the departement
    # print('Extracting the MNS of the departement ...')
    # get_associated_mns(aux_dir, source_images_dir)
    # print('Done.')


if __name__ == '__main__':

    # run the initialization of the auxiliary files.
    main()