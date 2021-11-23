# -*- coding: utf-8 -*-

"""
TILES PREPROCESSING

This script takes the raw input images and preprocesses them in order 
to be handled by the model.
Images are processed and stored in a dedicated directory, also passed as input.
"""
import sys 
sys.path.append('../src')

import glob
from fiona import collection
import os
import tiles_processing
import concurrent
import tqdm
import json
import shutil



def initialize_tiles_list(source_dir, dpt):
    """
    returns a dictionnary where each key is the tile name
    and each value is set to False, meaning that none of the 
    tiles have yet been proceeded.
    """

    tiles_list = {}

    dnsSHP = glob.glob(source_dir + "/**/dalles.shp", recursive = True) 

    if not dnsSHP:
        print("Error, the shape file could not be found.")
        raise ValueError
        
    with collection(dnsSHP[0], 'r') as input: # look for the tile that contains the point
        for shapefile_record  in input:
            name = shapefile_record["properties"]["NOM"][2:-4]
            tiles_list[name] = False


    print('There are {} tiles for the departement {}.'.format(len(tiles_list), dpt))
    return tiles_list


class TilesTracker():
    """
    Reads the shapefile of the inputed departement and gets the list of tiles
    Then, during the inference process, the list of tiles is updated and the corresponding 
    folders in the data/thumbnails folder are deleted in order to limit the space taken by
    the process.
    """

    def __init__(self, configuration, dpt, force = False):
        """
        Retrieve the location of the folders from the configuration file.

        arguments: 
        - the configuration file
        - dpt : the number of the departement on which inference is being made.
        - (optional) force : a boolean passed indicating whether at 
        each initialization, the tiles file will be erased (meaning the whole
        inference process will be restarted from scratch)
        """

        self.source_dir = configuration.get('source_images_dir')
        self.thumbnails_dir = configuration.get('thumbnails_dir')
        self.outputs_dir = configuration.get("outputs_dir")

        # Create the thumbnails direcetory if the latter does not exist.
        if not os.path.isdir(self.thumbnails_dir):
            os.mkdir(self.thumbnails_dir)     

        # Open the tiles list file.

        # If the file exists and if force is false, load it.
        if os.path.isfile(os.path.join(self.thumbnails_dir, "tiles_list.json")):

            if not force : 
                with open(os.path.join(self.thumbnails_dir, "tiles_list.json")) as f:
                    tiles_list = json.load(f)
            else:# if force is set on True, initialize the file
                tiles_list = initialize_tiles_list(self.source_dir, dpt)
                with open(os.path.join(self.thumbnails_dir, 'tiles_list.json'), 'w') as f:
                    json.dump(tiles_list, f, indent = 2)

                # also initialize the approximate_coordinates and raw results fildes
                if os.path.exists(os.path.join(self.outputs_dir, "raw_detection_results.json")):
                    os.remove(os.path.join(self.outputs_dir, "raw_detection_results.json"))
                
                if os.path.exists(os.path.join(self.outputs_dir, "approximate_coordinates.json")):
                    os.remove(os.path.join(self.outputs_dir, "approximate_coordinates.json"))

        else: # if the file does not exist create it in all cases.

            tiles_list = initialize_tiles_list(self.source_dir, dpt)
            with open(os.path.join(self.thumbnails_dir, 'tiles_list.json'), 'w') as f:
                json.dump(tiles_list, f, indent = 2)

        # save the list of tiles as an additional attribute.
        self.tiles_list = tiles_list

    def update(self):
        """update the list of tiles based on the approximate_coordinates.json file
        this file is returned at the end of the inference phase, before the postprocessing
        if panels have been found on a given tile, the key still exists but its 
        value is an empty dictionnary.
        """

        # open the file
        approximate_coordinates = json.load(open(os.path.join(self.outputs_dir, "approximate_coordinates.json")))
        completed_tiles = list(approximate_coordinates.keys())

        print("{} tiles have beed proceeded.".format(len(completed_tiles)))

        for tile in completed_tiles: # loop over the tiles for which the approximate coordinates 
                                     # have been computed
            self.tiles_list[tile] = True

        # update the file in the source folder.
            with open(os.path.join(self.thumbnails_dir, 'tiles_list.json'), 'w') as f:
                json.dump(self.tiles_list, f, indent=2)

    def clean(self):
        """cleans the thumbnails directory"""

        # open the file
        approximate_coordinates = json.load(open(os.path.join(self.outputs_dir, "approximate_coordinates.json")))
        completed_tiles = list(approximate_coordinates.keys())

        # remove the folders
        folders = 0
        for tile in completed_tiles:
            if os.path.isdir(os.path.join(self.thumbnails_dir, tile)):
                folders += 1
                shutil.rmtree(os.path.join(self.thumbnails_dir, tile))

        print("{} folders have beed deleted.".format(folders))

    def completed(self):
        """checks whether the tiles_list still contains 
        tiles flagged with False"""

        return bool({k : v for k, v in self.tiles_list.items() if v == False})


class PreProcessing():

    """
    Preprocesses the data. 

    The preprocessing step stores in the folder "thumbnails_dir" 
    all the thumbnails generated from each tile of the departement. 

    The structure is the following :
    - thumbnails_dir
        |
        |- tile_name
        |   |- thumbnail
        |   |- thumbnail
        |- tile_name
        |   |- thumbnail
        |   |- thumbnail
        |- tile_name
        |   |- thumbnail
        |   |- thumbnail
        | ...

    Remark : the preprocessing is always attached to a department. 

    The main preprocessing steps are the following : 

    - get the list of tiles that have not been proceeded
    - generates the thumbnails from each tile, with a given thumbnail
    size 
    """

    def __init__(self, configuration, count):
        """
        initialization of the pre processing step

        configuration is a yaml file
        count is an integer, correspondig to the number of tiles to 
        proceed in each batch. The larger the count, 
        the heavier the data/thumbnails folder.
        """
        # Retrieve the directories for this part
        self.source_dir = configuration.get('source_images_dir')
        self.thumbnails_dir = configuration.get('thumbnails_dir')

        # Parameters for this part
        self.patch_size = configuration.get('patch_size')
        self.count = count

        # Tiles list. The tiles that will be proceeded are the one that are flagged as false
        # Tiles that are marked as 'False'
        full_tiles_list = json.load(open(os.path.join(self.thumbnails_dir, "tiles_list.json")))

        # filter the dictionnary to keep only the tiles for which the value is "False"
        self.tiles_list = {k : v for k,v in full_tiles_list.items() if v == False}

        # number of tiles to proceed (cannot be greated than the len of the tiles_list)
        self.count = min(len(list(self.tiles_list.keys())), count) 
        

    def run(self):
        """preprocess the number of tiles that have been flagged as
        False """

        # the tiles batch is simply a list taken from the complete tiles_list.
        tiles_batch = list(self.tiles_list.keys())[:self.count]

        # debug only : when the process is not run in parallel.
        #for tile in tqdm.tqdm(tiles_batch):
        #    tiles_processing.generate_thumbnails_from_tile(self.source_dir, self.thumbnails_dir, tile, self.patch_size)  

        def f(tile) : # Local function to be put in the threadpoolexecutor
            print('Processing tile {}...'.format(tile))
            return tiles_processing.generate_thumbnails_from_tile(self.source_dir, self.thumbnails_dir, tile, self.patch_size)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(f, tiles_batch)


