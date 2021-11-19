# -*- coding: utf-8 -*-

"""
TILES PREPROCESSING

This script takes the raw input images and preprocesses them in order 
to be handled by the model.
Images are processed and stored in a dedicated directory, also passed as input.
"""
import sys 
sys.path.append('../src')

#import tqdm
import glob
from fiona import collection
import os
from PIL import Image
import warnings
import tiles_processing
import concurrent
import tqdm

"""
A class that performs the preprocessing
"""
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

    - get the list of tiles
    - generates the thumbnails from each tile, with a given thumbnail
    size 

    
    """

    def __init__(self, configuration):
        """
        initialization of the pre processing step

        configuration is a yaml file
        """
        # Retrieve the directories for this part
        self.source_dir = configuration.get('source_images_dir')
        self.thumbnails_dir = configuration.get('thumbnails_dir')

        # Parameters for this part
        self.patch_size = configuration.get('patch_size')

    def run(self):

        # Create the thumbnails directory if the latter does not exist

        if not os.path.isdir(self.thumbnails_dir):
            os.mkdir(self.thumbnails_dir)            

        # Retrieve the tiles list

        tiles_list = []

        dnsSHP = glob.glob(self.source_dir + "/**/dalles.shp", recursive = True) 

        if not dnsSHP:
            print("Error, the shape file could not be found.")
            raise ValueError
            
        with collection(dnsSHP[0], 'r') as input: # look for the tile that contains the point
            for shapefile_record  in input:
                name = shapefile_record["properties"]["NOM"][2:]
                tiles_list.append(name)


        print('There are {} tiles to proceed.'.format(len(tiles_list)))

        ## TODO. Add a small helper that loops over the target ditectory and sees which images have already been processed.


        #for tile in tqdm.tqdm(tiles_list[:2]):
        #    tiles_processing.generate_thumbnails_from_tile(self.source_dir, self.thumbnails_dir, tile, self.patch_size)  

        def f(tile) : # Local function to be put in the threadpoolexecutor
            print('Processing tile {}...'.format(tile))
            return tiles_processing.generate_thumbnails_from_tile(self.source_dir, self.thumbnails_dir, tile, self.patch_size)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(f, tiles_list)


