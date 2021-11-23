#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DETECTION PIPELINE

This script chains together all components of the detection pipeline.

- pre_processing : loads the raw images and converts them as thumbnails

- detection : runs the detection and extracts the localization based on the CAM of the model

- post_processing : associates the detections with buildings or plants to remove false postives
                    and formats the output as a geojson file.

"""

import sys

sys.path.append('scripts/pipeline_components/')
sys.path.append('scripts/src/')


import preprocessing, detection, postprocessing
import helpers
import yaml
import sys
import geojson
import torch
import os
import argparse


def main(): 
 
    # - - - - - - - STEP 0 - - - - - - -  

    # Parse the arguments
    # Arguments are the following 
    # - run_classification
    # - run_postprocessing
    # - force : indicate whether existing files should be overwritten or not.
    # - dpt : the departement to process.
    # - count : the number of tiles to preprocess at each batch
    # Arguments
    parser = argparse.ArgumentParser(description = 'Large scale detection pipeline')

    parser.add_argument('--force', default = False, help = "Indicates whether the inference process should be started from the beginning", type = bool)
    parser.add_argument('--count', default = 16, help = "Number of tiles to process simultaneoulsy", type=int)

    parser.add_argument('--dpt', default = None, help = "Department to proceed", type=int)

    parser.add_argument('--run_classification', default = None, help = "Whether detection should be done.", type=bool)
    parser.add_argument('--run_postprocessing', default = None, help = "Whether postprocessing should be done.", type=bool)

    args = parser.parse_args()

    count = args.count
    force = args.force

    # Load the configuration file
    config = 'config.yml'

    with open(config, 'rb') as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)

    # Parameters that are specific to the wrapper
    # Overwrite the configuration parameters whenever relevant.

    run_classification = configuration.get('run_classification')

    if args.run_classification is not None:
        run_classification = args.run_classification

    run_postprocessing = configuration.get('run_postprocessing')

    if args.run_postprocessing is not None:
        run_postprocessing = args.run_postprocessing

    save_map = configuration.get("save_map")
    map_center = configuration.get("center_latitude"), configuration.get("center_longitude")

    # department number
    # overwrite the configuration file if necessary.

    dpt = configuration.get("departement_number")
    if args.dpt is not None:
        dpt = args.dpt

    # output directory

    outputs_dir = configuration.get('outputs_dir')
    results_dir = configuration.get('geo_dir')

    # - - - - - - - STEP 1 - - - - - - -  
    # Run parts of the process or all of it

    if run_classification:

        # Initialize the tiles tracker helper, that will keep track of the 
        # tiles that have been completed and those that still need to be proceeded
        tiles_tracker = preprocessing.TilesTracker(configuration, dpt, force = force) 

        while tiles_tracker.completed():
            # While the full list of tiles has not been completed,
            # do the following :
            # 1) Split a batch of unprocessed tiles
            # 2) Do inference and save the inferred locations
            # 3) Update the list of tiles 
            # 4) Clean the thumbnails

            print('Starting pre processing...')

            pre_processing = preprocessing.PreProcessing(configuration, count)
            pre_processing.run()

            print('Preprocessing complete. ')

            print('Strarting detection ...')

            inference = detection.Detection(configuration)
            inference.run()

            print('Detection complete. ')

            # update the tiles tracker and clean the thumbnails folder
            print('Updating and cleaning the tiles list...')

            tiles_tracker.update()

            tiles_tracker.clean()

            print('Complete.')
        
        print('Detection of the tiles on the departement {} complete.'.format(dpt))

    if run_postprocessing:

        print('Starting postprocessing... ')

        post_processing = postprocessing.PostProcessing(configuration, dpt, force)
        post_processing.run()

        print('Postprocessing complete.')

    print('Pipeline completed. All itermediary outputs are in the folder {}.'.format(outputs_dir))

    # - - - - - - - STEP 3 - - - - - - -  
    # Save the map if specified.

    if save_map:

        # open and load the file

        with open(os.path.join(results_dir,'installations_{}.geojson'.format(dpt))) as f:
            installations_features = geojson.load(f)

        # save the file.

        helpers.save_map(results_dir, map_center, installations_features, dpt = dpt)

if __name__ == '__main__':

    # Setting up the seed for reproducibility

    torch.manual_seed(42)

    # Run the pipeline.
    main()