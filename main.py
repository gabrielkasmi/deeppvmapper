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


import preprocessing
import detection, segmentation, aggregation, carbon
import yaml
import sys
import torch
import os
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

def main(): 
 
    # - - - - - - - STEP 1 : INITIALIZATION - - - - - - -  

    # Parse the arguments
    parser = argparse.ArgumentParser(description = 'Large scale detection pipeline')

    parser.add_argument('--count', default = 16, help = "Number of tiles to process simultaneoulsy", type=int)
    parser.add_argument('--dpt', default = None, help = "Department to proceed", type=int)

    parser.add_argument('--run_classification', default = None, help = "Whether detection should be done.", type=bool)
    parser.add_argument('--run_segmentation', default = None, help = "Whether segmentation should be done.", type=bool)
    parser.add_argument('--run_postprocessing', default = None, help = "Whether postprocessing should be done.", type=bool)

    args = parser.parse_args()

    # Load the configuration file
    config = 'config.yml'

    with open(config, 'rb') as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)

    # Parameters that are specific to the wrapper
    # Overwrite the configuration parameters whenever relevant.

    run_classification = configuration.get('run_classification')

    run_segmentation = configuration.get('run_segmentation')
    
    run_aggregation = configuration.get('run_aggregation')
    
    # department number

    if args.dpt is not None:
        dpt = args.dpt
    else:
        print('Please input a departement number to run the pipeline.')
        raise ValueError

    # directories : 
    # the aux directory contains auxiliary information needed at different stages of inference.
    # the outputs directory stores the results of teh model
    # the temp directory stores the temporary outputs and is erased at the end of inference.

    outputs_dir = configuration.get('outputs_dir')
    aux_dir = configuration.get('aux_dir')
    carbon_dir = configuration.get('carbon_dir')

    # Check that the aux directory is not empty. 
    # If it is the case, stop the script and tell the user to 
    # run auxiliary inference first.
    if not os.listdir(aux_dir):
        print('Auxiliary directory not found. Run auxiliary.py before running the main script.')
        raise ValueError

    # also check that the files corresponding to the departements exist. Otherwise raise an error
    if not os.path.exists(os.path.join(aux_dir, "buildings_locations_{}.json".format(args.dpt))):
        print('No auxiliary files associated to the directory found in the {} directory. run auxiliary.py before running the main script.'.format(aux_dir))
        raise ValueError

    # if the carbon directory does not exist, create it
    if not os.path.isdir(carbon_dir):
        os.mkdir(carbon_dir)

    # - - - - - - - STEP 2 : EXECUTION - - - - - - -  

    if run_classification:

        # initialize the energy consumption tracker
        #tracker, startDate = carbon.initialize()
        #tracker.start()

        # Initialize the tiles tracker helper, that will keep track of the 
        # tiles that have been completed and those that still need to be proceeded
        tiles_tracker = preprocessing.TilesTracker(configuration, dpt) 

        i = 0

        print('Starting classification. Batches of tiles will be subsequently proceeded.')

        while tiles_tracker.completed():
            # While the full list of tiles has not been completed,
            # do the following :
            # 1) Split a batch of unprocessed tiles
            # 2) Do inference and save the list of thumbnails that are identified 
            #    as positives
            # 3) Update the list of tiles that have been processed 
            # 4) remove the negative images

            i += 1

            print('Starting pre processing...')

            pre_processing = preprocessing.PreProcessing(configuration, args.count, args.dpt)
            pre_processing.run()

            print('Preprocessing complete. ')

            print('Starting detection ...')

            inference = detection.Detection(configuration)
            inference.run()

            print('Detection complete. ')

            # update the tiles tracker and clean the thumbnails folder
            print('Updating and cleaning the tiles list...')

            tiles_tracker.update()

            tiles_tracker.clean()

            print('Complete.')

            # end the energy consumption tracker

            #if i == 3:
            #    break
        
        print('Detection of the tiles on the departement {} complete.'.format(dpt))

        # save the carbon instances
        # tracker.stop() # stop the tracker
        # endDate = datetime.now()
        # carbon.add_instance(startDate, endDate, tracker, carbon_dir, dpt, 'cls')
        
    if run_segmentation:

        # initialize the energy consumption tracker
        # tracker, startDate = carbon.initialize()
        # tracker.start()

        print('Starting segmentation... ')

        # create the outputs direectory if the latter does not exist
        if not os.path.isdir(outputs_dir):
            os.mkdir(outputs_dir)


        
        segmenter = segmentation.Segmentation(configuration, args.dpt)
        segmenter.run()

        print('Segmentation of the positive thumbnails of department {} complete.'.format(dpt))

        # save the carbon instances
        #tracker.stop() # stop the tracker
        #endDate = datetime.now()
        #carbon.add_instance(startDate, endDate, tracker, carbon_dir, dpt, 'seg')

    if run_aggregation:

        # initialize the energy consumption tracker
        #tracker, startDate = carbon.initialize()
        #tracker.start()

        print('Starting aggregation...')

        aggregator = aggregation.Aggregation(configuration, dpt)
        aggregator.run()

        print('Aggregation complete.')

        # save the carbon instances
        #tracker.stop() # stop the tracker
        #endDate = datetime.now()
        #carbon.add_instance(startDate, endDate, tracker, carbon_dir, dpt, 'agg')
        
    # Cleaning the temporary directories. 
    #clean = postprocessing.Cleaner(configuration)
    #clean.run()

if __name__ == '__main__':

    # Setting up the seed for reproducibility
    torch.manual_seed(42)

    # Run the pipeline.
    main()    