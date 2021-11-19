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


def main(): 
 
    # - - - - - - - STEP 0 - - - - - - -  

    # Parse the arguments
    # Arguments are the following 
    # - run_preprocessing
    # - run_detection
    # - run_postprocessing
    # - force : indicate whether existing files should be overwritten or not.
    # - dpt : the departement to process.

    # Load the configuration file
    config = 'config.yml'

    with open(config, 'rb') as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)

    # Parameters that are specific to the wrapper

    run_preprocessing = configuration.get('run_preprocessing')
    run_detection_step = configuration.get('run_main_detection')
    run_postprocessing = configuration.get('run_postprocessing')

    save_map = configuration.get("save_map")
    map_center = configuration.get("center_latitude"), configuration.get("center_longitude")

    # department number to save the output map

    dpt = configuration.get("departement_number")

    # output directory

    outputs_dir = configuration.get('outputs_dir')
    results_dir = configuration.get('geo_dir')

    # Recap of the parameters for the detection pipeline

    print("""The detection pipeline is set up as follows: \n 
    - preprocessing : {}\n
    - main detection : {}\n
    - postprocessing : {}\n
    """.format(run_preprocessing, run_detection_step, run_postprocessing))

    # Load the component-specific parameters in order to display them
    # to the user.

    run_detection =  configuration.get('run_detection')
    postprocessing_initialization = configuration.get('postprocessing_initialization')

    print("""The options have been set as follows: \n

    - run_detection (does the inference on the tiles and saves the raw output) : {}
    - postprocessing_initialization (run all steps of the initialization in the postprocessing) : {}

    """.format(run_detection, postprocessing_initialization)
    )


    # - - - - - - - STEP 1 - - - - - - -  
    # Run parts of the process or all of it

    if run_preprocessing:

        print('Starting pre processing...')

        pre_processing = preprocessing.PreProcessing(configuration)
        pre_processing.run()

        print('Preprocessing complete. ')

    if run_detection_step:

        print('Strarting detection ...')

        inference = detection.Detection(configuration)
        inference.run()

        print('Detection complete. ')

    if run_postprocessing:

        print('Starting postprocessing... ')

        post_processing = postprocessing.PostProcessing(configuration)
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
    # Setting up the seed

    torch.manual_seed(42)

    # Run the pipeline.
    main()