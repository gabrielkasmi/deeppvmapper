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


import preprocessing, detection, postprocessing, segmentation
import helpers
import yaml
import sys
import geojson
import torch
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

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

    parser.add_argument('--count', default = 16, help = "Number of tiles to process simultaneoulsy", type=int)
    parser.add_argument('--force', default = False, help = "Whether inteference should be started over.", type=bool)
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

    run_characterization = configuration.get('run_characterization')
    
    # department number

    if args.dpt is not None:
        dpt = args.dpt
    else:
        print('Please input a departement number to rune the pipeline.')
        raise ValueError

    # directories : 
    # the aux directory contains auxiliary information needed at different stages of inference.
    # the outputs directory stores the results of teh model
    # the temp directory stores the temporary outputs and is erased at the end of inference.

    outputs_dir = configuration.get('outputs_dir')
    aux_dir = configuration.get('aux_dir')
    temp_dir = configuration.get('temp_dir')

    # Check that the aux directory is not empty. 
    # If it is the case, stop the script and tell the user to 
    # run auxiliary inference first.
    if not os.listdir(aux_dir):
        print('No outputs found in the auxiliary directory. Run aux.py before running the main script.')
        raise ValueError

    # - - - - - - - STEP 1 - - - - - - -  
    # Run parts of the process or all of it

    if run_classification:

        # Initialize the tiles tracker helper, that will keep track of the 
        # tiles that have been completed and those that still need to be proceeded
        tiles_tracker = preprocessing.TilesTracker(configuration, dpt, force = args.force) 

        #i = 0

        print('Starting classification. Batches of tiles will be subsequently proceeded.')

        while tiles_tracker.completed():
            # While the full list of tiles has not been completed,
            # do the following :
            # 1) Split a batch of unprocessed tiles
            # 2) Do inference and save the list of thumbnails that are identified 
            #    as positives
            # 3) Update the list of tiles that have been processed 
            # 4) remove the negative images

            #i += 1

            print('iteration : {}'.format(i))

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

            #if i == 3:
            #    break
        
        print('Detection of the tiles on the departement {} complete.'.format(dpt))

    if run_segmentation:
        print('Starting segmentation... ')

        # create the outputs direectory if the latter does not exist
        if not os.path.isdir(outputs_dir):
            os.mkdir(outputs_dir)


        segmenter = segmentation.Segmentation(configuration, args.dpt)
        segmenter.run()

        print('Segmentation of the positive thumbnails of department {} complete.'.format(dpt))

    # A la fin de l'exécution du script, supprimer aussi le fichier avec les auxiliaires (vu qu'on démarre un nv département)

    #if run_postprocessing:

    #    print('Starting postprocessing...')

    #    post_processing = postprocessing.PostProcessing(configuration, dpt, args.force)
    #    post_processing.run()

    #    print('Postprocessing complete.')


        # As this stage : generate pseudo arrays and discard power plants from distributed PV 
        # using the BD TOPO



    # if run_characterization:

        # print('Computing the arrays characteristics...')

        # partie correspondant au stage de YT.




    #if run_formatting:

    # dans cette section, mise en forme de tout. A reprendre.

    #    print('Starting postprocessing... ')

    #    post_processing = postprocessing.PostProcessing(configuration, dpt, force)
    #    post_processing.run()
    # dans le post processing, étapes suivantes : 
    # identifier les détections correspondant à une centrale
    # grouper les détections par batiment
    # fusionner les détections appartenant au même batiment

    #    print('Postprocessing complete.')

    #print('Pipeline completed. All itermediary outputs are in the folder {}.'.format(outputs_dir))

    # - - - - - - - STEP 3 - - - - - - -  
    # Save the map if specified.

    #if save_map:

        # open and load the file

    #    with open(os.path.join(results_dir,'installations_{}.geojson'.format(dpt))) as f:
    #        installations_features = geojson.load(f)

        # save the file.

    #    helpers.save_map(results_dir, map_center, installations_features, dpt = dpt)

if __name__ == '__main__':

    # Setting up the seed for reproducibility

    torch.manual_seed(42)

    # Run the pipeline.
    main()