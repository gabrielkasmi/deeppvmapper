# -*- coding: utf-8 -*-

"""
PANEL DETECTION

This script takes the thumbnails as input and does the inference on the tiles
So far this script detects arrays with a low threshold as the postprocessing 
step removes false positives.
The outputs are stored in a dictionnary named `approximate_coordinates.json`
with the structure :
    approximate_coordinates != {
        tile : installation_id : [pixel_coordinates, geograpic_coordinates]
    }
the pixel coordinates are expressed wrt the upper left corner of the tile

Detection is done in two parts : 
- Identfying the positively labelled thumbnails
- Leveraging the CAM to localize the array on the thumbnail
"""

import torch
import os
from src import dataset, tiles_processing
import tqdm
from torch.nn import functional as F
import json
from torch.utils.data import DataLoader
import numpy as np
import sys
sys.path.append('../src')

class Detection():

    """
    Given the configuration file, detects the installations with a pretrained
    model.


    """

    def __init__(self, configuration):
        """
        Get the parameters and the directories.
        """

        # Retrieve the directories for this part
        self.thumbnails_dir = configuration.get('thumbnails_dir')
        self.outputs_dir = configuration.get('outputs_dir')
        self.model_dir = configuration.get('model_dir')
        self.source_images_dir = configuration.get('source_images_dir')

        # Parameters for this part
        self.device = configuration.get('device')
        self.batch_size = configuration.get('batch_size')
        self.threshold = configuration.get('threshold')
        self.patch_size = configuration.get('patch_size')
        self.run_detection = configuration.get('run_detection')


    def initialization(self):
        """
        Intializes the model. Returns the model and the device
        """

        # Setting up the device

        if self.device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = self.device

        # Create the outputs directory if the latter does not exist

        try:
            os.mkdir(self.outputs_dir)
        except:
            pass

        # Load the model 
        # In the model directory, we look for the .pth file. 
        for item in os.listdir(self.model_dir):
            if item[-4:] == '.pth':
                model_name = item
                break

        model = torch.load(os.path.join(self.model_dir, model_name))
        model.to(device)
        model.eval()

        return model, device

    def inference(self, model, device):
        """
        Does inference on the tiles list.

        Returns model_outputs : a dictionnary that lists for each
        tile the name of the thumbnails that have been labelled as positive.
        """

        tiles_names = os.listdir(self.thumbnails_dir)

        print('There are {} tiles to proceed.'.format(len(tiles_names)) )
        
        # Dictionnary that will contain the raw outputs.
        # Raw outputs are in the form 
        # {tilename : [thumbnail_name, ...], 
        #  tilename : [thumbnail_name] }
        # where the thumbnailname correspond to the name of the thumbnails
        # that have been labelled as postive by the model.

        model_outputs = {}
        
        for tile in tiles_names:

            print('Proceeding tile {}...'.format(tile))

            # access the folder and load the data
            dataset_dir = os.path.join(self.thumbnails_dir, tile)
            data_source = dataset.BDPVNoLabels(dataset_dir)
            inference_data = DataLoader(data_source, batch_size = self.batch_size)

            # outputs per tile
            model_outputs[tile] = []

            # Inference loop
            for inputs, names in tqdm.tqdm(inference_data):

                with torch.no_grad():

                    inputs = inputs.to(device)
                    outputs = F.softmax(model(inputs), dim = 1)[:,1] # Get thee probability to have an array on the image
                    predictions = (outputs >= self.threshold).long().detach().cpu().numpy() # Thresold to get a binary vector

                    # store the name of the images that have been marked as positive
                    positive_indices = np.where(predictions == 1)[0]

                    for index in positive_indices:
                        model_outputs[tile].append(names[index])

        # save the raw model outputs

        with open(os.path.join(self.outputs_dir, 'raw_detection_results.json'), 'w') as f:
            json.dump(model_outputs, f)

        return model_outputs

    def compute_localizations(self, model, model_outputs, device):
        """
        Leverages the class activation map to compute the localization of the arrays
        on the positive thumbnails.

        Returns a dictionnary where for each tile we get the list of coordinates
        """

        approximate_coordinates = {}

        for tile in model_outputs.keys():

            print('Computing the localizations for the arrays on tile {}...'.format(tile))

            thumbnails_folder = os.path.join(self.thumbnails_dir, tile)

            img_list = tiles_processing.compute_location_dictionary(model_outputs[tile], model, thumbnails_folder, device = device)

            tile_coords = tiles_processing.extract_estimated_array_coordinates_per_tile(img_list, self.source_images_dir, patch_size = self.patch_size) 

            approximate_coordinates[tile] = tile_coords

        # Save the approximate coordinates file
        with open(os.path.join(self.outputs_dir, 'approximate_coordinates.json'), 'w') as f:
            json.dump(approximate_coordinates, f, indent=2)

    def run(self):
        """
        chains all parts together
        """

        # get the model and the device
        model, device = self.initialization()

        # do the inference or load the raw results directly.
        # TODO. Detection si le fichier de sortie existe et passer
        if self.run_detection:
            model_outputs = self.inference(model, device)
        else:
            with open(os.path.join(self.outputs_dir, "raw_deteection_results.json")) as f:
                model_outputs = json.load(f)

        # compute the coordinates based on the raw outputs
        self.compute_localizations(model, model_outputs, device)

