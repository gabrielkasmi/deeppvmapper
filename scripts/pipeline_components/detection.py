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

import sys

from tqdm.std import TqdmExperimentalWarning 
sys.path.append('../src')


import torch
import os
import dataset, tiles_processing
import tqdm
from torch.nn import functional as F
import json
from torch.utils.data import DataLoader
import numpy as np
import torchvision

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
        self.temp_dir = configuration.get('temp_dir')
        self.thumbnails_dir = os.path.join(self.temp_dir, "classification")
        self.model_dir = configuration.get('model_dir')
        self.source_images_dir = configuration.get('source_images_dir')

        # Parameters for this part
        self.device = configuration.get('device')
        self.batch_size = configuration.get('cls_batch_size')
        self.threshold = configuration.get('cls_threshold')
        self.patch_size = configuration.get('patch_size')

        self.model_name = configuration.get('cls_model')

    def initialization(self):
        """
        Intializes the model. Returns the model and the device
        """

        # Setting up the device

        if self.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = self.device


        # Load the model 
        # Add in the config file
        model_name = self.model_name + '.pth'

        model = torch.load(os.path.join(self.model_dir, model_name), map_location = device)
        # model.to(device)
        model.eval()

        return model, device

    def inference(self, model, device):
        """
        Does inference on the tiles list.

        Returns model_outputs : a dictionnary that lists for each
        tile the name of the thumbnails that have been labelled as positive.
        """

        # exclude the 'tiles_list.json' from the list of tiles to proceed
        # and potential miscelanneous hidden files and folders.
        tiles_names = [item for item in os.listdir(self.thumbnails_dir) if not (item[-5:] == '.json') | (item[0] == '.')]

        print('There are {} tiles to proceed.'.format(len(tiles_names)))
        
        # Dictionnary that will contain the raw outputs.
        # Raw outputs are in the form 
        # {tilename : [thumbnail_name, ...], 
        #  tilename : [thumbnail_name] }
        # where the thumbnail_name corresponds to the name of the thumbnails
        # that have been labelled as postive by the model.

        # transforms : normalize the images
        transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
        ])

        model_outputs = {}
        
        for tile in tiles_names:

            print('Proceeding tile {}...'.format(tile))

            # access the folder and load the data
            dataset_dir = os.path.join(self.thumbnails_dir, tile)
            data_source = dataset.BDPVClassificationNoLabels(dataset_dir, transform = transforms)
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
        # without erasing the existing file.

        # if no file exists : 
        if not os.path.isfile(os.path.join(self.temp_dir, "raw_detection_results.json")):
            with open(os.path.join(self.temp_dir, 'raw_detection_results.json'), 'w') as f:
                json.dump(model_outputs, f, indent=2)
        else:
            # update the file
            # open the file
            previous_outputs = json.load(open(os.path.join(self.temp_dir, "raw_detection_results.json")))
            
            # add the latest tiles
            for tile in model_outputs.keys():
                previous_outputs[tile] = model_outputs[tile]

            # save the new file
            with open(os.path.join(self.temp_dir, 'raw_detection_results.json'), 'w') as f:

                json.dump(previous_outputs, f, indent=2)

        return None

#    def compute_localizations(self, model, model_outputs, device):
#        """
#        Leverages the class activation map to compute the localization of the arrays
#        on the positive thumbnails.
#
#        Returns a dictionnary where for each tile we get the list of coordinates
#        """
#
#        temp_approximate_coordinates = {}
#
#        for tile in model_outputs.keys():
#
#            print('Computing the localizations for the arrays on tile {}...'.format(tile))
#
#            thumbnails_folder = os.path.join(self.thumbnails_dir, tile)
#
#            img_list = tiles_processing.compute_location_dictionary(model_outputs[tile], model, thumbnails_folder, device = device)
#            
#            tile_coords = tiles_processing.extract_estimated_array_coordinates_per_tile(img_list, thumbnails_folder)
#
#            temp_approximate_coordinates[tile] = tile_coords
#
#        return temp_approximate_coordinates

#    def save(self, temp_approximate_coordinates):
#
#        # Save the approximate coordinates file
#        # without erasing the existing file.
#
#        # if no file exists : 
#        if not os.path.isfile(os.path.join(self.outputs_dir, "approximate_coordinates.json")):
#            with open(os.path.join(self.outputs_dir, 'approximate_coordinates.json'), 'w') as f:
#                json.dump(temp_approximate_coordinates, f, indent=2)
#
#        else:
#            # update the file
#            # open the file
#            approximate_coordinates = json.load(open(os.path.join(self.outputs_dir, "approximate_coordinates.json")))
#
#            # add the latest tiles
#            for tile in temp_approximate_coordinates.keys():
#                approximate_coordinates[tile] = temp_approximate_coordinates[tile]
#
#            # save the new file
#            with open(os.path.join(self.outputs_dir, 'approximate_coordinates.json'), 'w') as f:
#                json.dump(approximate_coordinates, f, indent=2)
#
    def run(self):
        """
        chains all parts together
        """

        # get the model and the device
        model, device = self.initialization()

        # do the inference or load the raw results directly.
        self.inference(model, device)

        # compute the coordinates based on the raw outputs
        # temp_approximate_coordinates = self.compute_localizations(model, model_outputs, device)

        # save the computations for this batch and update the main
        # approximate coordinates file
        # self.save(temp_approximate_coordinates)

