# -*- coding: utf-8 -*-

"""
PANEL SEGMENTATION


the script takes thumbnails from the thumbanails/segmentation folder
and does inference on this set of tiles.

It outputs a file that contains the polygon coordinates 
associated with the panels, tile by tile.



"""
from distutils.command.config import config
import sys
sys.path.append('../src')


import os
import torch
import numpy as np
import tqdm
import torch.nn as nn
import json
import dataset, tiles_processing
import torchvision
from torch.utils.data import DataLoader
import shutil


"""
helper that saves the outputs in a file name stored
in the target directory. 
if the file already exists, append current outputs to 
the existing file.
"""
def save(outputs, target_directory, file_name):
    """
    saves the raw outputs of the model 
    the outputs should be a dictionnary.
    """
# if no file exists : 
    if not os.path.isfile(os.path.join(target_directory, file_name)):
        with open(os.path.join(target_directory, file_name), 'w') as f:
            json.dump(outputs, f, indent=2)
    else:
        # update the file
        # open the file
        previous_outputs = json.load(open(os.path.join(target_directory, file_name)))
        
        # add the latest tiles
        for key in outputs.keys():
            previous_outputs[key] = outputs[key]

        # save the new file
        with open(os.path.join(target_directory, file_name), 'w') as f:

            json.dump(previous_outputs, f, indent=2)

    return None

class Segmentation():
    """
        Given the configuration file, segments the installations with a pretrained
    model.
    """

    def __init__(self, configuration):
        """
        Get the parameters and the directories.
        """

        # Retrieve the directories for this part
        self.temp_dir = configuration.get('temp_dir')
        self.model_dir = configuration.get('model_dir')
        self.aux_dir = configuration.get('aux_dir')
        self.source_images_dir = configuration.get('source_images_dir')

        # Parameters for this part
        self.device = configuration.get('device')

        self.batch_size = configuration.get('seg_batch_size')
        self.threshold = configuration.get('seg_threshold')
        self.num_gpu = configuration.get('num_gpu')

        # inference model
        self.model_name = configuration.get('seg_model')


    def initialization(self):
        """
        does the initialization. 
        returns the model and the device.
        """

        # Setting up the device

        if self.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = self.device

        # setting up the model

        model = torch.load(os.path.join(self.model_dir, self.model_name + '.pth'), map_location = self.device)
        model.eval()

        # parallelize if inputed to do so.
        if self.num_gpu > 1:

            devices = list(range(self.num_gpu))
            
            model = nn.DataParallel(model, device_ids = devices)
            model.to(self.device)

        return model, device

    def inference(self, model, device):
        """
        does inference on the thumbnails
        returns the raw segmentation polygons
        """

        print('Segmenting positive thumbnails ...')

        # transforms : normalize the images
        transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
        ])

        # access the folder and load the data
        dataset_dir = os.path.join(self.temp_dir, 'segmentation')
        data_source = dataset.BDPVSegmentationNoLabels(dataset_dir, transform = transforms)
        inference_data = DataLoader(data_source, batch_size = self.batch_size)

        # outputs file
        outputs, img_names = [], []

        # Inference loop
        for data in tqdm.tqdm(inference_data):

            with torch.no_grad():

                images, names = data

                # forward pass
                predicted = model.forward(images.to(device))
                predictions = predicted['out']
                # min-max scaling
                predictions = (predictions - torch.min(predictions)) / (torch.max(predictions) - torch.min(predictions) + 0.000000001)

                binary_outputs = (predictions >= self.threshold).long().squeeze(1).detach().cpu().numpy()
                outputs.append(binary_outputs)   
                img_names.append(names)

        img_names = list(sum(img_names, ()))
        outputs = np.concatenate(outputs)

        return outputs, img_names

    def convert_to_coordinates(self, outputs, img_names):
        """
        takes the outputs of the inference part and 
        turns it into a dictionnary 
        where each key is a tile and the values are raw 
        polygon coordinates (in pixels and in coordinates)
        """

        polygons = tiles_processing.masks_to_coordinates(outputs, img_names, os.path.join(self.temp_dir, 'segmentation'))

        coordinates = tiles_processing.sort_polygons(polygons, self.source_images_dir)

        # save the outputs without erasing the existing file.
        save(coordinates, self.temp_dir, "raw_segmentation_results.json")

        return None

    def run(self):
        """
        chain all parts together.
        """

        # initialize the model
        model, device = self.initialization()

        # inference
        segmentation_arrays, img_names = self.inference(model, device)

        # transform the segmentation masks into polygon coordinates
        # with respect to their respective tile and in LAMB93
        self.convert_to_coordinates(segmentation_arrays, img_names)

        # Finally convert the detection polygons into pseudo arrays by merging
        # adjacent polygons 

        # once segmentation is complete, remove the directory containing the thumbnails
        # to segment
        shutil.rmtree(os.path.join(self.temp_dir, 'segmentation'))
