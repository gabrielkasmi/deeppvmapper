#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Generates a dataset based on source images and labels in 
a specified folder.

Can generate datasets based on three modalities : 
- Raw conversion of images into thumbnails
- Conversion with application of filters on the thumbnails : 
    - Random assignation 
    - Deterministic assignation

Can control the subset of images that is generated
"""

# Libraries
import argparse
import os
import tqdm
import json
import random
import gdal
import matplotlib.pyplot as plt
import concurrent
from skimage import io
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import warnings
import shutil
from distutils.dir_util import copy_tree

warnings.filterwarnings("ignore")


# Arguments of the script 
parser = argparse.ArgumentParser(description = 'Dataset creation')

# non_empty (bool) : only focuses on the tiles that contain at least one array.
parser.add_argument('--non_empty', default = True, help = "only focuses on the tiles that contain at least one array", type = bool)

# size (int) the size of the thumbnails to be generated
parser.add_argument('--size', default = 299, help = "the size of the thumbnails", type = int)

# share (float) : the share of positive samples targeted in the data
parser.add_argument('--share', default = .04, help = "share of positive samples", type = float)

# data (str) : the folder in which the source data is located 
parser.add_argument('--data', default = '/data/GabrielKasmi/data/calibration', help = "the folder in which the source data is located", type = str)
# annotations(str) : the name of the annotation file
parser.add_argument('--annotations', default = 'arrays.json', help = "the folder in which the source data is located", type = str)
# out_folder (str) : the folder in which the images will be stored 
parser.add_argument('--out_folder', default = 'dataset', help = "the folder in which the images will be stored", type = str)

# Parse the arguments
args = parser.parse_args()

# Load the data and the annotations file
data_path = args.data
annotations_file = args.annotations
annotations = json.load(open(os.path.join(data_path, annotations_file)))

# tiles list that will be the basis for the train/val/test split
if args.non_empty:
    tiles = [key for key in annotations.keys() if len(annotations[key]) > 0]
else:
    tiles = [key for key in annotations.keys()]

# Final destination directory.
# Takes into account the folder in which the data is stored,
# the name given and whether samples are only created for 
# test images. 
# The structure is the following :
# * indicates the target directory
#
# If test is false : 
# 
# out_folder
#   |
#   - name*
#       |
#       - train
#       - val
#       - test
#
# If test is true
#
# out_folder
#   |
#   - name
#       |
#       - test_case*

"""
Generates thumbnails from a tile and stores it in a target directory
"""    
def generate_thumbnails(source_directory, target_directory, tile, size):
    """
    generates thumbnails from the tile located in the source directory
    and stores them in the target directory.

    tiles have the size 'size'.

    args : 
    - source_directory : the source directory
    - target_directory : the target directory in which the images are stored
    - tile : the tile name from which the thumbnails are computed
    - size : the size of the thumbnails

    returns: None
    """
    # Directory of the image
    # By convention, tile names do not end by .jp2, so we add the 
    # extension

    path_to_tile = os.path.join(os.path.join(source_directory, 'tiles'), tile + '.jp2')

    jp2 = gdal.Open(path_to_tile)

    width, height = jp2.RasterXSize, jp2.RasterYSize

    # number of steps to the left and to the right that will be needed 

    x_shifts, y_shifts = int(width / size) + 1, int(height / size) + 1

    # set up the rightmost and lowermost boundaries. The center cannot be 
    # farther than those points

    x_max, y_max = width - (size / 2), height - (size / 2)

    # initialize the row number
    row = -1

    for i in tqdm.tqdm(range(x_shifts * y_shifts)): # loop over the tile to extract the thumbnails
    #for i in tqdm.tqdm(range(10)): # loop over the tile to extract the thumbnails


        if i % x_shifts == 0:
            row += 1

        xNN = min((size / 2) + size * (i % x_shifts), x_max)
        yNN = min((size / 2) + size * row, y_max)

            
        xOffset=xNN-(size/2) # upper left corner (x coordinate)
        yOffset=yNN-(size/2) # upper left corner (y coordinate)
        

        R0=jp2.GetRasterBand(1)
        G0=jp2.GetRasterBand(2)
        B0=jp2.GetRasterBand(3)

        R = R0.ReadAsArray(xOffset, yOffset, size, size)
        G = G0.ReadAsArray(xOffset, yOffset, size, size)
        B = B0.ReadAsArray(xOffset, yOffset, size, size)

        img_name = tile + '_' + str(xNN) + '-' + str(yNN) + '.png'   # name of the image : coordinates of the center (in px)

        # save the image as a a .png
        io.imsave(os.path.join(target_directory, img_name), np.dstack((R,G,B)))

    return None

"""
Assigns labels to the thumbnails
"""
def assign_labels(annotations, target_folder, args):
    """
    Creates the labels. 
    Only needed for the baseline (then since images are the same
    only need to open the label files from the baseline folder)
    """

    def clean(coordinates):
        """
        small helper parse the coordinates
        """
        coordinates = coordinates.split('-')
        coordinates[1] = coordinates[1][:-4]
        return [float(c) for c in coordinates]


    # The generation of the labels is called within 
    # each train/val/test instatiation.

    # initialize the label dictionnary
    labels = {}

    # get the thumbnails list in the target directory:
    thumbnails = os.listdir(target_folder)

    for thumbnail in thumbnails:

        if  thumbnail[-4:] == '.png':

            labels[thumbnail] = 0 # will be changed only if we find an array

            # extract the tile from which the thumbnail comes from
            # and the center wrt to this tile
            # center is then cleaned to keep only the numbers
            corresponding_tile, center = thumbnail.split('_')[0], clean(thumbnail.split('_')[1])              

            # deduce the coordinates of the tile in pixels
            coordinates = np.array([
                [c - args.size / 2 - 4 for c in center], # ul
                [center[0] - args.size / 2 - 4, center[1] + args.size / 2 - 4], # ur
                [c + args.size / 2 - 4 for c in center], # lr
                [center[0] + args.size / 2 - 4, center[1] - args.size / 2 - 4], # ll
                [c - args.size / 2 - 4 for c in center] # ul (again) to close the polygon
            ])
            # transform the tile as a polygon
            Thumbnail = Polygon(coordinates)

            # retrieve the arrays list
            candidate_arrays = annotations[corresponding_tile]

            for key in candidate_arrays.keys():

                array = np.array(candidate_arrays[key]['PX']) # retrieve the coordinates
                Array = Polygon(np.vstack((array, array[0,:]))) # with the last coordinate repeated

                if Thumbnail.intersects(Array):
                    labels[thumbnail] = 1
                    break # We do classification, so we need at least one array per thumbnail
                
    # now that all images have been assigned a label, we save it in 
    # a dataframe that is exported as a .csv in the target directory folder
    labels_df = pd.DataFrame.from_dict(labels, orient = 'index').reset_index()
    labels_df.to_csv(os.path.join(target_folder, 'labels.csv'), header = None, index = False)

    return None

"""
A function that removes excess negative samples
"""
def balance_samples(target_directory, args):
    """
    balances the samples in the specified target directory.
    """

    # open the labels file
    labels = pd.read_csv(os.path.join(target_directory, 'labels.csv'), header = None)
    labels.columns = ['img', 'label']

    positive_items = labels[labels['label'] == 1].shape[0]

    # compute the total number of items one needs to keep in order 
    # to respect the target share and the associated number of target
    # negatives
    target_total = int(positive_items / args.share)
    target_negative = target_total - positive_items

    # sample the negative items to keep and concatenate the positive items
    to_keep = labels[labels['label'] == 0]['img'].sample(target_negative, random_state = 42).values.tolist()

    for thumbnail in labels[labels['label'] == 1]['img'].values.tolist():
        to_keep.append(thumbnail)    

    # remove the excess images
    for thumbnail in os.listdir(target_directory):
        if not thumbnail in to_keep: 
            if os.path.exists(os.path.join(target_directory, thumbnail)):
                os.remove(os.path.join(target_directory, thumbnail))

    # generate the new label files and save it
    labels = labels[labels['img'].isin(to_keep)]
    labels.to_csv(os.path.join(target_directory, 'labels.csv'), header = None, index = False)

    return None

"""
A function that splits data into train/val/test subsets
"""
def train_val_test_split(target_folder):
    """
    from a pool of labelled thumbanails stored in target_folder/temp,
    and a labels.csv file, performs the train/val/test split and places
    the data into the folders.
    """

    # set up the source directory:
    source_directory = os.path.join(target_folder, 'temp')
    source_labels = pd.read_csv(os.path.join(source_directory, 'labels.csv'), header = None)
    source_labels.columns = ['img', 'label']

    # perform the based on the number of items in the dataframe
    count = source_labels.shape[0]
    train_count, val_count = int(count * .7), int(count * .1)

    indices = list(range(count)) 
    random.shuffle(indices) # shuffle the list of index
    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count + val_count]
    test_indices = indices[train_count + val_count:]

    # store the output in a dictionnary
    cases = {
        'train' : train_indices,
        'validation' : val_indices,
        'test' : test_indices
    }

    for case in cases.keys():
        # create the target directory
        target_directory = os.path.join(target_folder, case)
        if not os.path.isdir(target_directory):
            os.mkdir(target_directory)

        # generate the subdataframe and store it
        labels = source_labels.loc[cases[case], :]
        labels.to_csv(os.path.join(target_directory, 'labels.csv'), header = None, index = False)

        # move the items accordingly from the source to the target folder
        for thumbnail in labels['img'].values:
            shutil.move(os.path.join(source_directory, thumbnail), target_directory)



    return None

"""
main function of the script.
"""
def main():
    """
    Takes the list of tiles and stores the thumbnails generated from the tiles
    in the desired folders. Two cases are possible : 
    - either generate train/val/test samples
    - generate test samples only

    in the first case, train/val and test folders are generated on the fly
    """

    # Set up the folders
    source_directory = args.data
    out_folder = args.out_folder

    # create the out folder if the latter does not exist
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)


    # Create the folers if necessary
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    def f(tile) : # Local function to be put in the threadpoolexecutor
        print('Processing tile {}...'.format(tile))
        return generate_thumbnails(source_directory, target_directory, tile, args.size)

    # 1. Generate the samples

    # We generate samples in the baseline (unaltered) case. 
    # Otherwise if we are in another case but have to generate 
    # train/val samples, but the latter will be duplicated from the baseline
    # finally, in the test case, we only duplicate samples from the baseline/test folder.
    
    # Case where we generate train, val and test samples

    # create a temporary folder where all the thumbnails will be temporarily
    # stored. Once completion finished, they'll be split into
    # train/val/test
    target_directory = os.path.join(out_folder, 'temp')
    if not os.path.isdir(target_directory):
        os.mkdir(target_directory)

    # Generate the thumbnails from the tiles (parallelize the operation)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(f, tiles)

    # once generation is complete, labels are assigned 
    assign_labels(annotations, target_directory, args)

    # balance the samples in the data      
    balance_samples(target_directory, args)

    # now based on the labels.csv file, split the data 
    # into train/val/test sets and create the labels accordingly
    train_val_test_split(out_folder)

    # Finally, remove the temp/ directory
    shutil.rmtree(target_directory)

        
if __name__ == '__main__':
    random.seed(42)
    main()
