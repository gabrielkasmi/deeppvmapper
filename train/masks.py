#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Create the segmentation masks for the training dataset.
Samples must have been created first.
"""

# Libraries
import argparse
import os
import tqdm
import json
import random
import gdal
import concurrent
from skimage import io
import numpy as np
from shapely.geometry import Polygon, Point
import warnings

warnings.filterwarnings("ignore")


# Arguments of the script 
parser = argparse.ArgumentParser(description = 'Segmentation masks generation creation')

# size (int) the size of the thumbnails to be generated
parser.add_argument('--size', default = 299, help = "the size of the thumbnails", type = int)


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

# Initialize the paths to directories for the image samples
# and the masks (that will be created if necessary)
img_subdirs = {
    'train' : os.path.join(args.out_folder, 'train'),
    'validation' : os.path.join(args.out_folder, 'validation'),
    'test' : os.path.join(args.out_folder, 'test'),
}

masks_subdirs = {
    'train' : os.path.join(args.out_folder, 'train_masks'),
    'validation' : os.path.join(args.out_folder, 'validation_masks'),
    'test' : os.path.join(args.out_folder, 'test_masks'),
}

# create the directory if they do not exist yet.
for case in masks_subdirs.keys():
    if not os.path.isdir(masks_subdirs[case]):
        os.mkdir(masks_subdirs[case])


"""
A function that creates a mask given an image
"""
def create_masks(masks, img_name ,size = args.size):
    """
    gets the polygons of arrays attached to the targeted tile
    from the masks dictionnary
    
    returns a dictionnary where each key is the array i and each value
    an array with the coordinates (wrt ul corner of the image) of the mask
    """
    
    def tile_and_center_from_thumbnail(thumnail_name):
        """returns the tile name (str) and the center 
        of the thumbnail (tuple(str, str))
        """
        tile, center = thumnail_name.split('_')[0], thumnail_name.split('_')[1]
        center = float(center.split('-')[0]), float(center.split('-')[1][:-4])

        return tile, center

    tile, center = tile_and_center_from_thumbnail(img_name)

    # deduce the coordinates of the tile in pixels
    coordinates = np.array([
        [c - size / 2 - 4 for c in center], # ul
        [center[0] - size / 2 - 4, center[1] + size / 2 - 4], # ur
        [c + size / 2 - 4 for c in center], # lr
        [center[0] + size / 2 - 4, center[1] - size / 2 - 4], # ll
        [c - size / 2 - 4 for c in center] # ul (again) to close the polygon
    ])

    Thumbnail = Polygon(coordinates)
    
    arrays = {}
    _id = 0

    for mask_id in masks[tile].keys():
        
        array = np.array(masks[tile][mask_id]['PX'])
        Array = Polygon(np.vstack((array, array[0,:]))) # with the last coordinate repeated

        if Thumbnail.intersects(Array):

            array = np.vstack((array, array[0,:]))

            centered_polygon = np.zeros(array.shape)
            centered_polygon[:,0] = array[:,0] - (center[0] - size / 2)
            centered_polygon[:,1] = array[:,1] - (center[1] - size / 2)

            arrays[_id] = {} 
            arrays[_id]['polygon'] = Polygon(centered_polygon)
            arrays[_id]['array'] = centered_polygon
            
            _id += 1
            
    # now that the polygons have been retrieved, create the mask
    # generate a dark image
    mask = np.zeros((size, size), dtype=np.uint8)
    
    for array_id in arrays:
        Array = arrays[array_id]['polygon']

        for ix, iy in np.ndindex((size, size)):
            
            Pixel = Point(iy, ix)
            
            if Array.contains(Pixel):
                
                mask[ix, iy] = 255     
    return mask

"""
A function that generates and save the mask 
"""
def generate_and_save_mask(mask, image, directory, size = args.size):

    #generate the mask
    mask = create_masks(annotations, image, size = size)

    # save the mask
    io.imsave(os.path.join(directory, image), mask)

    return None

"""
main function of the script.
"""
def main():
    """
    Generate the masks and saves them in the dedicated directory
    """

    # loop over the images in each directory
    for case in ['validation']:#img_subdirs.keys():

        print('Generating segmentation masks for the {} images...'.format(case))

        # get the thumbnails (end by .png)
        images = [item for item in os.listdir(img_subdirs[case]) if item[-4:] == '.png']

        print(len(images))

        def f(image):
            return generate_and_save_mask(annotations, image, masks_subdirs[case], size = args.size)


            # Generate the thumbnails from the tiles (parallelize the operation)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(f, images)
        
if __name__ == '__main__':
    random.seed(42)
    main()
