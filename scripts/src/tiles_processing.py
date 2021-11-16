# -*- coding: utf-8 -*-

# Contains a set of funtions that handle the tiles.

import sys
sys.path.append('../src')

import numpy as np
import pandas as pd
from PIL import Image
import os
import pyproj
from pyproj import Proj
from fiona import collection
import glob
import gdal
from fiona import collection
import cam
import torchvision
from skimage import transform


import warnings
warnings.filterwarnings('ignore')

Image.MAX_IMAGE_PIXELS = None    # No limit when opening an image.


"""
Expresses a point on a thumbnail (e.g. (120,225)) as a coordinate 
"""

def translate_thumbnail_point_to_geo(point, patch_size, thumbnail_center, tile_name, dataset_dir):
    """
    translates the position on a thumbnail in a coordinate in Lambert93
    
    args
    
    - point : a list or tuple, corresponding to the x, y coordinates
    - patch_size : the size of the image
    - thumbnail_center : a list or a tuple, corresponding to the center of the thumbnail
    - tile_name : the name of the tile to which the thumbnail belongs to
    - dataset_dir : the path/to/dataset
    """
    
    # express the coordinates in terms of pixels on the tile
    x, y = point
    x_0, y_0 = thumbnail_center
    
    x_coord = x + (x_0 - (patch_size / 2))
    y_coord = y + (y_0 - (patch_size / 2))
    
    # get the upper left coordinate of the corresponding tile
    dnsSHP = glob.glob(dataset_dir + "/**/dalles.shp", recursive = True)

    with collection(dnsSHP[0], 'r') as input:
        for shapefile in input:
            if tile_name == shapefile["properties"]["NOM"][2:]:
                dns=shapefile["properties"]["NOM"][2:]
                dnsJP2 = glob.glob(dataset_dir + "/**/" +  dns,recursive = True)[0]
                ds=gdal.Open(dnsJP2)
                ulx, xres, _, uly, _, yres  = ds.GetGeoTransform()
                
    x_final = x_coord * xres + ulx
    y_final = y_coord * yres + uly
    

    return x_final, y_final
        

"""
a function that computes the CAM, upscales it and returns the maximal point, corresponding
to the point where the probability of an array is maximized. 

Remark : 1st version of the function, does not accomotate for multimodal cams.
"""

def get_approximate_location(image, model, device = 'cuda'):
    """
    given an input image, returns the approximate location based
    on the class activation map
    
    args:
    image : a PIL image
    model : the model used for inference
    device (optional) the device on which the model and image should be sent.
    
    returns
    class_activation_map : the upscaled (at the size of the image) class activation map
    """
    
    # move the model to the device
    model = model.to(device)
        
    # transforms to convert the image as a tensor.
    transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # compute the class activation map    
    cam_img = cam.class_activation_map_inception(transforms(image), model, device = device)
    
    # upscale the class activation map
    upscaled_cam = transform.resize(cam_img, image.size)
    
    # return the coordinate of the point that has the largest value
    # returns the reversed coordinates as the origin is not the same 
    # on an image and on an array.
    return np.unravel_index(upscaled_cam.argmax(), upscaled_cam.shape)[1], np.unravel_index(upscaled_cam.argmax(), upscaled_cam.shape)[0]


"""
A wrapper that returns a dictionnary for a whole set of images
"""
def compute_location_dictionary(images_list, model, folder, device = 'cuda'):
    """
    Wrapper that computes the approximate location for a set of images
    and returns these locations in a dictionnary.
    
    args
    img_list : a list or array of image names
    model : the model with which the cam is computed
    folder : the folder where the images are located

    returns 
    locations : a dictionnary where each key is the image name 
    and each value the coordinates of the point
    """
    
    locations = {}
    
    for image in images_list:
        img = Image.open(os.path.join(folder, image))
        locations[image] = get_approximate_location(img, model, device = device)
        
    return locations

"""
For a dictionnary of positively labelled images, where the key is the image name 
and the value the estimate of the center of the KDE/CAM, translates this coordinate
(relative to the thumbnail) in coordinate relative to the tile and geo coordinates.

Put otherwise, we no longer approximate but rather consider the "true" estimated value of the 
array on the thumbnail, as guessed by the model.
"""
def extract_estimated_array_coordinates_per_tile(img_list, folder, patch_size = 299):
    """
    Returns a dictionnary where each key is the tile name
    For each tile, we have the approximate coordinate of the array, based on the 
    coordinates of the center of the thumbnail
    
    args
    img_list:  a dictionnary where each key is a positive image and the value is the center of the KDE/CAM
    folder : the folder where the geo information is located
    patch_size : the size of the thumbnail (default : 299)
    
    returns the coordinates expressed in pixels and in geo coordinates (decimal)
    """
    
    def extract_coordinates(name):
        """
        Helper that extract the pixel coordinates from the image name

        Returns the pixel coordinates
        """

        _, pixels_coords = name.split('_')
        pixels_coords = pixels_coords[1:-5].split(',')
        coords = [float(c) for c in pixels_coords]

        return coords
    
    
    # get the list of tile names
    
    tile_names = []
    for name in img_list.keys():
        tile_names.append(name.split('_')[0])

    tile_names = list(set(tile_names))

    if not len(tile_names) <= 1:
        print('Images coming from more than one tile for this tile.')
        raise ValueError

    if len(tile_names) == 0:
        # in this case, there is no detection on the tile, so we pass and return None.
        return None
    
    # main dictionnary
    approximate_coordinates = {}
    
    for tile_name in tile_names:

        for i, name in enumerate(list(img_list.keys())): # names of thumbnails that contain an array
            if tile_name in name: # check whether we are in the correct tile
                
                # extract the coordinates
                pixel_coords = img_list[name] # correspond to the value in the dictionnary
                
                # get the center of the thumbnail
                center = extract_coordinates(name)
                
                # by default, we extract the coordinatesof the center of the thumbnail
                # so the value passed for the location on the image (1st argument of the function)
                # is (0,0)
                
                geo_coords = translate_thumbnail_point_to_geo(pixel_coords, patch_size, center, tile_name + '.jp2', folder)

                print(geo_coords)
                
                # rescale the pixel coordinates to express them relative
                # to the upper left of the tile
                
                x, y = pixel_coords
                x0, y0 = center
    
                x_coord = x + (x0 - (patch_size / 2))
                y_coord = y + (y0 - (patch_size / 2))
                
                approximate_coordinates[str(i + 1)] = [(x_coord, y_coord), geo_coords]
                
                
    return approximate_coordinates


"""
Generates thumbnails for a whole tile.
"""

def generate_thumbnails_only_for_a_tile(folder, target_folder, tile, patch_size):
    """
    For a given tile, generates a set of thumbnails and the corresponding labels
    """
    
    # Generate the thumbnails
    img_centers, images = generate_thumbnails_from_tile(folder, tile, patch_size)
        
    # Set up the target directory
    
    target_directory = os.path.join(target_folder, tile[:-4])
    try:
        os.mkdir(target_directory) # if it does not exist, create the folder and store the outputs in it
        
        # Export the thumbnails
        export_thumbnails(images, tile, img_centers, target_directory)
    
    except : # if the directory already exists
        
        # Export the thumbnails
        export_thumbnails(images, tile, img_centers, target_directory)
        
    return None

"""
Exports a list of thumbnails to a desired location
"""

def export_thumbnails(images, tile, img_centers, target_folder):
        """
        exports the images in the desired folder
        """
        
        for image, name in zip(images, img_centers):
                    
            image_name = tile[:-4] + '_' + str(name) + '.png'
                    
            location = os.path.join(target_folder, image_name)
            
            image.save(location, 'PNG')
            
        return None

"""
Generates a list of thumbnails given a imputed tile name and 
a patch size

At the rightmost and lowermost edges, patches overlap if
the patch size is not a factor of the width/height of the 
tile.
"""
def generate_thumbnails_from_tile(folder, tile_name, patch_size):
        """
        crops the input image into thumbnails of a given size
        returns the list of images
        """
        
        path = [os.path.join(dirpath,filename) for dirpath, _, filenames in os.walk(folder) for filename in filenames if filename == tile_name][0]

        
        # lists that contain the image centers and the images
        img_centers, images = [], []
        
        # open the image
        
        im = Image.open(path)
        print('Image {} opened.'.format(tile_name))
            
        width, height = im.size


        # number of steps to do in the x, y directions
        x_shifts, y_shifts = int(width / patch_size) + 1, int(height / patch_size) + 1

        # set up the rightmost and lowermost boundaries. The center cannot be 
        # farther than those points


        x_max, y_max = width - (patch_size / 2), height - (patch_size / 2)

        # initialize the row number
        row = -1

        for i in range(x_shifts * y_shifts):

            if i % x_shifts == 0:
                row += 1
                if row % 10 == 0:
                    print('Processing row {}/{}...'.format(row + 1, y_shifts))

            # center of the thumbnail
            x = min((patch_size / 2) + patch_size * (i % x_shifts), x_max)
            y = min((patch_size / 2) + patch_size * row, y_max)

            # store the center
            img_centers.append((x,y))

            # boundaries of the image
            left = x - patch_size / 2
            top = y - patch_size / 2

            # for the right and bottom boundaries, consider
            # the potential limit

            right = x + patch_size/2
            bottom = y + patch_size / 2
        
            # extract the image
            img = im.crop((left, top, right, bottom))
            images.append(img)
                
        return img_centers, images
                
