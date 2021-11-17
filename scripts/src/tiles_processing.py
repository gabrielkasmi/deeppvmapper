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
import imageio

import warnings
warnings.filterwarnings('ignore')

# Image.MAX_IMAGE_PIXELS = None    # No limit when opening an image.


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
def generate_thumbnails_from_tile(folder, target_folder, tile_name, patch_size):
        """
        Crops the input impage into thumbnails of a given inputed size.
        Returns the thumbnail names (cooresponding to the coordinates of the 
        center of the thumbnail)
        """

        # retrieve the location of the image
        dnsSHP = glob.glob(folder + "/**/dalles.shp", recursive = True)

        # create the destination directory 
        destination_directory = os.path.join(target_folder, tile_name[:-4])

        # creates the directory if the latter does not already exists
        if not os.path.isdir(destination_directory):
            os.mkdir(destination_directory)

        if dnsSHP: # if the list is not empty, then look for the shapefile of the tile

            with collection(dnsSHP[0], "r") as input:
                for shapefile_record  in input:
                    if shapefile_record['properties']['NOM'][2:] == tile_name:
                        dns=shapefile_record['properties']['NOM'][2:] #open the corresponding tile

                        dnsJP2=glob.glob(folder + "/**/" + dns,recursive = True)[0]

            ds=gdal.Open(dnsJP2) # open the image

            # get the geographical characteristics
            ulx, xres, xskew, uly, yskew, yres  = ds.GetGeoTransform()


            width, height = ds.RasterXSize, ds.RasterYSize
        
            # number of steps to the left and to the right that will be needed 

            x_shifts, y_shifts = int(width / patch_size) + 1, int(height / patch_size) + 1


            # set up the rightmost and lowermost boundaries. The center cannot be 
            # farther than those points

            x_max, y_max = width - (patch_size / 2), height - (patch_size / 2)

            # initialize the row number
            row = -1

            for i in range(x_shifts * y_shifts): # loop over the tile to extract the thumbnails


                if i % x_shifts == 0:
                    row += 1

                xNN = min((patch_size / 2) + patch_size * (i % x_shifts), x_max)
                yNN = min((patch_size / 2) + patch_size * row, y_max)
    
                    
                xOffset=xNN-(patch_size/2) # upper left corner (x coordinate)
                yOffset=yNN-(patch_size/2) # upper left corner (y coordinate)
                

                R0=ds.GetRasterBand(1)
                G0=ds.GetRasterBand(2)
                B0=ds.GetRasterBand(3)

                R = R0.ReadAsArray(xOffset, yOffset, patch_size, patch_size)
                G = G0.ReadAsArray(xOffset, yOffset, patch_size, patch_size)
                B = B0.ReadAsArray(xOffset, yOffset, patch_size, patch_size)

                rgb=np.dstack((R,G,B)) # convert as an array

                img_name = str(ulx + xNN * xres) + '-' + str(uly + yNN * xres) + '.png'    



                # save the image in the destination folder
                imageio.imwrite(os.path.join(destination_directory, img_name), rgb)

                