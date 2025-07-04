# -*- coding: utf-8 -*-

# Contains a set of funtions that handle the tiles.

import sys
sys.path.append('../src')

import numpy as np
import pandas as pd
from PIL import Image
import os
import glob
import gdal
from fiona import collection
import cam
import torchvision
from skimage import transform
import osr
import tqdm
import gdal
import cv2
from shapely.geometry import Polygon

"""
Expresses a point on a thumbnail (e.g. (120,225)) as a coordinate 
"""

def translate_thumbnail_point_to_geo(point, thumbnail):
    """
    translates the position on a thumbnail in a coordinate in Lambert93
    the input is the thumbnail

    """
    
    # express the coordinates in terms of pixels on the tile
    x, y = point # coordinates

    ulx, xres, _, uly, _, yres = thumbnail.GetGeoTransform()
     
    x_final = x * xres + ulx
    y_final = y * yres + uly

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
def extract_estimated_array_coordinates_per_tile(img_list, target_folder):
    """
    Returns a dictionnary where each key is the tile name
    For each tile, we have the approximate coordinate of the array, based on the 
    coordinates of the center of the thumbnail
    
    args
    img_list:  a dictionnary where each key is a positive image and the value is the center of the KDE/CAM
    folder : the folder where the geo information is located
    
    """
    
    # main dictionnary
    approximate_coordinates = {}

    for i, thumbnail in enumerate(list(img_list.keys())):

        pixel_coords = img_list[thumbnail]

        # open the thumbnail
        thumbnail_image = gdal.Open(glob.glob(target_folder + "/**/" + thumbnail, recursive=True)[0])
    
        geo_coords = translate_thumbnail_point_to_geo(pixel_coords, thumbnail_image)
                
        approximate_coordinates[str(i + 1)] = geo_coords
                
                
    return approximate_coordinates


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
        destination_directory = os.path.join(target_folder, tile_name)

        # creates the directory if the latter does not already exists
        if not os.path.isdir(destination_directory):
            os.mkdir(destination_directory)

        if dnsSHP: # if the list is not empty, then look for the shapefile of the tile

            with collection(dnsSHP[0], "r") as input:
                for shapefile_record  in input:
                    if shapefile_record['properties']['NOM'][2:-4] == tile_name:
                        dns=shapefile_record['properties']['NOM'][2:] #open the corresponding tile

                        dnsJP2=glob.glob(folder + "/**/" + dns,recursive = True)[0]

            ds=gdal.Open(dnsJP2) # open the image

            # get the geographical characteristics
            geotransform  = ds.GetGeoTransform() # keep the wrapped variable
            ulx, xres, xskew, uly, yskew, yres  = geotransform 

            width, height = ds.RasterXSize, ds.RasterYSize
        
            # number of steps to the left and to the right that will be needed 

            x_shifts, y_shifts = int(width / patch_size) + 1, int(height / patch_size) + 1


            # set up the rightmost and lowermost boundaries. The center cannot be 
            # farther than those points

            x_max, y_max = width - (patch_size / 2), height - (patch_size / 2)

            # initialize the row number
            row = -1

            for i in tqdm.tqdm(range(x_shifts * y_shifts)): # loop over the tile to extract the thumbnails


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

                img_name = str(ulx + xNN * xres) + '-' + str(uly + yNN * yres) + '.tif'    

                # save the image as a geotiff
                save_geotiff(R,G,B, xNN, yNN, patch_size, geotransform, os.path.join(destination_directory,img_name))

                del R0, G0, B0


        ds = None
"""
Small helpers that returns the thumbnail as a geotiff file.
"""
def save_geotiff(R,G,B, xNN, yNN, patch_size, geotransform, filename):
    """
    saves the thumbnail as a geotiff image.

    - R, G, B : arrays that corresponds to the R,G,B rasters
    of the thumbnail
    - xNN, yNN : the coordinates (pixel) of the center of the image
    - patch_size : the size of the patch
    - geotransform : the geographical characteristics of the tile

    """

    def pixel_to_lambert(point, geotransform, type):
        """
        converts a pixel point into a Lambert93 coordinate
        specifies whether we are dealing with an x or y point
        with the argument 'type'.
        """


        ulx, xres, _, uly, _, yres  = geotransform

        if type == 'x':
            return ulx + point * xres
            
        elif type == 'y':
            return uly + point * yres

    # setting up the latitude and longitude box of the image
    longitude = [pixel_to_lambert(point, geotransform, type = 'x') for point in [xNN - (patch_size / 2), xNN + (patch_size / 2)]]
    latitude = [pixel_to_lambert(point, geotransform, type = 'y') for point in [yNN - (patch_size / 2), yNN + (patch_size / 2)]]
    
    # get the geotransforms
    xmin, ymin, xmax, ymax = [min(longitude), min(latitude), max(longitude), max(latitude)]

    xres = (xmax - xmin) / float(patch_size)
    yres = (ymax - ymin) / float(patch_size)

    # set up the geotransform reversely, or get the coordinates for the four corners
    geotrans = (xmin, xres, 0, ymax, 0, -yres)

    # create the 3-band raster file
    dst_ds = gdal.GetDriverByName('GTiff').Create(filename, patch_size, patch_size, 3, gdal.GDT_Byte)

    dst_ds.SetGeoTransform(geotrans)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(2154)                # Lambert93 coordinate system
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(R)   # write r-band to the raster
    dst_ds.GetRasterBand(2).WriteArray(G)   # write g-band to the raster
    dst_ds.GetRasterBand(3).WriteArray(B)   # write b-band to the raster
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None


"""
converts a segmentation mask into a polygon with coordinates
expressed as lambert93 coordinates.

the returned dictionnary has the shape
{
    thumbnail_id : {polygon_id : [coords],
    }
}
"""
def masks_to_coordinates(outputs, img_names, img_dir):
    """
    converts a mask associated with a .tif image into a polygon
    the polygon is expessed in geographical coordinates
    """

    # output file
    polygons = {}
    _id = 0 # index of the polygons (there can be several for each thumbnail)

    # loop over the images 

    for i, name in enumerate(img_names):

        mask = outputs[i]
        img_name = img_names[i]
        
        polygons[name] = {}

        # open the thumbnail
        thumbnail = gdal.Open(os.path.join(img_dir, '{}'.format(img_name)))
        # extract the geographical properties of the thumbnail
        ulx, xres, _, uly, _, yres = thumbnail.GetGeoTransform()

        # extract the contours of the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours deteceted and add them
        # to the output file
        for contour in contours:
            _id += 1
            polygons[name][_id] = contour.squeeze(1) 
            polygons[name][_id][:,0] = polygons[name][_id][:,0] * xres + ulx
            polygons[name][_id][:,1] = polygons[name][_id][:,1] * yres + uly
            # polygons[_id] = polygons[_id].tolist()       

    return polygons    

def sort_polygons(polygons, source_images_dir):
    """
    associate a polygon to a tile and adds the coordinates 
    in pixel relative to the upper left corner of the tile
    
    all polygons associated to the same tile are gathered under the same
    key.

    args :
    polygons : the dictionnary {thumbnail_name : mask}

    """

    def lamb_to_px(coord, geotransform):
        """
        converts a lambert coordinate to a pixel coordinate
        """
        ulx, xres, _, uly, _, yres  = geotransform
        x, y = coord

        x0 = int((x - ulx) / xres)
        y0 = int((y - uly) / yres)

        return x0, y0

    # retrieve the location of the image
    dnsSHP = glob.glob(source_images_dir + "/**/dalles.shp", recursive = True)

    raw_polygons = {}

    if dnsSHP: # if the list is not empty, then look for the shapefile of the tile

        with collection(dnsSHP[0], "r") as input:

            for shapefile_record  in input:

                tile = shapefile_record['properties']['NOM'][2:-4]             

                # open the tile and get its geographical properties in order to 
                # convert the LAMB93 coordinates into px coordinates wrt the upper
                # left corner of the image
                path_to_image = glob.glob(source_images_dir + "/**/{}.jp2".format(tile),recursive = True)
                
                if path_to_image:
                    raw_polygons[tile] = {}

                    ds = gdal.Open(path_to_image[0]) # open the image

                    # get the geographical characteristics
                    geotransform  = ds.GetGeoTransform() 

                    # Create a polygon from the tile
                    Tile = Polygon(shapefile_record['geometry']['coordinates'][0]) 

                    for image in polygons.keys():
                        for polygon_id in polygons[image].keys():

                            annotation = polygons[image][polygon_id]

                            if annotation.shape[0] > 2:

                                Annotation = Polygon(annotation)

                                if Tile.contains(Annotation):

                                    raw_polygons[tile][polygon_id] = {}

                                    raw_polygons[tile][polygon_id]['LAMB93'] = annotation.tolist()
                                    raw_polygons[tile][polygon_id]['PX'] = np.array([lamb_to_px(an, geotransform) for an in annotation]).tolist()
                else:
                    continue
                        

    # once completed, remove the empty keys
    old_keys = list(raw_polygons.keys())

    for key in old_keys:
        if len(raw_polygons[key].keys()) == 0:
            del raw_polygons[key]
            
    return raw_polygons
