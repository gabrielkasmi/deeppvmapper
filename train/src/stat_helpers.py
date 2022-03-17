# -*- coding: utf-8 -*-

# Libraries

from shapely.geometry import Polygon, Point
import numpy as np
from PIL import Image, ImageOps
import os
import tqdm

"""
retrieves the polygons associated to a thumnail
"""
def get_polygons(masks, img_name ,size = 224):
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
                
    return arrays

"""
computes the w/l ratio of a polygon
"""
def compute_w_l_ratio(polygon):
    """computes the w/l ratio and returns the scalar"""
    
    # get minimum bounding box around polygon
    box = polygon.minimum_rotated_rectangle

    # get coordinates of polygon vertices
    x, y = box.exterior.coords.xy

    # get length of bounding box edges
    edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))

    # get length of polygon as the longest edge of the bounding box
    length = max(edge_length)

    # get width of polygon as the shortest edge of the bounding box
    width = min(edge_length)
    
    return np.divide(width, length)

"""
computes the blueness of the polygon
"""
def blueness(img, array):
    """
    given a RGB image and the coordinates of the arrays,
    extracts the areas matching the array and computes
    max(R,G,B) = B. Returns 1 if this is the case, 0 otherwise
    """

    # extract the area
    extracted_area =  np.zeros(np.array(img).shape, dtype = int)

    for iy, ix in np.ndindex(np.array(img).shape[:2]):
        
        Pixel = Point(ix, iy)

        if array.contains(Pixel):
            extracted_area[iy, ix, :] = np.array(img)[iy, ix, :] # color it only if it is in the polygon
                
    # compute the average color channel-wise

    channel_means = np.empty(3)

    for channel in range(extracted_area.shape[2]):
            
        # filter the locations where the intensity is positive
        rows, cols, _ = np.where(extracted_area > 0)
        
        # compute the mean
        channel_means[channel] = np.mean(extracted_area[rows, cols, channel])
        
    # see whether the brightest area is in the blue channel
    return int(np.argmax(channel_means) == 2)

"""
computes the darkness of the polygon
"""
def darkness(gray_img, array):
    """
    given a grayscale image, extract the area of the array
    and returns the average darkness, normalized by 255 to
    return a value comprised between 0 and 1.
    """

    extracted_area = np.zeros(np.array(gray_img).shape)

    # fil the extracted area at the corresponding location
    for iy, ix in np.ndindex(np.array(gray_img).shape):

        Pixel = Point(ix, iy)

        if array.contains(Pixel):
            extracted_area[iy, ix] = np.array(gray_img)[iy, ix]

    rows, cols = np.where(extracted_area > 0)

    darkness = np.mean(extracted_area[rows,cols])

    return darkness / 255


"""
wrapper that computes the dictonnary of characteristics
"""
def compute_simple_factors(labels, images_directory, masks, size = 224):
    """
    computes the factors (w/l ratio, blueness, darkness)
    and returns it as a dictionnary where each key is an array
    and the values are three scalars : 
    - blueness (1 or 0)
    - w/l ratio (comprised between O and 1)
    - darkness (comprised between O and 1)
    
    args :
    - labels (pd.DataFrame) :  with "img" and "labels" columns
    should only contain images with arrays
    images_directory (str) 
    masks (dict) : the dictionnary with the masks
    """
    
    statistics = {}
    index = 0
    
    for i in tqdm.tqdm(range(labels.shape[0])):
        
        img_name = labels['img'].iloc[i] # get the name of the image

        statistics[img_name] = {}
        
        # compute the image and the grayscale image
        img = Image.open(os.path.join(images_directory, img_name)).convert('RGB')
        gray_img = ImageOps.grayscale(img)
        
        # get the polygons associated to the image
        arrays = get_polygons(masks, img_name ,size = size)
        
        for array_id in arrays:

        
            # for each array, compyte the statistics of interest
        
            # w/l ratio 
            w_l_ratio = compute_w_l_ratio(arrays[array_id]['polygon'])
        
            # blueness of the array
            blue = blueness(img, arrays[array_id]['polygon'])
    
            # darkness of the array
            dark = darkness(gray_img, arrays[array_id]['polygon'])
        
            statistics[img_name][index] = (w_l_ratio, blue, dark)

    return statistics
        
        