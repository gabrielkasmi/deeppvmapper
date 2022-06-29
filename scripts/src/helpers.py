# -*- coding: utf-8 -*-

import folium
from folium.plugins import MarkerCluster
from numpy.core.numeric import count_nonzero
from numpy.core.records import array
import tqdm
from shapely.geometry.polygon import Polygon
import numpy as np
import os
from shapely import geometry
import gdal
import glob
import cv2
from fiona import collection




"""
Function used in the detection pipeline.
"""

def save_map(directory, map_center, installations_features, dpt = None):

    """
    loads and save the registry of installations as a .html map

    args:
    directory : where the map should be stored
    map_center: the coordinates of the center of the map to draw
    installations_features: the dictionnary extracted from the geojson that
    contains the coordinates of the installations.

    dpt (optionnal) : the departement number. Used in the name of the output file.

    returns None, saves the files in the maps folder, created from the current 
    directory.

    
    """

    # Generate the map and store it
    m = folium.Map(location=map_center, zoom_start = 9)

    # Add a google maps layout

    basemaps = {
        'Google Satellite Hybrid': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Satellite',
            overlay = True,
            control = True
        ),
    }

    basemaps['Google Satellite Hybrid'].add_to(m)

    # initialize the marker cluster
    marker_cluster = MarkerCluster().add_to(m)


    for installation in installations_features['features']:
        # TODO. Take into account the case where compute_iris = False.

        if installation['properties']["type"] == 'plant': # extract plant and directly plot them

            # add a different layer for the marker
            # and create the popup based on the info 

            id = installation['properties']['id']
            iris, code_iris = installation['properties']['iris'], installation['properties']['code_iris']
            code_commune, commune = installation['properties']['code_commune'], installation['properties']['nom_commune'] 

            popup = """ <b> ID </b> : {}\n
             <b> iris </b>: {}\n
             <b> code iris </b>: {}\n
             <b> code commune </b> : {}\n
             <b> commune </b>: {}\n
            """.format(id, iris, code_iris, code_commune, commune)

            folium.Marker(installation['geometry']['coordinates'][::-1],
            icon = folium.Icon(color = 'red'),
            popup = popup
            ).add_to(m) # Informations to be added here (e.g. city)

        else: # cluster rooftop installations

            # create the popup
            id = installation['properties']['id']
            iris, code_iris = installation['properties']['iris'], installation['properties']['code_iris']
            code_commune, commune = installation['properties']['code_commune'], installation['properties']['nom_commune'] 

            popup = """ <b> ID </b> : {}\n
             <b> iris </b>: {}\n
             <b> code iris </b>: {}\n
             <b> code commune </b> : {}\n
             <b> commune </b>: {}\n
            """.format(id, iris, code_iris, code_commune, commune)

            folium.Marker(installation['geometry']['coordinates'][::-1],
            popup = popup
            ).add_to(marker_cluster)

    folium.LayerControl().add_to(m)  

    if dpt is not None:

        m.save(os.path.join(directory,'map_installations_{}.html'.format(dpt)))

    else:
        m.save(os.path.join(directory,'map_installations.html'))

"""
Assign the buildings to the tiles. Takes the form 
{tile : 
"""
def assign_building_to_tiles(covered_tiles, buildings):
            """
            returns a dictionnary where each key is a tile
            and the values are the buildings that belong to this tile
            """
            
            # instanciate a new key for the buildings.
            # this key counts the number of times 
            for building in buildings.keys():
                buildings[building]["counts"] = 0        
            
            items = {} #output dictionnary
            
            for tile in tqdm.tqdm(covered_tiles.keys()):
                
                items[tile] = {}
                
                Tile = Polygon(covered_tiles[tile]['coordinates']) # convert the tile as a polygon
                
                items[tile] = {} # create an empty dictionnary

                for building in buildings.keys(): # loop over the buildings. These ids are numbers
                    
                    Building = Polygon(buildings[building]['coordinates'])
                    
                    if Tile.intersects(Building):

                        items[tile][building] = buildings[building] # copy the dictionnary

                        # count the number of intersections. There cannot be more than 4 
                        # in this case the building is at the crossing of 4 different tiles
                        # and we remove the building from the dictionnary as it cannot
                        # be found anywhere else.
                        buildings[building]['counts'] += 1
                        
                        
                            
            return items

"""
Merge the localization and the buildings in a single dictionnary
We get an dictionnary with the following structure : 
{tile : 
    buildings : [locations, ...],
    localizations : [coordinates, ...]
}
"""
def merge_buildings_and_installations_coordinates(buildings_in_tiles, locations_coordinates):
        """
        merges the installations (detected) and the buildings 
        that are on that tile
        
        within each tile, we will use the returned dictionnary to 
        see which locations intersects with which building
        """
    
        
        # initialize the dictionnary
        items = {}
        
        # loop over the tiles
        
        for tile in buildings_in_tiles.keys():
            # reference is the building in tiles, i.e. the building that we are proceeding
            
            items[tile] = {}
            
            # add the list of buildings
            items[tile]['buildings'] = buildings_in_tiles[tile]
            # extract the coordinates in Lambert93

            if locations_coordinates is not None: # return a dictionnary of locations or a none if there were no detection
                items[tile]['array_locations'] = locations_coordinates[tile] #{k : v for k,v in locations_coordinates[tile].items()}
            else :
                items[tile]['array_locations'] = None
        
        return items

"""
Within each tile, treats the localization : 

- either marks them as belonging to a plant
- or merge detections that belong to the same building
- or discard detections that are neither a plant nor a building
"""
def merge_location_of_arrays(merged_dictionnary, plants_location, tiles_dir):
        """
        for each building in each tile, assigns to each building the list of locations that
        are within the building
        
        then returns the average coordinate over the coordinates of the arrays
        
        the unsorted dictionnary contains all points that have not been assigned to a building. 
        it can be either because it belon to a plant or because the point is outside
        of a building
        
        returns 
        - array_coordinates, a dictionnary with {tile : locations} 
        - plants_coordinates, a dictionnary with {tile : locations}
        """

        def intersect_or_contain(p1, p2):
            """
            check whether p1 contains or intersects p2
            """

            return p1.contains(p2) or p1.intersects(p2)
        
        array_coordinates = {}
        # unsorted_coordinates = {} # will be populated by the installations that have not 
        # been assigned to any array
        plants_coordinates = {}
            
        # loop over the tiles
        
        print('Grouping polygons that belong to the same building...')

       
        for tile in tqdm.tqdm(merged_dictionnary.keys()):

            # create a new key
            array_coordinates[tile] = {}

            # loop over the buildings that are on that tile
            for building_id in merged_dictionnary[tile]["buildings"].keys():

                # get the coordinates of the building
                building_coordinates = merged_dictionnary[tile]['buildings'][building_id]['coordinates']

                # convert the coordinates as a polygon
                building_polygon = Polygon(building_coordinates)
                #print(type(building_polygon))

                # loop over the detections that have been made over this tile
                for detection_id in merged_dictionnary[tile]['array_locations'].keys():

                    candidate_location = np.array(merged_dictionnary[tile]['array_locations'][detection_id]['LAMB93'])

                    # convert as a point
                    candidate_point = geometry.Polygon(candidate_location)

                    # check if the building contains the point or not
                    if intersect_or_contain(building_polygon, candidate_point):

                        # here, add the pixel location as well to ease the computation of the mask. TODO.

                        candidate_mask = lambert_pixel_conversion(candidate_location, tile, tiles_dir, to_geo = False)
                        

                        # either create a new key (whose ID is the building id)
                        # or append the location to the list of locations that have 
                        # already been assigned to that building
                        if not building_id in array_coordinates[tile]:

                            array_coordinates[tile][building_id] = [candidate_mask]

                        else:
                            array_coordinates[tile][building_id].append(candidate_mask)

        print('Generating pseudo arrays from the annotations that have been merged together...')                   
        # now that the coordinates have been brought together, we merge them
        pseudo_arrays = {}

        for tile in array_coordinates.keys():

            pseudo_arrays[tile] = {}

            for building_id in array_coordinates[tile].keys():

                pseudo_arrays[tile][building_id] = convert_annotations_to_pseudo_mask(array_coordinates[tile][building_id])

                           
        print('Associating coordinates of localizations to power plants...')
        # final part, power plants
        
        for tile in tqdm.tqdm(merged_dictionnary.keys()):

            if merged_dictionnary[tile]['array_locations'] is not None:
            
                plants_coordinates[tile] = {}
                
                # loop over the plants
                for plant in plants_location.keys():
                    
                    coordinates = plants_location[plant]['coordinates']
                    # transform the coordinates as a polygon and 
                    # compute the mean localization of the plant.
                    if len(coordinates) >= 3: # compute the polygon only if it contains more than three coordinates
                        plant_poly = Polygon(coordinates)
                        #print(type(plant_poly))
                        plant_barycenter = list(np.array(coordinates).mean(axis =0))
                        
                        
                        # loop over the detections of the tile
                        # loop over all the locations of the tiles
                        for location_id in merged_dictionnary[tile]['array_locations'].keys() :
                            
                            candidate_location = geometry.Polygon(np.array(merged_dictionnary[tile]['array_locations'][location_id]['LAMB93']))
                            
                            if intersect_or_contain(plant_poly, candidate_location):
                                # add the localization
                                plants_coordinates[tile][plant] = plant_barycenter
                                break # we only need only localization per plant.                            

                            
        print('Done.')
        return pseudo_arrays, plants_coordinates

def return_converted_coordinates(tiles_coordinates, transformer):
    """
    convers the coordinates of a set of coordinates from Lambert93 
    to decimal.
    
    args:
    - tiles_coordinates : a dictionnary {building_id : avg_coord}
    - transformer : a pyproj generator, used to do the conversion
    """
    
    # convert the dictionnary as an array 
    # reshape to always have a two dimensional array
    # TODO. Version modifiée qui prend la première coordonnée du polygone uniquement. 


    in_coordinates = np.array([tiles_coordinates[k][0][0] for k in tiles_coordinates.keys()]).reshape(-1,2)    
    
    # initialize the output array
    out_coordinates = np.empty((in_coordinates.shape[0], in_coordinates.shape[1]))
    
    # do the conversion
    converted_coords = transformer.itransform(in_coordinates)
    
    # populate the output array
    for i, point in enumerate(converted_coords):

        out_coordinates[i,:] = point
    
    
    return out_coordinates


"""
conversion from pixel to lambert
"""
def lambert_pixel_conversion(location, tile, source_images_dir, to_geo = True):
    """
    converts lambert to pixel wrt to the upper left corner 
    of the tile

    to_geo : indicates whether convertion should be made from pixel to lambert.
    if false, it is made from lambert to pixel.

    candidate location is an array, returns a converted location
    """

    # get information on the tile
    ds = gdal.Open(glob.glob(source_images_dir + "/**/{}.jp2".format(tile),recursive = True)[0]) # open the image

    # get the geographical characteristics
    ulx, xres, _, uly, _, yres  = ds.GetGeoTransform() 

    out_coords = np.zeros(np.shape(location))

    if not to_geo:

        # input coordinates are in pixel
        # output coordinates are in lambert

        out_coords[:,0] += (location[:,0] - ulx) / xres
        out_coords[:,1] += (location[:,1] - uly) / yres

        out_coords.astype(np.int64)

    else:
        # input coordinates are in lambert
        # output coordinates are in pixel

        out_coords[:,0] += location[:,0] * xres + ulx
        out_coords[:,1] += location[:,1] * yres + uly


    return out_coords


"""
converts annotations into pseudo masks by merging together adjacent 
polygons. 
due to computational contraints, if there are more than 50 annotations
to merge, it returns the raw annotations.
"""
def convert_annotations_to_pseudo_mask(annotations_list, tile, source_image_dir):
    '''
    merges the arrays from a list of annotation into pseudo masks
    by merging annotations that

    returns a list where each item is an array of coordinates, in the same
    coordinate system as in the input 
    
    due to computational constraints, if there are too many annotations, they are
    not proceeded.
    
    Final postprocessing should take this into account when counting the arrays
    '''

    if len(annotations_list) == 1:
        merged_masks = annotations_list
        #output_mask = np.zeros((10,10))

    elif len(annotations_list) < 50:   
                
        # close the polygons and convert them into pixels
        polygons_list = []
        
        for annotation in annotations_list:
            # close the polygon by repeating the first coordinate
            # and append it to the list
            
            coord = lambert_pixel_conversion(np.array(annotation), tile, source_image_dir, to_geo = False)
            polygons_list.append(np.vstack((coord, coord[0])))
            
            
        # initialize the pseudo mask 
        flatten_coordinates = np.array(
            list(sum([p.tolist() for p in polygons_list], []))
        )
                
        # define the bounding box by taking the 
        # min and max across all coordinates of the list of arrays
        x_min, x_max = np.min(flatten_coordinates[:,0]), np.max(flatten_coordinates[:,0])
        y_min, y_max = np.min(flatten_coordinates[:,1]), np.max(flatten_coordinates[:,1])
        
        # size of the bb
        width, height = int(x_max - x_min), int(y_max - y_min)
                        
        # initialize the pseudo mask
        output_mask = np.zeros((width + 20, height + 20), dtype = np.int64)     
                        
        # express the annotations relative to the upper left corner
        # of the pseudo_mask and see which pixels it recovers
        
        # generate the mask         
        for coords in polygons_list:
            
            new_coord = coords.astype(np.int64)
            new_coord[:,0] = new_coord[:,0] - (x_min - 10)
            new_coord[:,1] = new_coord[:,1] - (y_min - 10)     
            
            # create the polygon  
            Coords = Polygon(np.array(new_coord))
                        
            for ix, iy in np.ndindex(output_mask.shape):
                
                Pixel = Point(ix, iy)
                                
                if Coords.intersects(Pixel):
                    
                    output_mask[ix, iy] = 1
                    
        # extract the contours        
        ulx, xres, _, uly, _, yres = ds.GetGeoTransform()

        # extract the contours of the mask
        output_mask = output_mask.transpose() 
        contours, _ = cv2.findContours(output_mask.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours deteceted and add them
        # to the output file
        
        merged_masks = []
        
        for contour in contours:
            
            pseudo_array = contour.squeeze(1) 
            
            # express in terms of pixels relative to the tile
            pseudo_array[:,0] = pseudo_array[:,0] + (x_min - 10)
            pseudo_array[:,1] = pseudo_array[:,1] + (y_min - 10)
            
            # express in Lamb93
            pseudo_array[:,0] = pseudo_array[:,0] * xres + ulx
            pseudo_array[:,1] = pseudo_array[:,1] * yres + uly
            
            merged_masks.append(pseudo_array)
            
    else:
        print('Too many raw masks to proceed. Returns the unmodified masks.')
        merged_masks = annotations_list
        #output_mask = np.zeros((10,10))

    return merged_masks


"""
computes the coordinates of the tiles
"""
def compute_tiles_coordinates(tiles_dir):
    """
    returns a dictionnary of tiles (and their coordinates)
    if the latter are in the covered tiles. 
    """

    # location of the file with the power plants
    dnsSHP = glob.glob(tiles_dir + "/**/dalles.shp", recursive = True)

    # dictionnary
    items = {}

    # loop over the elements of the file
    with collection(dnsSHP[0], 'r') as input: 
        for shapefile_record  in tqdm.tqdm(input):

            name = shapefile_record["properties"]["NOM"][2:-4]
            coords = shapefile_record["geometry"]['coordinates']
                        
            path = glob.glob(tiles_dir + "/**/{}".format(name), recursive = True)
                    
            items[name] = coords

    return items