# -*- coding: utf-8 -*-

import folium
from folium.plugins import MarkerCluster
import tqdm
from shapely.geometry.polygon import Polygon
import numpy as np
import os
from shapely import geometry


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
                        
                        
                        if buildings[building]['counts'] == 4:
                            del buildings[building]
                            
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
def merge_location_of_arrays(merged_dictionnary, plants_location):
        """
        for each building in each tile, assigns to each building the list of locations that
        are within the building
        
        then returns the average coordinate over the coordinates of the arrays
        
        the unsorted dictionnary contains all points that have not been assigned to a building. 
        it can be either because it belon to a plant or because the point is outside
        of a building
        
        returns 
        - merged_coordinates, a dictionnary with 
        {tile : locations} 
        - plants_coordinates, a dictionnary with {tile : locations}
        """
        
        merged_coordinates = {}
        # unsorted_coordinates = {} # will be populated by the installations that have not 
        # been assigned to any array
        plants_coordinates = {}
            
        # loop over the tiles
        
        print('Merging points that belong to the same building...')
        
        for tile in tqdm.tqdm(merged_dictionnary.keys()):

            # carry on only if the tile contains arrays.
            if merged_dictionnary[tile]['array_locations'] is not None:

                merged_coordinates[tile] = {} 
                            
                # loop over the buildings
                
                for building_id in merged_dictionnary[tile]['buildings'].keys():
                    
                    coordinates = merged_dictionnary[tile]['buildings'][building_id]["coordinates"]
                    
                    # building_center = list(np.array(coordinates).mean(axis = 0))
                                
                    building_polygon = Polygon(coordinates)          
                
                    for location_id in merged_dictionnary[tile]['array_locations'].keys() :
                    
                        candidate_location = geometry.Point(merged_dictionnary[tile]['array_locations'][location_id])
                    
                        if building_polygon.contains(candidate_location):
                            # Create a new key if the building contains an array
                            # add the id to the list of identified ids
                            try :                    
                                merged_coordinates[tile][building_id].append(merged_dictionnary[tile]['array_locations'][location_id])
                            except:
                                merged_coordinates[tile][building_id] = [merged_dictionnary[tile]['array_locations'][location_id]]    

                            
        # now that the coordinates have been brought together, we merge them
        for tile in merged_coordinates.keys():
            
            for building in merged_coordinates[tile].keys():
                
                # extract the coordinates as an array
                coords = np.array(merged_coordinates[tile][building])
                # compute and return the mean
                mean_coords = list(coords.mean(axis = 0))

                # replace the coordinates in the dictionnary.
                merged_coordinates[tile][building] = mean_coords
                
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
                    plant_poly = Polygon(coordinates)
                    plant_barycenter = list(np.array(coordinates).mean(axis =0))
                    
                    
                    # loop over the detections of the tile
                    # loop over all the locations of the tiles
                    for location_id in merged_dictionnary[tile]['array_locations'].keys() :
                        
                        candidate_location = geometry.Point(merged_dictionnary[tile]['array_locations'][location_id])
                        
                        if plant_poly.contains(candidate_location):
                            # add the localization
                            plants_coordinates[tile][plant] = plant_barycenter
                            break # we only need only localization per plant.                            

                            
        print('Done.')
        return merged_coordinates, plants_coordinates

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
    in_coordinates = np.array([tiles_coordinates[k] for k in tiles_coordinates.keys()]).reshape(-1,2)    
    
    # initialize the output array
    out_coordinates = np.empty((in_coordinates.shape[0], in_coordinates.shape[1]))
    
    # do the conversion
    converted_coords = transformer.itransform(in_coordinates)
    
    # populate the output array
    for i, point in enumerate(converted_coords):

        out_coordinates[i,:] = point 
    
    
    return out_coordinates
    