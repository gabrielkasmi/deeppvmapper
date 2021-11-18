# -*- coding: utf-8 -*-

import folium
from folium.plugins import MarkerCluster
import tqdm
from fiona import collection
import tqdm
from shapely.geometry.polygon import Polygon
import glob
import numpy as np
import os
from shapely import geometry
from pyproj import Proj, transform


"""
Function used in the detection pipeline.
"""

def save_map(map_center, installations_features, dpt = None):

    """
    loads and save the registry of installations as a .html map

    args:
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

            folium.Marker(installation['geometry']['coordinates'][::-1]).add_to(m) # Informations to be added here (e.g. city)

        else: # cluster rooftop installations

            folium.Marker(installation['geometry']['coordinates'][::-1]).add_to(marker_cluster)

    folium.LayerControl().add_to(m)  

    if not os.path.isdir('maps'):
        os.mkdir('maps')

    if dpt is not None:

        m.save('maps/map_installations_{}.html'.format(dpt))

    else:

        m.save('maps/map_installations.html')

"""
Gets the list of power plant in the BDTOPO
"""    
def get_power_plants(bd_topo_path):
            '''
            returns a dictionnary of polygons. each item is a plant
            the coordinates are converted in decimal coordinates
            
            remark: aggregates all power plants and not only PV power plants.
            '''
            
            # location of the file with the power plants
            dnsSHP = glob.glob(bd_topo_path + "/**/ZONE_D_ACTIVITE_OU_D_INTERET.shp", recursive = True)
            
            i = 0 # set up the iterator
            
            # set up the coordinates converter
            inProj = Proj(init="epsg:2154")
            outProj = Proj(init="epsg:4326")
            
            # dictionnary
            items = {}
                
            # loop over the elements of the file
            with collection(dnsSHP[0], 'r') as input: 
                for shapefile_record  in tqdm.tqdm(input):

                    if shapefile_record['properties']['NATURE'] == "Centrale électrique": # if a power plant is found
                        coords_list = shapefile_record["geometry"]['coordinates']

                        for coords in coords_list:

                            # create a new instance

                            items[i] = {}

                            items[i]['coordinates'] = []
                            for c in coords: # convert the coordinates
                                x0, y0 = c            
                                y,x = transform(inProj,outProj, x0, y0)

                                # save the converted coordinates as a list
                                items[i]['coordinates'].append((x,y))

                            # save the polygon as well
                            #items[i]['polygon'] = Polygon(items[i]['coordinates'])

                            i += 1

            return items

"""
Gets the location of the buldings in the BDTOPO
"""
def get_buildings_locations(bd_topo_path):
            '''
            returns a dictionnary of polygons. each item is a building
            the coordinates are converted in decimal coordinates
            '''
            
            # location of the file with the power plants
            dnsSHP = glob.glob(bd_topo_path + "/**/BATIMENT.shp", recursive = True)
            
            i = 0 # set up the iterator
            
            # set up the coordinates converter
            # inProj = Proj(init="epsg:2154")
            # outProj = Proj(init="epsg:4326")
            
            # dictionnary
            items = {}
                
            # loop over the elements of the file
            with collection(dnsSHP[0], 'r') as input: 
                for shapefile_record  in tqdm.tqdm(input):

                    coords_list = shapefile_record["geometry"]['coordinates']
                    
                    for coord in coords_list:
                        # create a new instance

                        items[i] = {}

                        items[i]['coordinates'] = []

                        # add the building type
                        items[i]['building_type'] = shapefile_record["properties"]["NATURE"]


                        for c in coord: # convert the coordinates                  
                            
                            x0, y0 = c[0], c[1] # consider only the two first coordinates   
                            # y,x = transform(inProj,outProj, x0, y0)

                            # save the converted coordinates as a list

                            items[i]['coordinates'].append((x0,y0))
                        # save the polygon as well
                        #items[i]['polygon'] = Polygon(items[i]['coordinates'])

                        i += 1

            return items

"""
Assign the buildings to the tiles. Takes the form 
{tile : [bulding_locations],
tile : [bulding_locations],}
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
            
            items = {}
            
            for tile in covered_tiles.keys():
                
                items[tile] = {}
                
                Tile = Polygon(covered_tiles[tile]['coordinates']) # convert the tile as a polygon
                
                items[tile]['coordinates'] = {} # create an empty list
                for i, building in enumerate(list(buildings.keys())): # loop over the buildings
                    
                    
                    Building = Polygon(buildings[building]['coordinates'])
                    
                    if Tile.intersects(Building):
                        
                        items[tile]['coordinates'][i] = buildings[building]['coordinates']
                        
                        # count the number of intersections. There cannot be more than 4 
                        # in this case the building is at the crossing of 4 different tiles
                        # and we remove the building from the dictionnary as it cannot
                        # be found anywhere else.
                        buildings[building]['counts'] += 1
                        
                        
                        if buildings[building]['counts'] == 4:
                            del buildings[building]
                            
            return items

"""
Get the list of tiles over which localization have been spotted
"""
def get_relevant_tiles(data_path, covered_tiles):
            """
            returns a dictionnary of tiles (and their converted coordinates)
            if the latter are in the covered tiles. 
            """

            # location of the file with the power plants
            dnsSHP = glob.glob(data_path + "/**/dalles.shp", recursive = True)

            i = 0 # set up the iterator

            # set up the coordinates converter
            #inProj = Proj(init="epsg:2154")
            #outProj = Proj(init="epsg:4326")

            # dictionnary
            items = {}

            # loop over the elements of the file
            with collection(dnsSHP[0], 'r') as input: 
                for shapefile_record  in tqdm.tqdm(input):

                    name = shapefile_record["properties"]["NOM"][2:-4]
                    if name in covered_tiles: # if the tile is in the list of tiles that have been proceeded

                        coords = shapefile_record["geometry"]['coordinates']

                        if len(coords) > 1: #sanity check, there should be only one item in the coords
                            print('Len greater than 1.')
                            raise ValueError 

                        # create a new instance

                        items[name] = {}

                        items[name]['coordinates'] = []
                        for c in coords[0]: # convert the coordinates
                            x0, y0 = c # consider only the two first coordinates            
                            # y,x = transform(inProj,outProj, x0, y0)

                            # save the converted coordinates as a list

                            items[name]['coordinates'].append((x0,y0))
                        # save the polygon as well
                        #items[i]['polygon'] = Polygon(items[i]['coordinates'])
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
        
        # check that the two keys list match
        if not buildings_in_tiles.keys() == locations_coordinates.keys():
            print('Keys are not matching.')
            raise ValueError
        
        # initialize the dictionnary
        items = {}
        
        # loop over the tiles
        
        for tile in buildings_in_tiles.keys():
            
            items[tile] = {}
            
            # add the list of buildings
            items[tile]['buildings'] = buildings_in_tiles[tile]
            # extract the coordinates in Lambert93

            if locations_coordinates is not None: # return a dictionnary of locations or a none if there were no detection
                items[tile]['array_locations'] = {k : v for k,v in locations_coordinates[tile].items()}
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
        - unsorted_coordinates, a dictionnary with {tile : locations}
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
                
                for building_id in merged_dictionnary[tile]['buildings']['coordinates'].keys():
                    
                    coordinates = merged_dictionnary[tile]['buildings']['coordinates'][building_id]
                    
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

                # convert the coordinates

                merged_coordinates[tile][building] = [mean_coords]
                
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
                    plant_coords = list(np.array(coordinates).mean(axis =0))
                    
                    
                    # loop over the detections of the tile
                    # loop over all the locations of the tiles
                    for location_id in merged_dictionnary[tile]['array_locations'].keys() :
                        
                        candidate_location = geometry.Point(merged_dictionnary[tile]['array_locations'][location_id])
                        
                        if plant_poly.contains(candidate_location):
                            # add the localization

                            # Create a new key if the building contains an array
                            # add the id to the list of identified ids
                            try :                    
                                plants_coordinates[tile][plant].append(plant_coords)
                            except:
                                plants_coordinates[tile][plant] = [plant_coords]
                                
                            # continue, we only need one localization per plant.
                            break
                            

                            
        print('Done.')
        return merged_coordinates, plants_coordinates

def return_converted_coordinates(coordinates, inProj, outProj):
    """
    returns an array of merged coordinates given an inputed
    dictionnary of coordinates. Each key of the dictionnary is 
    the id of a coordinate to compute.

    args :
    coordinates ; a dictionnary with the input coordinates as Lambert, 
    inProj, outProj : the projectors
    """

    # Create the array of coordinates
    in_coords = np.array((len(list(coordinates.keys())), 2))

    for i, installation in enumerate(list(coordinates.keys())):
        # fill the array with the coordinates
        # TODO. See whether we need to remove the [0] or not, if there is additional info or not to be included 
        # for each installation
        x0, y0 = coordinates[installation]

        in_coords[i, 0], in_coords[i, 1] = x0, y0

    # Now that the array is computed, compute the out_array
    # which is the transform of the coordinates.

    out_coords = transform(inProj,outProj, in_coords[:,0], in_coords[:,1]) 

    return out_coords