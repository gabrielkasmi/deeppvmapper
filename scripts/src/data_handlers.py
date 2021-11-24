# -*- coding: utf-8 -*-

from fiona import collection
import tqdm
import glob
import numpy as np
from shapely.geometry.polygon import Polygon
from shapely import geometry
from pyproj import Transformer



"""
helper that will convert the departement as a str properly.
"""
def format_dpt(dpt):
    """helper to format the departement
    number if the latter is < 10"""
    
    if dpt < 10:
        str_dpt = str(0) + str(dpt)
    else:
        str_dpt = str(dpt)
        
    return str_dpt

"""
Filter the IRIS that belong to the departement under consideration.
"""
def get_iris(source_iris_dir, dpt):
    """
    filters the iris that are attached to the departement under 
    consideration.
    
    returns a dictionnary with the structure
    {iris_id : {coordinates : [...],
               {properties : { iris : value            
                             },
                            {code_iris : value                            
                            }                            
               }
               }
    }
    Warining : looks for the folder that contains the data for metropolitan France
    if the data is different than 2021, need to edit the first line below.
    
    NOTE : coordinates are in the lambert 93 format
    """
    

    # Warining, look for the folder that contains data for metropolitan France
    # edit this line to get the correct access if the data is different from 2021.
    dnsSHP = glob.glob(source_iris_dir + "/**/CONTOURS-IRIS_2-1_SHP_LAMB93_FXX-2021/CONTOURS-IRIS.shp", recursive = True)

    iris_dict = {}
    # loop over the elements of the file
    with collection(dnsSHP[0], 'r') as input: 
        for shapefile_record  in tqdm.tqdm(input):       

            # extract the "code commune"
            insee_com = shapefile_record['properties']['INSEE_COM'] 

            # see whether the code matches the departemnet number
            if str(dpt) == insee_com[:2]:

                # extract the id, the properties and the geometry
                iris_id = shapefile_record["id"]
                geometry = shapefile_record["geometry"]['coordinates']
                iris, code_iris = shapefile_record['properties']['IRIS'], shapefile_record['properties']['CODE_IRIS']

                # add it to the dictionnary            
                iris_dict[iris_id] = {}

                iris_dict[iris_id]["coordinates"] = geometry
                iris_dict[iris_id]["properties"] = {}
                iris_dict[iris_id]["properties"]['iris'] = iris
                iris_dict[iris_id]["properties"]['code_iris'] = code_iris
    return iris_dict


"""
Get the list of tiles over which localization have been spotted
"""
def get_relevant_tiles(data_path, covered_tiles):
            """
            returns a dictionnary of tiles (and their coordinates)
            if the latter are in the covered tiles. 
            """

            # location of the file with the power plants
            dnsSHP = glob.glob(data_path + "/**/dalles.shp", recursive = True)

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

                            # save the coordinates as a list

                            items[name]['coordinates'].append((x0,y0))

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
            
            # set up the projector
            transformer = Transformer.from_crs(2154, 4326)
            
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

                                converted_coords = transformer.itransform(c)

                                for pt in converted_coords :
                                    x, y = pt

                                # save the converted coordinates as a list
                                items[i]['coordinates'].append((x,y))

                            # save the polygon as well
                            #items[i]['polygon'] = Polygon(items[i]['coordinates'])

                            i += 1

            return items

"""
filters the communes of a departement
"""

def get_communes(source_commune_dir, dpt):
    """
    filters the communes that are attached to the departement number.
    
    returns a dictionnary with the structure
    { commune_id : {
                    coordinates : [...]
    },
                    properties : {
                    'nom_commune' : value,
                    'code_insee'  : value
                    }    
    }
    
    Remark : coordinates are in the decimal format (long, lat)
    """
    dnsSHP = glob.glob(source_commune_dir + "/**/communes-20210101.shp", recursive = True)

    commune = {}
    # loop over the elements of the file
    with collection(dnsSHP[0], 'r') as input: 
        for shapefile_record  in tqdm.tqdm(input):

            code_insee = shapefile_record['properties']['insee']

            # if the departement matches the
            if format_dpt(dpt) == code_insee[:2]:

                id_commune = shapefile_record['id']

                nom_commune = shapefile_record['properties']['nom']
                code_insee = shapefile_record['properties']['insee']
                geometry = shapefile_record['geometry']['coordinates']

                # save in the dict

                commune[id_commune] = {}
                commune[id_commune]['coordinates'] = geometry
                commune[id_commune]['properties'] = {}
                commune[id_commune]['properties']['code_insee'] = code_insee
                commune[id_commune]['properties']['nom_commune'] = nom_commune

    return commune


def get_location_info(coordinate, iris_location, communes_locations):
    """
    extracts the information regarding the iris and the commune to which the observation belong.
    
    if iris and communes are a none, returns a none
    """
    # directly return a None if the inputs are None
    if iris_location is None or communes_locations is None:
        return None, None, None, None
    
    else:
        # initial values are none
        iris, code_iris, code_commune, nom_commune = None, None, None, None       
        
        
        # set up the conversions
        #
        # coordinate is inputed in the format decimal (lat, long)
        # commune are in the format decimal (long, lat)
        # iris are in the format lambert93 
        # common format is decimal (long, lat)
        
        # reverse the input coordinate
        coordinate = coordinate[::-1]
        
        # transform as a point
        array_coordinate = geometry.Point(coordinate)                
                
        # set up the projector for the iris
        transformer = Transformer.from_crs(2154, 4326, always_xy = True)
        
        # loop over the communes first
        for commune in communes_locations.keys():
            
            commune_coords = communes_locations[commune]['coordinates']
            
            # some communes are composed of several 
            for sub_coord in commune_coords:
                sub_array = np.squeeze(np.array(sub_coord))
                sub_poly = Polygon(sub_array) 
        

                if sub_poly.contains(array_coordinate):
                    # get the info and exit the loop
                    code_commune = communes_locations[commune]['properties']['code_insee']
                    nom_commune = communes_locations[commune]['properties']['nom_commune']
                    break
                    
        # then over the iris
        for iris_value in iris_location.keys():
            
            iris_coords = iris_location[iris_value]['coordinates']
                        
            # some iris are composed of several coords
            for sub_coord in iris_coords:
                sub_array = np.squeeze(np.array(sub_coord, dtype = object)).reshape(-1,2)
                
                # continue only if the shape contains at least three points
                if not sub_array.shape[0] >= 3:
                    continue
                else:
                    # initialize a target array with the converted coordinates
                    out_array = np.empty((sub_array.shape[0], sub_array.shape[1]))                    
                    
                    # convert the coordinates
                    converted_points = transformer.itransform(sub_array)
                    
                    for i, pt in enumerate(converted_points):
                        out_array[i,:] = pt
                    
                    # convert as a polygon
                    sub_poly = Polygon(out_array) 
        

                    if sub_poly.contains(array_coordinate):
                        # get the info and exit the loop
                        iris = iris_location[iris_value]['properties']['iris']
                        code_iris = iris_location[iris_value]['properties']['code_iris']
                        break
                    
    return iris, code_iris, code_commune, nom_commune
        

