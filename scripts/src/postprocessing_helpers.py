# -*- coding: utf-8 -*-


from tempfile import tempdir
import numpy as np
from area import area
from shapely.geometry import Point, Polygon
from pyproj import Transformer
import tqdm
import glob
from fiona import collection
import pandas as pd
import os
import json
import geojson
import math
import itertools
import matplotlib.pyplot as plt
import cv2 as cv


"""
helpers for the first postprocessing, which converts raw masks 
into installations characteristics using simple heuristics.
"""


def retrieve_city_code(center, communes):
    """
    returns the "code insee" of the commune
    where the installation is located
    """
    Center = Point(center)
    
    code = None
    
    
    for commune in communes.keys():

        # handle the multipolygons
        raw_coords = communes[commune]["coordinates"]

        if len(raw_coords) > 1:

            for rc in raw_coords:
                if np.array(rc).shape[0] == 1:# filter the communes that have a correct dimension
                    coords = np.array(rc).squeeze(0)
                    Rc = Polygon(coords)

                    if Rc.contains(Center):
                        code = communes[commune]['properties']['code_insee']
                        break

        else:
            raw_coords = np.array(raw_coords).squeeze(0)
            
            if raw_coords.shape[0] == 1:# filter the communes that have a correct dimension
            
                coords = raw_coords.squeeze(0)
                Coords = Polygon(coords)
            
                if Coords.contains(Center):
                    code = communes[commune]['properties']['code_insee']
                    break
            
    return code

def return_surface_id(projected_area, surface_categories):
    """
    returns the surface id of the installation given
    its projected area (in sq meters)
    
    returns an int, O to 3 corresponding
    to the suface cluster
    """

    surface_id = None


    for surface_key in surface_categories.keys():
        lb, ub = surface_categories[surface_key]

        if projected_area <= ub and projected_area > lb:
            surface_id = surface_key
            break
    
    
    return surface_id

def return_latitude_and_longitude_ids(center, latitude_categories, longitude_categories):
    """
    returns the latitude and longitude groups (both in [0, 48])
    given a location. 
    """

    lon, lat = center
        
    for latitude_key in latitude_categories.keys():
        lb, ub = latitude_categories[latitude_key]
        if lat > ub and lat <= lb: # reversed because the latitudes are defined from the top to the bottom 
            lat_id = latitude_key
            break
            
    
    for longitude_key in longitude_categories.keys():
        lb, ub = longitude_categories[longitude_key]
        if lon <= ub and lon > lb:
            lon_id = longitude_key
            break
            
    return lat_id, lon_id


def compute_tilt(geometry, lut, rf, scaler, constant, method):
    """
    computes the tilt angle of an installation. Methods incorporated so far:

    - constant : imputes the average value specified by the user
    - look_up_table : imputes a value based on a look up table
    - random_forest : imputes a value based on a RF regressor


    args : 

    - geometry : the array['geometry'] item of the installation
    - lut : the look up table
    - constant: the constant tilt to be passed as value
    - rf : the random forest
    - scaler : the scaler for the random forest
    """

    # Coefficients correspondance for the look up table
    longitude_categories = {
        0: [-5.587941812553222, -5.282510198057422],
     1: [-5.282510198057422, -4.977078583561623],
     2: [-4.977078583561623, -4.671646969065824],
     3: [-4.671646969065824, -4.366215354570024],
     4: [-4.366215354570024, -4.060783740074225],
     5: [-4.060783740074225, -3.7553521255784257],
     6: [-3.7553521255784257, -3.4499205110826265],
     7: [-3.4499205110826265, -3.144488896586827],
     8: [-3.144488896586827, -2.8390572820910274],
     9: [-2.8390572820910274, -2.5336256675952282],
     10: [-2.5336256675952282, -2.228194053099429],
     11: [-2.228194053099429, -1.9227624386036295],
     12: [-1.9227624386036295, -1.61733082410783],
     13: [-1.61733082410783, -1.3118992096120312],
     14: [-1.3118992096120312, -1.0064675951162316],
     15: [-1.0064675951162316, -0.701035980620432],
     16: [-0.701035980620432, -0.39560436612463246],
     17: [-0.39560436612463246, -0.09017275162883287],
     18: [-0.09017275162883287, 0.21525886286696583],
     19: [0.21525886286696583, 0.5206904773627654],
     20: [0.5206904773627654, 0.826122091858565],
     21: [0.826122091858565, 1.1315537063543637],
     22: [1.1315537063543637, 1.4369853208501633],
     23: [1.4369853208501633, 1.7424169353459629],
     24: [1.7424169353459629, 2.0478485498417625],
     25: [2.0478485498417625, 2.353280164337562],
     26: [2.353280164337562, 2.6587117788333607],
     27: [2.6587117788333607, 2.9641433933291594],
     28: [2.9641433933291594, 3.26957500782496],
     29: [3.26957500782496, 3.5750066223207586],
     30: [3.5750066223207586, 3.880438236816559],
     31: [3.880438236816559, 4.185869851312358],
     32: [4.185869851312358, 4.4913014658081565],
     33: [4.4913014658081565, 4.796733080303957],
     34: [4.796733080303957, 5.102164694799756],
     35: [5.102164694799756, 5.407596309295556],
     36: [5.407596309295556, 5.713027923791355],
     37: [5.713027923791355, 6.0184595382871535],
     38: [6.0184595382871535, 6.323891152782954],
     39: [6.323891152782954, 6.629322767278753],
     40: [6.629322767278753, 6.934754381774551],
     41: [6.934754381774551, 7.240185996270352],
     42: [7.240185996270352, 7.545617610766151],
     43: [7.545617610766151, 7.851049225261949],
     44: [7.851049225261949, 8.15648083975775],
     45: [8.15648083975775, 8.461912454253548],
     46: [8.461912454253548, 8.767344068749349],
     47: [8.767344068749349, 9.072775683245148],
     48: [9.072775683245148, 9.378207297740948]}

    latitude_categories = {
        0: [51.7888767398501, 51.5880990885912],
     1: [51.5880990885912, 51.38732143733229],
     2: [51.38732143733229, 51.186543786073386],
     3: [51.186543786073386, 50.98576613481448],
     4: [50.98576613481448, 50.78498848355557],
     5: [50.78498848355557, 50.58421083229667],
     6: [50.58421083229667, 50.383433181037766],
     7: [50.383433181037766, 50.182655529778856],
     8: [50.182655529778856, 49.98187787851995],
     9: [49.98187787851995, 49.78110022726105],
     10: [49.78110022726105, 49.58032257600214],
     11: [49.58032257600214, 49.379544924743236],
     12: [49.379544924743236, 49.178767273484326],
     13: [49.178767273484326, 48.97798962222542],
     14: [48.97798962222542, 48.77721197096652],
     15: [48.77721197096652, 48.57643431970761],
     16: [48.57643431970761, 48.375656668448705],
     17: [48.375656668448705, 48.1748790171898],
     18: [48.1748790171898, 47.97410136593089],
     19: [47.97410136593089, 47.77332371467199],
     20: [47.77332371467199, 47.572546063413085],
     21: [47.572546063413085, 47.371768412154175],
     22: [47.371768412154175, 47.17099076089527],
     23: [47.17099076089527, 46.97021310963637],
     24: [46.97021310963637, 46.76943545837746],
     25: [46.76943545837746, 46.568657807118555],
     26: [46.568657807118555, 46.36788015585965],
     27: [46.36788015585965, 46.16710250460074],
     28: [46.16710250460074, 45.96632485334184],
     29: [45.96632485334184, 45.765547202082935],
     30: [45.765547202082935, 45.564769550824025],
     31: [45.564769550824025, 45.36399189956512],
     32: [45.36399189956512, 45.16321424830622],
     33: [45.16321424830622, 44.96243659704731],
     34: [44.96243659704731, 44.761658945788405],
     35: [44.761658945788405, 44.560881294529494],
     36: [44.560881294529494, 44.36010364327059],
     37: [44.36010364327059, 44.15932599201169],
     38: [44.15932599201169, 43.958548340752785],
     39: [43.958548340752785, 43.757770689493874],
     40: [43.757770689493874, 43.55699303823497],
     41: [43.55699303823497, 43.35621538697606],
     42: [43.35621538697606, 43.15543773571716],
     43: [43.15543773571716, 42.954660084458254],
     44: [42.954660084458254, 42.753882433199344],
     45: [42.753882433199344, 42.55310478194044],
     46: [42.55310478194044, 42.35232713068154],
     47: [42.35232713068154, 42.15154947942263],
     48: [42.15154947942263, 41.950771828163724]
    }

    surfaces_categories = {
        0: [-0.1e-6, 15.556349186104047],
     1: [15.556349186104047, 18.186533479473212],
     2: [18.186533479473212, 20.84507910184295],
     3: [20.84507910184295, 1e6]}


    coordinates =  np.array(geometry['coordinates']).squeeze(0) # coordinates - corresponds to the array extracted.
    center = np.mean(coordinates, axis = 0) # center of the installation
    projected_area = area(geometry)

    if method == "constant":

        return constant

    if method == 'lut':

        # compute the coefficients to access the look up table
        surface_id = return_surface_id(projected_area, surfaces_categories) 
        lat_id, lon_id = return_latitude_and_longitude_ids(center, latitude_categories, longitude_categories)


        return lut[str(surface_id)][lon_id][lat_id]

    if method == 'random_forest':

        lon, lat = center

        df_values = pd.DataFrame([[lat, lon, projected_area]], columns = ['lat', 'lon', 'proj_surf'])
        X_fit = scaler.transform(df_values)

        return rf.predict(X_fit)[0]

    else:
        print('Unknown method imputed for the tilt estimation.')
        raise ValueError


def compute_azimuth(coordinates, method):
    """ 
    computes the azimuth of the installation. methods implemented so far:
    hough (+bb)

    args
    - coordinates : the np.array of coordinates
    - masks path : the path where the masks are stored
    - method : the type of method
    """

    if method == 'hough':

        # generate a mask mask from the arrays in the temp/masks directory
        cnt = generate_mask(coordinates)
        return BB_timing(cnt)

    else:
        print('Unknown method imputed for the azimuth estimation.')
        raise ValueError


def generate_mask(coordinates):
    """
    generates a mask by computing a local grid
    around the mask. 
    
    used as input for the Hough method
    """

        
    # helper functions
    def gps_to_lambert(coord):
        (long, lat) = coord
        transformer = Transformer.from_crs("epsg:4326", "epsg:2154")
        x,y = transformer.transform(lat, long)
        return (x,y)

    def convert_to_pixel(ulx, uly, coords):

        xres, yres = 0.2, 0.2

        coords[:,0] = (coords[:,0] - ulx) / xres
        coords[:,1] = (coords[:,1] - uly) / yres

        return coords.astype(int)
    
    # conver the mask into lambert coordiantes
    coords_lamb = np.array([gps_to_lambert(c) for c in coordinates])

    # define the bounding box, which itself will define the number of pixels
    offset = 2
    left, top = int(np.min(coords_lamb[:,0])) - offset, int(np.min(coords_lamb[:,1])) - offset

    coordinates = convert_to_pixel(left, top, coords_lamb)    
    
    return coordinates


def BB_timing(cnt):

    """
    computes the bounding box from the contour, obtained from 
    the conversion to pixel coordinates of the image.

    Speeds up the computations compared to the case where the method
    is implemented from segmentation masks.

    Adapted from Y. Tremenbert
    """

    # make it match the shape of a typical open cv contour.
    cnt = np.expand_dims(cnt, axis = 1)

    # Get rotated BndRect
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    # Infer associated lengths and angles
    x0, y0 = box[0][0], box[0][1]
    x2, y2 = box[2][0], box[2][1]
    x3, y3 = box[3][0], box[3][1]
    lengths = [np.sqrt((x3-x0)**2+(y3-y0)**2), np.sqrt((x3-x2)**2+(y3-y2)**2)]
    orientations = [-np.arctan2((x3-x0),(y3-y0))*180/np.pi, -np.arctan2((x3-x2),(y3-y2))*180/np.pi]
    likely_phi = - orientations[np.argmin(lengths)]
    
    return likely_phi


def compute_installed_capacity(coordinates, surface, tilt, efficiency, p, rf, scaler, method):
    """
    computes the installed capacity
    implemented methods : 

    - linear : constant linear fit
    - clustered_linear : clustered linear fit
    - quadratic : quadratic regression
    - random_forest : random forest implementation

    args : 
    coordinates : 
    surface : the *projected surface*
    tilt : the estimated tilt
    efficiency : the efficiency parameter in case of a linear fit
    rf, scaler : the random forest and the scaler
    p : a dictionnary that contains the parameters of the clustered regression
        and quadratic regression
    method : the method to be used
    """

    if method == "linear":
        
        real_surface = surface / np.cos(tilt * np.pi / 180)
        return real_surface * efficiency

    if method == "clustered_linear":

        # extract the dictionnaries that contain the regression coefficients
        surfaces_categories, regression_coefficients = p['surfaces_categories'], p['regression_coefficients']

        # compute the real surface
        real_surface = surface / np.cos(tilt * np.pi / 180)

        surface_id = return_surface_id(surface, surfaces_categories)
        
        if surface_id is not None:
            
            return real_surface * regression_coefficients[surface_id]
        else: # if the surface is not in the boundaries, set a value such that the installation will be discarded. 
            return 0.

    if method == "quadratic":

        # quadratic coefficients is a np.array with the two regression coefficients

        quadratic_coefficients = p['quadratic_coefficients']       

        # compute the real surface
        real_surface = surface / np.cos(tilt * np.pi / 180)

        return np.dot(np.array([real_surface, real_surface**2]), quadratic_coefficients)

    if method == "random_forest":

        # compute the center of the polygon
        center = np.mean(coordinates, axis = 0)
        lon, lat = center

        df_values = pd.DataFrame([[lat, lon, surface, tilt]], columns = ['lat', 'lon', 'proj_surf', 'tilt'])
        X_fit = scaler.transform(df_values)

        return rf.predict(X_fit)[0]

def compute_characteristics(array, installation_id, lut, communes, p):
    """
    array is a geogson object
    its characteristics are extracted and we return a list with :
    
    surface, tilt, code insee, installed capacity, lat, lon
    
    args : 
    - array : an item of the geojson
    - lut : the lookup table 
    - communes : a json with the communes of the current departement
    - temp_dir : the directory of temporary outputs
    - p : a dictionnary of parameters
    """

    constant_tilt = p['constant_tilt'] # extract the constant tilt value
    method_tilt = p['method_tilt'] # extract the chosent method for tilt
    rf, scaler = p['random_forest_tilt'] # extract the random forest and scaler for the tilt

    method_azimuth = p['method_azimuth'] # chosen method for the azimuth

    efficiency = p['efficiency'] # the value of the efficiency
    method_ic = p['method_ic']
    rf_ic, scaler_ic = p['random_forest_ic']

    
    
    # extract the values
    coordinates = np.array(array['geometry']['coordinates']).squeeze(0) # array of coordinates (of the polygon)
    center = np.mean(coordinates, axis = 0) # barycenter of the array
    # extract latitude and longitude from the center
    lon, lat = center


    #### TILT computation
    tilt = compute_tilt(array['geometry'], lut, rf, scaler, constant_tilt, method_tilt)


    ### SURFACE computation
    # based on the tilt and the projected surface, compute
    # the "real" surface
    # compute the projected area
    projected_area = area(array['geometry'])    
    estimated_surface = projected_area / np.cos(tilt * np.pi / 180)

    ### AZIMUTH computation
    azimuth = compute_azimuth(coordinates, method_azimuth)

    ### INSTALLED CAPACITY computation
    # define the dictionnary of parameters

    # hard coded values for the clustered linear regression
    surfaces_categories = {0: [0, 20.886516629316457], 1: [20.886516629316457, 24.9038016107337], 2: [24.9038016107337, 30.06039115705773], 3: [30.06039115705773, 5000]}
    regression_coefficients = {0: 0.1588534217908767, 1: 0.124406296756714, 2: 0.10930903461834848, 3: 0.09946834214084056}
    quadratic_coefficients = np.array([1.21262809e-01, 8.96808844e-06])


    params = {
        'surfaces_categories'   : surfaces_categories,
        'regression_coefficients' : regression_coefficients,
        "quadratic_coefficients" : quadratic_coefficients
    }

    installed_capacity = compute_installed_capacity(coordinates, projected_area, tilt, efficiency, params, rf_ic, scaler_ic, method_ic)
   
    # finally, get the commune code
    # returns None if the installation does not lie in the departement of interest.
    city_code = retrieve_city_code(center, communes)

    # tile name - used in subsequent processing steps
    tile_name = array['properties']['tile']
    
    # return everything    
    return [estimated_surface, tilt, azimuth, installed_capacity, city_code, lat, lon, tile_name, installation_id]


def filter_installations(df, annotations, sorted_buildings, communes):
    '''
    filters the detections that are not on a building

    if more than 1 detection is on the same building, 
    merges the detections as one installation

    returns a dataframe of filtered installations
    '''

    # assign the installations in the tiles to speed up the computations
    # tiles are the keys of the buildings dictionnary 

    clustered_installations = {}

    target_tiles = list(annotations.keys())

    for tile in target_tiles:


        for building_id in sorted_buildings[tile].keys():


            building = sorted_buildings[tile][building_id]['coordinates'] # get the coordinates
            Building = Polygon(building) # convert as polygon

            for installation in annotations[tile].keys():

                coordinates = annotations[tile][installation]['LAMB93']
                Coords = Point(coordinates)

                if Building.contains(Coords):
                    # if the installation is on a building, then
                    # we add its index to the list. 
                    # at the end we sort all installations by building

                    if building_id in clustered_installations.keys():
                        clustered_installations[building_id].append(installation)

                    else:
                        clustered_installations[building_id] = [installation]

    # we now create a new dataframe as follows : 

    # - the former installations that have not been clustered are dropped 
    # - the installations that have been associated to the same building 
    #   are merged together.

    filtered_installations = []


    for building in clustered_installations.keys():

        if len(clustered_installations[building]) == 1:

            # take the index in the source dataframe
            corresponding_index = clustered_installations[building][0]

            # take the values and convert them as a list
            values = df.loc[corresponding_index].values.tolist()

        else: # if there are more than 1 installation, sum the found values

            corresponding_indices = clustered_installations[building]
            # values : 
            aggregated_surface = sum(df.loc[corresponding_indices]['surface'].values)
            aggregated_capacity = sum(df.loc[corresponding_indices]['kWp'].values)

            # localization : consider the barycenter 
            coordinates = np.array([
                df.loc[corresponding_indices]['lat'].values, df.loc[corresponding_indices]['lon'].values 
            ]).transpose()

            coordinates = np.mean(coordinates, axis = 0)

            # city and tilt : should be the same
            city = df.loc[corresponding_indices[0]]['city'] #all values should be the same

            if pd.isna(city): # retrieve manually the coordinates if the coordinate is a nan
                city = retrieve_city_code((coordinates[1], coordinates[0]), communes)

            tilt = df.loc[corresponding_indices[0]]['tilt'] #all values should be the same
            azimuth = df.loc[corresponding_indices[0]]['azimuth']
            
            tile_name = df.loc[corresponding_indices[0]]['tile_name']
            id_installation = df.loc[corresponding_indices[0]]['installation_id']

            # values to be added    
            values = [aggregated_surface, tilt, azimuth, aggregated_capacity, city, coordinates[0], coordinates[1], tile_name, id_installation]

        # add the merged values in the dictionnary
        filtered_installations.append(values)

    # convert the dictionnary to a dataframe
    filtered_df = pd.DataFrame(filtered_installations, columns = ['surface', 'tilt', 'azimuth' ,'kWp', 'city', 'lat', 'lon', 'tile_name', 'installation_id'])

    return filtered_df


"""
handles the conversions between lambert and gps
"""
def convert(coordinates, to_gps = True):
    "converts lambert to wgs coordinates"

    doubled = False

    if to_gps:
        transformer = Transformer.from_crs(2154, 4326, always_xy = True)
    else:
        transformer = Transformer.from_crs(4326, 2154, always_xy = True)
        
    # reshape the coordinates
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    
    if coordinates.shape[0] == 1:         
        coordinates = np.array(coordinates).squeeze(0)
        
    elif len(coordinates.shape) == 3: 
        out_coordinates = np.empty((3,3))
        return out_coordinates

    if len(coordinates.shape) == 1: # case of a single coordinate to convert.
        coordinates = np.vstack([coordinates, coordinates])
        doubled = True
    
    # array that stores the output coordinates
    out_coordinates = np.empty(coordinates.shape)


    # do the conversion and store it in the array
    converted_coords = transformer.itransform(coordinates)

    for i, c in enumerate(converted_coords):

        out_coordinates[i, :] = c

    # new instance
    # out_coords.append(out_coordinates.tolist())

    if doubled:
        return out_coordinates.tolist()[0]
    else:       
        return out_coordinates.tolist()


def intersect_or_contain(p1, p2):
    """
    check whether p1 contains or intersects p2
    """

    return p1.contains(p2) or p1.intersects(p2)


"""
Function that formats the .geojson if necessary
"""
def reshape_dataframe(df):

    """
    takes a pd.DataFrame file as input and returns it in the form of a json file.
    notably, explicits the tiles on which arrays have been found.
    """

    tiles_list = np.unique(df['tile_name'].values).tolist()
   

    print("Annotations have been found on {} tiles.".format(len(tiles_list)))

    annotations = {tile : {} for tile in tiles_list}

    for i in tqdm.tqdm(df.index):

        coordinates = df['lon'][i], df['lat'][i]
        tile = df["tile_name"][i]
        id_installation = df['installation_id'][i]


        annotations[tile][id_installation] = {}

        annotations[tile][id_installation]['WGS'] = list(coordinates)
        annotations[tile][id_installation]['LAMB93'] = convert(np.array(coordinates), to_gps = False)    

    return annotations

"""
assigns the buildings to the tiles based on the coordinates
of the buildings and the coordinates of the tiles
"""
def assign_building_to_tiles(tiles_list, buildings, tiles_dir, temp_dir, dpt):
    """
    returns a dictionnary where each key is a tile
    and the values are the buildings that belong to this tile
    """

    # see whether the file has already been computed. In this case, load it.
    if os.path.exists(os.path.join(temp_dir, 'sorted_buildings_{}.json'.format(dpt))):
        items = json.load(open(os.path.join(temp_dir, 'sorted_buildings_{}.json'.format(dpt))))

        return items
    
    else:

        
        # get the coordinates of the tiles
        tiles_coordinates = compute_tiles_coordinates(tiles_dir)

        covered_tiles = {}

        for tile in tiles_list:
            covered_tiles[tile] = tiles_coordinates[tile]

        # instanciate a new key for the buildings.
        # this key counts the number of times 
        for building in buildings.keys():
            buildings[building]["counts"] = 0        
        
        items = {} #output dictionnary
        
        for tile in tqdm.tqdm(covered_tiles.keys()):

            items[tile] = {}

            coordinates = np.array(covered_tiles[tile]).squeeze(0)
            
            Tile = Polygon(coordinates) # convert the tile as a polygon
            
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


        # save the item in the temporary directory
        if not os.path.exists(os.path.join(temp_dir, 'sorted_buildings_{}.json'.format(dpt))):
            with open(os.path.join(temp_dir, 'sorted_buildings_{}.json'.format(dpt)), 'w') as f:
                json.dump(items, f)

            
        return items


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
    
#    print('coucou')
#    print(tiles_dir)

    # dictionnary
    items = {}

    # loop over the elements of the file
    with collection(dnsSHP[0], 'r') as input: 
        for shapefile_record  in tqdm.tqdm(input):

            name = shapefile_record["properties"]["NOM"][2:-4]
            coords = shapefile_record["geometry"]['coordinates']

            items[name] = coords

    return items


def associate_characteristics_to_pv_polygons(characteristics, arrays, data_dir, dpt):
    """
    puts the information contained in the dataframe back into 
    the geojson for plotting
    """
    
    
    features = []
    
    # get the list of ids
    installation_ids = characteristics['installation_id'].values

    # extract the information in the dataframe and return it in the 
    # geosjon
    
    features = []
    for installation_id in installation_ids :
        
        array = arrays['features'][installation_id]
        
        table = characteristics[characteristics["installation_id"] == installation_id]

        city = table['city'].values.item()
        surface = table['surface'].values.item() 
        tilt = table['tilt'].values.item()
        kWp = table['kWp'].values.item()
        azimuth = table['azimuth'].values.item()
        
        # properties of the installation
        if pd.isna(city):
            city = 'nan'
        else:
            city = int(city)
        
        properties = {
            'city' : city,
            'surface' : float(surface),
            'tilt' : float(tilt),
            'azimuth' : float(azimuth),
            'kWp' : float(kWp)

        }
        
        # coordinates
        coordinates = array['geometry']['coordinates']
        
        #append to the feature list
        features.append(geojson.Feature(geometry=geojson.Polygon(coordinates), properties=properties))
 
    feature_collection = geojson.FeatureCollection(features)    
    
    with open(os.path.join(data_dir, 'arrays_characteristics_{}.geojson'.format(dpt)), 'w') as f:
        geojson.dump(feature_collection, f)
        
    return None
    

def merge_duplicates(characteristics):
    """
    removes the duplicate entries and merge
    the remaining installations
    """
    
    def check_coordinate(coord, rel_tol = 1e-6):
        """checks whether the coordinates in the array coord
        are close (up to rel_to). Returns a dictionnary 
        with the closeness (or not) of all the pairwise combinations
        """

        if len(coord) == 2:
            # if only two items to check, directly compare them
            combinations = {
                (coord[0], coord[1]) : math.isclose(coord[0], coord[1], rel_tol = rel_tol)
            }

        else: # otherwise consider all pairwise combinations        

            combinations = {}
            for x in itertools.combinations(coord, 2):
                combinations[x] = math.isclose(x[0], x[1], rel_tol = rel_tol)
    
        return combinations
    
    # remove duplicates
    characteristics = characteristics.drop_duplicates()
    
    installation_ids = characteristics['installation_id'].values
    
    
    for installation_id in installation_ids : 
        
        table = characteristics[characteristics['installation_id'] == installation_id]
        if table.shape[0] > 1:# if more than one observation is attached to the id
            
            # consider the latitude and longitude
            lats = table['lat'].values
            
            # see whether pairwise combinations are equal or not
            combinations = check_coordinate(lats, rel_tol = 1e-5)

            # loop over the combinations
            # and if two latitudes are close enough, check the longitudes
            # and if the longitudes are close as well, create a new entry
            # append it to the main frame and remove the two former 
            # entries.
            for combination in combinations.keys():

                i,j = table[table['lat'] == combination[0]].index.item(), table[table['lat'] == combination[1]].index.item() 

                if combinations[combination]: # if both latitudes are identical, check the longitudes as well                 
                    
                    coords = table.loc[i,'lon'], table.loc[j, 'lon']

                    long_combination = check_coordinate(coords, rel_tol = 1e-4) 

                    if long_combination[coords]: # both latitude and longitude are the same, so we
                                                 # can merge the installations

                            surface = table['surface'].sum()
                            city = table.loc[i, 'city']
                            tilt = table.loc[i,'tilt']
                            azim = table.loc[i,'azimuth']
                            kWp = table['kWp'].sum()

                            lat, lon = table.loc[i, 'lat'], table.loc[i, 'lon']
                            tile_name = table.loc[i, 'tile_name']

                            # create a new entry
                            merged_installation = [surface, tilt, kWp, azim, city, lat, lon, tile_name, installation_id]

                            new = pd.DataFrame([merged_installation], columns = ['surface', 'tilt', 'azimuth' ,'kWp', 'city', 'lat', 'lon', 'tile_name', 'installation_id'])

                            # append it to the dataframe and remove two former rows
                            characteristics = characteristics.drop(labels=[i,j], axis=0)
                            
                            characteristics = pd.concat([characteristics, new], ignore_index = True)
                  
    return characteristics