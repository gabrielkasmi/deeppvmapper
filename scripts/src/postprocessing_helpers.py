# -*- coding: utf-8 -*-


import numpy as np
from area import area
from shapely.geometry import Point, Polygon


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
        
        coords = np.array(communes[commune]['coordinates']).squeeze(0)
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

def compute_characteristics(array, lut, communes):
    """
    array is a geogson object
    its characteristics are extracted and we return a list with :
    
    surface, tilt, code insee, installed capacity, lat, lon
    
    args : 
    - array : an item of the geojson
    - lut : the lookup table 
    - communes : a json with the communes of the current departement
    """
    
    # helper dictionnaries
    # hard coded
    # auxiliary dictionnaries to access the values of the look-up table
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
        0: [7.347880794884119e-16, 15.556349186104047],
     1: [15.556349186104047, 18.186533479473212],
     2: [18.186533479473212, 20.84507910184295],
     3: [20.84507910184295, 3987.75426296126]}

    regression_coefficients = {
        0: 0.14746518407829015,
     1: 0.1422389858867244,
     2: 0.13350621534945112,
     3: 0.13179292212903068}

    
    # extract the values
    coordinates = np.array(array['geometry']['coordinates']).squeeze(0) # array of coordinates (of the polygon)
    center = np.mean(coordinates, axis = 0) # barycenter of the array
    
    # extract latitude and longitude from the center
    lon, lat = center
    
    # compute the projected area
    projected_area = area(array['geometry'])
    
    # based on the projected area, get the surface id to access the LUT
    surface_id = return_surface_id(projected_area, surfaces_categories)
    
    # get the latitude and longitude codes
    lat_id, lon_id = return_latitude_and_longitude_ids(center, latitude_categories, longitude_categories)
    
    # access the lut to get a tilt estimation
    tilt = lut[str(surface_id)][lon_id][lat_id]
    
    # based on the tilt and the projected surface, compute
    # the "real" surface
    estimated_surface = projected_area / np.cos(tilt * np.pi / 180)

    # compute the installed capacity based on the 
    # surface cluster and the estimated surface
    installed_capacity = estimated_surface * regression_coefficients[surface_id]
    
    # finally, get the commune code
    # returns None if the installation does not lie in the departement of interest.
    city_code = retrieve_city_code(center, communes)
    
    # return everything
    
    return [estimated_surface, tilt, installed_capacity, city_code, lat, lon]
