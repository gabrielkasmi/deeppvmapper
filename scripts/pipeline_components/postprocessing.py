# -*- coding: utf-8 -*-

"""
DETECTIONS POSTPROCESSING

This script takes as input the raw dictionnary of results taken from the previous stage 
and returns a geojson file with the aggregated coordinates. 
Power plants and rooftop mounted installations are flagged.
"""
from modulefinder import replacePackageMap
import sys 
sys.path.append('../src')

import json
import os
from geojson import Point, Feature, FeatureCollection
import geojson
import helpers, data_handlers
import tqdm
from pyproj import Transformer
import shutil


class Cleaner():
    """
    Class that cleans the directory once inference is completed. 
    It removes the temporary directories
    """
    
    def __init__(self, configuration):
        """
        initialization of the class
        """
        
        self.temp_dir = configuration.get("temp_dir")
        
    def clean(self):
        '''
        removes the temporary directory
        '''
        print('Deleting the temporary directory {}'.format(self.temp_dir))
        shutil.rmtree(self.temp_dir)
        print('Deletion complete.')

class PostProcessing():
    """
    Postprocesses the observations leveraging the BDTOPO
    """

    def __init__(self, configuration, dpt, force = False):
        """
        load the parameters
        """

        # Retrieve the directories for this part
        self.bd_topo_path = configuration.get("source_topo_dir")
        self.source_images_dir = configuration.get("source_images_dir")
        self.outputs_dir = configuration.get('outputs_dir')
        self.results_dir= configuration.get('geo_dir')
        self.aux_dir = configuration.get('aux_dir')
        self.source_iris_dir = configuration.get('source_iris_dir')
        self.source_commune_dir = configuration.get("source_commune_dir")

        # Parameters for this part
        self.pre_processing = configuration.get('postprocessing_initialization')
        self.compute_iris = configuration.get('compute_iris')
        self.dpt = dpt

        self.force = force

        # coordinates files from the detection part.
        with open(os.path.join(self.outputs_dir, 'approximate_coordinates.json')) as f:
            self.approximate_coordinates = json.load(f)


        # coordinates files from the segmentation part.
        self.approximate_polygons = json.load(open(os.path.join(self.outputs_dir, 'raw_segmentation_results.json')))

    def initialization(self):
        """
        initializes the auxiliary files that come from the BDTOPO
        Corresponds to the building list and the plants list

        If the user inputs it, also computes the dictionnary associated with the IRIS and communes
        (these information being added in the information on the rooftop installation/plant)
        """

        # List of buildings

            # open the existing file 
        print('Opening the buildings_locations_{}.json file... This can take some time...'.format(self.dpt))
        buildings_locations = json.load(open(os.path.join(self.aux_dir, 'buildings_locations_{}.json'.format(self.dpt))))
        print('File buildings_locations_{}.json loaded.'.format(self.dpt))        

        # Plants localization
        if not os.path.exists(os.path.join(self.aux_dir, 'plants_locations_{}.json'.format(self.dpt))):

            # List of power plants
            print('Extracting the localization of the power plants...')
            plants_locations = data_handlers.get_power_plants(self.bd_topo_path)

            # Saving the file

            with open(os.path.join(self.aux_dir, 'plants_locations_{}.json'.format(self.dpt)), 'w') as f:
                json.dump(plants_locations, f, indent = 2)

            print("Done.")
        else:
            # open the existing file
            plants_locations = json.load(open(os.path.join(self.aux_dir, 'plants_locations_{}.json'.format(self.dpt))))
            print('File plants_locations_{}.json loaded.'.format(self.dpt))        

        # IRIS and commune : to be done only if the user has specified it.
        if self.compute_iris:

            # Computation for the IRIS : 
            if not os.path.exists(os.path.join(self.aux_dir, 'iris_{}.json'.format(self.dpt))):

                print('Filtering the IRIS attached to departement {}...'.format(self.dpt))
                iris_location = data_handlers.get_iris(self.source_iris_dir, self.dpt)

                # save the file
                print('Computation complete. Saving the file.')

                with open(os.path.join(self.aux_dir, 'iris_{}.json'.format(self.dpt)), 'w') as f:
                    json.dump(iris_location, f, indent=2)

                print('Done.')

            else:
                # open the existing file 
                print('Opening the iris_{}.json file... This can take some time...'.format(self.dpt))
                iris_location = json.load(open(os.path.join(self.aux_dir, 'iris_{}.json'.format(self.dpt))))
                print('File loaded.')

            # Same for the commune
            if not os.path.exists(os.path.join(self.aux_dir, 'communes_{}.json'.format(self.dpt))):

                print('Filtering the communes attached to departement {}...'.format(self.dpt))
                communes_location = data_handlers.get_communes(self.source_commune_dir, self.dpt)

                # save the file
                print('Computation complete. Saving the file.')

                with open(os.path.join(self.aux_dir, 'communes_{}.json'.format(self.dpt)), 'w') as f:
                    json.dump(communes_location, f, indent=2)

                print('Done.')

            else:
                # open the existing file 
                print('Opening the communes_{}.json file... This can take some time...'.format(self.dpt))
                communes_location = json.load(open(os.path.join(self.aux_dir, 'communes_{}.json'.format(self.dpt))))
                print('File loaded.')

        else: # Iris and commune are None.
            iris_location, communes_location = None, None

        return plants_locations, buildings_locations, iris_location, communes_location
    
    def preparation(self, plants_locations, buildings_locations):
        """
        prepares the data for the postprocessing. This saves a dictionnary
        with the merged_locations of the following shape : 
            {tile : buildings  : [coordinates, ...],
                    detections : [coordinates, ...]
            }
        and a dictionnary with the plants locations.
        These dictionnaries are additional attributes of the class.
        """

        # The preparation steps are only performed if the merged_dictionnary for the departement 
        # does not already exists or if force is enabled

        if not os.path.exists(os.path.join(self.outputs_dir, 'merged_dictionnary_{}.json'.format(self.dpt))) :
            # Get the list of tiles for which a detection has been made.

            # List of tiles (of the departement that is being processed)
            print("Getting the list of relevant tiles...")
            # filter to process only the tiles that contain an array
            non_empty_tiles = []
            for tile in self.approximate_coordinates.keys():
                if len(self.approximate_coordinates[tile]) > 0:
                    non_empty_tiles.append(tile) 
             
            relevant_tiles = data_handlers.get_relevant_tiles(self.source_images_dir, non_empty_tiles)
            print('List completed. {} tiles containing at least one detection have been found.'.format(len(list(relevant_tiles.keys()))))    

            # Assignations of buildings to tiles
            print("Assigning the buildings to the tiles...")
            building_in_tiles = helpers.assign_building_to_tiles(relevant_tiles, buildings_locations)

            print('Assignation completed and file saved.')

            # Merging everything into a single dictionnary
            print("Mergining buildings and detections in a single dictionnary.")
            merged_dictionnary = helpers.merge_buildings_and_installations_coordinates(building_in_tiles, self.approximate_polygons)

            # Saving the file
            with open(os.path.join(self.outputs_dir, 'merged_dictionnary_{}.json'.format(self.dpt)), 'w') as f:
                json.dump(merged_dictionnary, f, indent=2)
            print("Done, and file saved.")

        elif self.force:
            # Get the list of tiles for which a detection has been made.

            # List of tiles (of the departement that is being processed)
            print("Getting the list of relevant tiles...")
            # filter to process only the tiles that contain an array
            non_empty_tiles = []
            for tile in self.approximate_coordinates.keys():
                if len(self.approximate_coordinates[tile]) > 0:
                    non_empty_tiles.append(tile) 
             
            relevant_tiles = data_handlers.get_relevant_tiles(self.source_images_dir, non_empty_tiles)
            print('List completed. {} tiles containing at least one detection have been found.'.format(len(list(relevant_tiles.keys()))))      

            # Assignations of buildings to tiles
            print("Assigning the buildings to the tiles...")
            building_in_tiles = helpers.assign_building_to_tiles(relevant_tiles, buildings_locations)

            print('Assignation completed and file saved.')

            # Merging everything into a single dictionnary
            print("Mergining buildings and detections in a single dictionnary.")
            merged_dictionnary = helpers.merge_buildings_and_installations_coordinates(building_in_tiles, self.approximate_polygons)

            # Saving the file
            with open(os.path.join(self.outputs_dir, 'merged_dictionnary_{}.json'.format(self.dpt)), 'w') as f:
                json.dump(merged_dictionnary, f, indent=2)
            print("Done, and file saved.")

        else:
            # open the existing file
            print('Opening the dictionnary merged_dictionnary_{}.json...'.format(self.dpt))
            merged_dictionnary = json.load(open(os.path.join(self.outputs_dir, 'merged_dictionnary_{}.json'.format(self.dpt))))
            print('Done.')
        
        return merged_dictionnary, plants_locations, buildings_locations

    def filter_detections(self, merged_dictionnary, buildings_locations, plants_locations, iris_location, communes_locations):
        """
        filters the detections from the merged coordinate dictionnary in order to
        - assign all observations that have been made for a plant to the same plant
        - keep only one detection per building (and take as reference location the average of the locations) 
        """

        # Get the cleaned dictionnaries
        rooftop_coordinates, plant_coordinates = helpers.merge_location_of_arrays(merged_dictionnary, plants_locations, self.source_images_dir)

        ## a supprimer
        print(rooftop_coordinates.keys())

        tile = list(rooftop_coordinates.keys())[0]

        print(rooftop_coordinates[tile].keys())

        item = list(rooftop_coordinates[tile].keys())[0]

        print(rooftop_coordinates[tile][item])

        tmp = {}

        for tile in rooftop_coordinates.keys():
            tmp[tile] = {}

            for building in rooftop_coordinates[tile].keys():

                tmp[tile][building] = []

                for array in rooftop_coordinates[tile][building]:

                    tmp[tile][building].append(array.tolist())


        #print(tmp.keys())

        #tile = list(tmp.keys())[0]

        #print(tmp[tile].keys())

        #item = list(tmp[tile].keys())[0]

        #print(tmp[tile][item])

        
        with open(os.path.join(self.outputs_dir, 'tmp_rooftop.json'), 'w') as f:
            json.dump(tmp, f)

        # Export the files
        print('Exporting the files...')
        installations = []

        ## TODO. Modifier pour que ça convertisse des polygônes

        # Conversion of the coordinates
        # set up the coordinates converter
        transformer = Transformer.from_crs(2154, 4326)

        for tile in tqdm.tqdm(list(rooftop_coordinates.keys())):

            if bool(rooftop_coordinates[tile]): # continue only if there are merged coordinates on the tile

                out_coords = helpers.return_converted_coordinates(rooftop_coordinates[tile], transformer)

                # finally, export the points

                for i, installation in enumerate(list(rooftop_coordinates[tile].keys())):

                    x, y = out_coords[i,:]

                    # reverse the points in the geojson
                    coordinate = Point((y,x))
                    # TODO. Modifier ici par un polygone correspondant à une pseudo installation

                    # add the information on the commune and the iris if relevant
                    iris, code_iris, code_commune, nom_commune = data_handlers.get_location_info(out_coords[i,:], iris_location, communes_locations)

                    # add the information in the properties if relevant

                    if self.compute_iris:
                    
                        properties = {'tile'          : tile, 
                                    'type'          : 'rooftop', 
                                    'id'            : installation, 
                                    'building_type' : merged_dictionnary[tile]['buildings'][installation]['building_type'],
                                    'iris'          : iris,
                                    'code_iris'     : code_iris,
                                    'code_commune'  : code_commune,
                                    'nom_commune'   : nom_commune 

                                    }
                    else:
                        properties = {'tile'          : tile, 
                                    'type'          : 'rooftop', 
                                    'id'            : installation, 
                                    'building_type' : buildings_locations[installation]['building_type']
                                    }
                    
                    installations.append(Feature(geometry = coordinate, properties = properties))

        # Process the plants
                
        for tile in plant_coordinates.keys():
            
            if bool(plant_coordinates[tile]) == True : # only consider non empty dictionnaries
                for installation in plant_coordinates[tile].keys():
                    
                    x, y = plant_coordinates[tile][installation]

                    x0, y0 = transformer.transform(x,y)

                    # reverse the points in the geojson
                    coordinate = Point((y0,x0))

                    # add the information on the location of the plant 
                    iris, code_iris, code_commune, nom_commune = data_handlers.get_location_info(plant_coordinates[tile][installation], iris_location, communes_locations)

                    if self.compute_iris:

                        properties = {'tile' : tile, 
                                      'type' : 'plant', 
                                      'id' : installation,
                                    'iris'          : iris,
                                    'code_iris'     : code_iris,
                                    'code_commune'  : code_commune,
                                    'nom_commune'   : nom_commune 
                                      }

                    else: 
                        properties = {'tile' : tile, 
                                      'type' : 'plant', 
                                      'id' : installation}

                    
                    installations.append(Feature(geometry = coordinate, properties = properties))
                    
        installations_features = FeatureCollection(installations) 

        # Export as a json.

        with open(os.path.join(self.results_dir, 'installations_{}.geojson'.format(self.dpt)), 'w') as f:
            geojson.dump(installations_features, f, indent = 2)

        print('Export complete. Installations have been saved in {}.'.format(self.results_dir))

    def run(self):
        """
        wraps everything together in order to export
        the geojson of detected installations
        """

        # location of the plants and the buildings from the BD TOPO
        # and information on the IRIs and commune (are None if the user doesnt specified them)
        plants_locations, buildings_locations, iris_location, communes_locations = self.initialization()

        # preparation of the detection data to get a merged dictionnary
        merged_dictionnary, plants_locations, buildings_locations = self.preparation(plants_locations, buildings_locations)

        # filter the detections and export the geojson file
        # add the information regarding the IRIS and the communes.
        self.filter_detections(merged_dictionnary, buildings_locations, plants_locations, iris_location, communes_locations)




