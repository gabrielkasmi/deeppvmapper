# -*- coding: utf-8 -*-

"""
DETECTIONS POSTPROCESSING

This script takes as input the raw dictionnary of results taken from the previous stage 
and returns a geojson file with the aggregated coordinates. 
Power plants and rooftop mounted installations are flagged.
"""
import sys 
sys.path.append('../src')

import json
import os
from geojson import Point, Feature, FeatureCollection
import geojson
import helpers
from pyproj import Proj, transform

import warnings
warnings.filterwarnings("ignore")


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

        # Parameters for this part
        self.pre_processing = configuration.get('postprocessing_initialization')
        self.dpt = dpt

        self.force = force

        # coordinates files from the detection part.
        with open(os.path.join(self.outputs_dir, 'approximate_coordinates.json')) as f:
            self.approximate_coordinates = json.load(f)

    def initialization(self):
        """
        initializes the auxiliary files that come from the BDTOPO
        Corresponds to the building list and the plants list
        """

        # List of buildings
        if not os.path.exists(os.path.join(self.outputs_dir, 'buildings_locations_{}.json'.format(self.dpt))):

            print('Computing the location of the buildings...')
            buildings_locations = helpers.get_buildings_locations(self.bd_topo_path)

            # save the file
            print('Computation complete. Saving the file.')

            with open(os.path.join(self.outputs_dir, 'buildings_locations_{}.json'.format(self.dpt)), 'w') as f:
                json.dump(buildings_locations, f, indent=2)

            print('Done.')

        else:
            # open the existing file 
            print('Opening the buildings_locations.json file... This can take some time...')
            buildings_locations = json.load(open(os.path.join(self.outputs_dir, 'buildings_locations_{}.json'.format(self.dpt))))
            print('File buildings_locations.json loaded.')

        if not os.path.exists(os.path.join(self.outputs_dir, 'plants_locations_{}.json'.format(self.dpt))):

            # List of power plants
            print('Extracting the localization of the power plants...')
            plants_locations = helpers.get_power_plants(self.bd_topo_path)

            # Saving the file

            with open(os.path.join(self.outputs_dir, 'plants_locations_{}.json'.format(self.dpt)), 'w') as f:
                json.dump(plants_locations, f, indent = 2)

            print("Done.")
        else:
            # open the existing file
            plants_locations = json.load(open(os.path.join(self.outputs_dir, 'plants_locations_{}.json'.format(self.dpt))))
            print('File plants_locations.json loaded.')        

        return plants_locations, buildings_locations
    
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
            relevant_tiles = helpers.get_relevant_tiles(self.source_images_dir, list(self.approximate_coordinates.keys()))
            print('List completed.')    

            # Assignations of buildings to tiles
            print("Assigning the buildings to the tiles...")
            building_in_tiles = helpers.assign_building_to_tiles(relevant_tiles, buildings_locations)

            print('Assignation completed and file saved.')

            # Merging everything into a single dictionnary
            print("Mergining buildings and detections in a single dictionnary.")
            merged_dictionnary = helpers.merge_buildings_and_installations_coordinates(building_in_tiles, self.approximate_coordinates)

            # Saving the file
            with open(os.path.join(self.outputs_dir, 'merged_dictionnary_{}.json'.format(self.dpt)), 'w') as f:
                json.dump(merged_dictionnary, f, indent=2)
            print("Done, and file saved.")

        elif self.force:
            # Get the list of tiles for which a detection has been made.

            # List of tiles (of the departement that is being processed)
            print("Getting the list of relevant tiles...")
            relevant_tiles = helpers.get_relevant_tiles(self.source_images_dir, list(self.approximate_coordinates.keys()))
            print('List completed.')    

            # Assignations of buildings to tiles
            print("Assigning the buildings to the tiles...")
            building_in_tiles = helpers.assign_building_to_tiles(relevant_tiles, buildings_locations)

            print('Assignation completed and file saved.')

            # Merging everything into a single dictionnary
            print("Mergining buildings and detections in a single dictionnary.")
            merged_dictionnary = helpers.merge_buildings_and_installations_coordinates(building_in_tiles, self.approximate_coordinates)

            # Saving the file
            with open(os.path.join(self.outputs_dir, 'merged_dictionnary_{}.json'.format(self.dpt)), 'w') as f:
                json.dump(merged_dictionnary, f, indent=2)
            print("Done, and file saved.")

        else:
            # open the existing file
            print('Opening the output dictionnary...')
            merged_dictionnary = json.load(open(os.path.join(self.outputs_dir, 'merged_dictionnary_{}.json'.format(self.dpt))))
            print('Done.')
        
        return merged_dictionnary, plants_locations, buildings_locations

    def filter_detections(self, merged_dictionnary, buildings_locations, plants_locations):
        """
        filters the detections from the merged coordinate dictionnary in order to
        - assign all observations that have been made for a plant to the same plant
        - keep only one detection per building (and take as reference location the average of the locations) 
        """

        # Get the cleaned dictionnaries

        merged_coordinates, plant_coordinates = helpers.merge_location_of_arrays(merged_dictionnary, plants_locations)


        # Export the files
        print('Exporting the files...')
        installations = []

        # Conversion of the coordinates

        # set up the coordinates converter
        inProj = Proj(init="epsg:2154")
        outProj = Proj(init="epsg:4326")

        for tile in merged_coordinates.keys():


            print(merged_coordinates[tile])

            if bool(merged_coordinates[tile]): # continue only if there are merged coordinates on the tile

                out_coords = helpers.return_converted_coordinates(merged_coordinates[tile], inProj, outProj)

                # finally, export the points

                for i, installation in enumerate(list(merged_coordinates[tile].keys())):

                    x, y = out_coords[i,:]

                    coordinate = Point((x,y))
                    properties = {'tile' : tile, 'type' : 'rooftop', 'id' : installation, 'building_type' : buildings_locations[installation]['building_type']}
                    
                    installations.append(Feature(geometry = coordinate, properties = properties))

        # Process the plants
                
        for tile in plant_coordinates.keys():
            
            if bool(plant_coordinates[tile]) == True : # only consider non empty dictionnaries
                for installation in plant_coordinates[tile].keys():
                    
                    coords = plant_coordinates[tile][installation][0]
                    y, x = coords
                    # Convert the coordinates from lambert to decimal
                    y0,x0 = transform(inProj,outProj, x, y)

                    coordinate = Point((x0,y0))
                    properties = {'tile' : tile, 'type' : 'plant', 'id' : installation}
                    
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
        plants_locations, buildings_locations = self.initialization()

        # preparation of the data to get a merged dictionnary
        merged_dictionnary, plants_locations, buildings_locations = self.preparation(plants_locations, buildings_locations)

        # filter the detections and export the geojson file
        self.filter_detections(merged_dictionnary, buildings_locations, plants_locations)




