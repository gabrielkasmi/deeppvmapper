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

    def __init__(self, configuration):
        """
        load the parameters
        """

        # Retrieve the directories for this part
        self.bd_topo_path = configuration.get("source_topo_dir")
        self.source_images_dir = configuration.get("source_images_dir")
        self.outputs_dir = configuration.get('outputs_dir')

        # Parameters for this part
        self.pre_processing = configuration.get('postprocessing_initialization')
        self.dpt = configuration.get('departement_number')

        # coordinates files from the detection part.
        with open(os.path.join(self.outputs_dir, 'approximate_coordinates.json')) as f:
            self.approximate_coordinates = json.load(f)
    
    def preparation(self):
        """
        prepares the data for the postprocessing. This saves a dictionnary
        with the merged_locations of the following shape : 
            {tile : buildings  : [coordinates, ...],
                    detections : [coordinates, ...]
            }
        and a dictionnary with the plants locations.
        These dictionnaries are additional attributes of the class.
        """

        if self.pre_processing:

            # List of tiles
            print("Getting the list of relevant tiles...")
            relevant_tiles = helpers.get_relevant_tiles(self.source_images_dir, list(self.approximate_coordinates.keys()))
            print('List completed.')

            # List of buildings

            print('Computing the location of the buildings...')
            # Case where we compute the buildings locations
            building_locations = helpers.get_buildings_locations(self.bd_topo_path)

            # save the file
            print('Computation complete. Saving the file.')

            with open(os.path.join(self.outputs_dir, 'building_locations_{}.json'.format(self.dpt)), 'w') as f:
                json.dump(building_locations, f)

            print('Done.')


            # List of power plants
            print('Extracting the localization of the power plants...')
            plants_locations = helpers.get_power_plants(self.bd_topo_path)

            # Saving the file

            with open(os.path.join(self.outputs_dir, 'plants_locations_{}.json'.format(self.dpt)), 'w') as f:
                json.dump(plants_locations, f)

            print("Done.")

            # Assignations of buildings to tiles
            print("Assigning the buildings to the tiles...")
            building_in_tiles = helpers.assign_building_to_tiles(relevant_tiles, building_locations)

            print('Assignation completed and file saved.')

            # Merging everything into a single dictionnary
            print("Mergining buildings and detections in a single dictionnary.")
            merged_dictionnary = helpers.merge_buildings_and_installations_coordinates(building_in_tiles, self.approximate_coordinates)

            # Saving the file
            with open(os.path.join(self.outputs_dir, 'merged_dictionnary_{}.json'.format(self.dpt)), 'w') as f:
                json.dump(merged_dictionnary, f)
            print("Done, and file saved.")

        else: # Directly load the merged_dictionnary and the plants location

            with open(os.path.join(self.outputs_dir, 'merged_dictionnary_{}.json'.format(self.dpt))) as f:
                merged_dictionnary = json.load(f)

            # Plants
            with open(os.path.join(self.outputs_dir, 'plants_locations_{}.json'.format(self.dpt))) as f:
                plants_locations = json.load(f)
        
        return merged_dictionnary, plants_locations

    def run(self):
        """
        wraps everything together in order to export
        the geojson of detected installations
        """
        merged_dictionnary, plants_locations = self.preparation()

        # Get the cleaned dictionnaries

        merged_coordinates, plant_coordinates = helpers.merge_location_of_arrays(merged_dictionnary, plants_locations)

        # Export the files
        print('Exporting the files...')
        installations = []


        # Conversion of the coordinates

        # set up the coordinates converter
        inProj = Proj(init="epsg:2154")
        outProj = Proj(init="epsg:4326")

        # Process the rooftops
        # TODO.. 
        # Créer un array avec toutes les coordonnées à convertir
        # Appliquer la fonction sur l'array directement 
        # faire un zip pour retrouver les propriétés


        for tile in merged_coordinates.keys():

            out_coords = helpers.return_converted_coordinates(merged_coordinates[tile])

            # finally, export the points

            for i, installation in enumerate(list(merged_coordinates[tile].keys())):

                x, y = out_coords[i,:]

                coordinate = Point((x,y))
                # TODO. Add more info on the point
                properties = {'tile' : tile, 'type' : 'rooftop', 'id' : installation}
                
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

        with open(os.path.join(self.outputs_dir, 'installations_{}.geojson'.format(self.dpt)), 'w') as f:
            geojson.dump(installations_features, f)

        print('Export complete. Installations have been saved in {}.'.format(self.outputs_dir))





        

