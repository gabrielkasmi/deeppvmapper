# -*- coding: utf-8 -*-

"""
PANEL AGGREGATION
the script takes the arrays geojson file as input
and returns the aggregated capacities

"""
from distutils.command.config import config
from lib2to3.pgen2 import driver
import sys
sys.path.append('../src')


import os
import json
import postprocessing_helpers
import geojson
import pandas as pd
import concurrent
import tqdm


"""
helper that saves the outputs in a file name stored
in the target directory. 
if the file already exists, append current outputs to 
the existing file.
"""
def save(outputs, target_directory, file_name):
    """
    saves the raw outputs of the model 
    the outputs should be a dictionnary.
    """
# if no file exists : 
    if not os.path.isfile(os.path.join(target_directory, file_name)):
        with open(os.path.join(target_directory, file_name), 'w') as f:
            json.dump(outputs, f, indent=2)
    else:
        # update the file
        # open the file
        previous_outputs = json.load(open(os.path.join(target_directory, file_name)))
        
        # add the latest tiles
        for key in outputs.keys():
            previous_outputs[key] = outputs[key]

        # save the new file
        with open(os.path.join(target_directory, file_name), 'w') as f:

            json.dump(previous_outputs, f, indent=2)

    return None

def save_geojson(outputs, target_directory, file_name):
    """
    saves the raw outputs of the model 
    the outputs should be a dictionnary.
    """
# if no file exists : 
    if not os.path.isfile(os.path.join(target_directory, file_name)):
        with open(os.path.join(target_directory, file_name), 'w') as f:
            geojson.dump(outputs, f, indent=2)
    else:
        # update the file
        # open the file
        previous_outputs = geojson.load(open(os.path.join(target_directory, file_name)))
        
        # add the latest tiles
        for key in outputs.keys():
            previous_outputs[key] = outputs[key]

        # save the new file
        with open(os.path.join(target_directory, file_name), 'w') as f:

            geojson.dump(previous_outputs, f, indent=2)

    return None

class Aggregation():
    """
    given a geojson of arrays locations, extracts the panel characteristics and
    outputs a pandas dataframe.
    filters these detections to keep plausible ones and removes the installations 
    above 36 kWc.
    """

    def __init__(self, configuration, dpt):
        """
        Get the parameters and the directories.
        """

        # Retrieve the directories for this part
        self.temp_dir = configuration.get('temp_dir')
        self.aux_dir = configuration.get('aux_dir')
        self.outputs_dir = configuration.get('outputs_dir')
        self.img_dir = configuration.get('source_images_dir')

        # arguments
        self.dpt = dpt

        # parameters
        self.building = configuration.get('filter_building')
        self.lut = configuration.get('filter_LUT')
        self.constant = configuration.get('constant_kWp')


    def initialize(self):
        '''
        initializes the characterization module
        '''

        print('Converting the raw masks into panel characteristics...')
        print(""" Parameters for the conversion have been set as follows : \n
        - Filter by building : {}
        - Impute a tilt using the look up table : {}
        - Use a single coefficient to infer the installed capacity : {}
        """.format(self.building, self.lut, self.constant))

        # load and prepare the lookup table
        lut = json.load(open(os.path.join(self.aux_dir, "look_up_table.json")))

        # load and prepare the communes directory for matching the installations
        # with the communes.
        communes = json.load(open(os.path.join(self.aux_dir, 'communes_{}.json'.format(self.dpt))))

        # open the main file arrays.geojson
        arrays = geojson.load(open(os.path.join(self.outputs_dir, 'arrays_{}.geojson'.format(self.dpt))))

        return arrays, lut, communes

    def characterize(self, arrays, lut, communes):
        """
        characterizes the polygons based on the LUT

        returns a dataframe where each line is an installation with the following columns : 
        surface, tilt, installed_capacity, lat, lon, commune
        """
        installations = []

        def f(array):
            installations.append(postprocessing_helpers.compute_characteristics(array, lut, communes, self.lut, self.constant))
            return None

        print('Extracting the characteristics from the geojson file...')

        #with concurrent.futures.ThreadPoolExecutor() as executor:
        #    executor.map(f, arrays['features'])

        for i, array in tqdm.tqdm(enumerate(arrays['features'])):
            installations.append(postprocessing_helpers.compute_characteristics(array, i, lut, communes, self.lut, self.constant))


        df = pd.DataFrame(installations, columns = ['surface', 'tilt', 'kWp', 'city', 'lat', 'lon', 'tile_name', 'installation_id'])
        
        print('Characteristics extraction completed.')
        
        return df

    def filter_installations(self, df, communes):
        """
        filter the installations based on their characteristics
        """

        print('Filtering the installations...')

        # remove the implausible installed capacities and capacities above 36 kWc
        # a first estimate is the minimum surface of a module (1.7 sq. m.)
        minimum = 1.7
        df = df[(df['kWp'] > minimum) & (df['kWp'] < 36.1)] 

        if not self.building:

            print('Filtering complete.')

            return df
        
        else:

            # convert the dataframe 
            annotations = postprocessing_helpers.reshape_dataframe(df)

            tiles_list = list(annotations.keys()) # subset of tiles on which there is an installation

            # buildings : open the file with all the buildings
            buildings = json.load(open(os.path.join(self.aux_dir, 'buildings_locations_{}.json'.format(self.dpt))))
                    
            # sort the buildings by tile
            sorted_buildings = postprocessing_helpers.assign_building_to_tiles(tiles_list, buildings, self.img_dir, self.temp_dir, self.dpt)

            # return a filtered dataframe where 
            # - the kWp is merged, as well as the surface
            # - installations that are not on a building are removed
            df_out = postprocessing_helpers.filter_installations(df, annotations, sorted_buildings, communes)

            print('Filtering complete.')

  
            return df_out

    def export(self, filtered_installations):
        """
        exports the file as three files : 
        - the .csv of the registry
        - the aggregated characteristics : that merge the characteristics by city
        - an updated geojson file which restricts to the filtered installations and includes 
        the estimated characteristics of the underlying polygon
        """
        # full registry
        filtered_installations = postprocessing_helpers.merge_duplicates(filtered_installations) # remove the duplicates of the dataframe

        filtered_installations.to_csv(os.path.join(self.outputs_dir, 'characteristics_{}.csv'.format(self.dpt)), index = False)

        # aggregated capacities computed for comparison
        # sum the installed capacity
        aggregated_capacity = filtered_installations[['kWp', 'city']].groupby(['city']).sum()

        # count the installations 
        installations_count = filtered_installations[['city', 'kWp']].groupby(['city']).count()
        installations_count.columns = ['count']

        # average the localization, surface and installed capacity
        means = filtered_installations[['surface', 'city', 'lat', 'lon', 'kWp']].groupby(['city']).mean()
        means.columns = ['avg_surface', 'lat', 'lon', 'avg_kWp']

        # aggregate in a single dataframe and save it in the outputs directory.
        aggregated = pd.concat([aggregated_capacity, installations_count, means], axis=1)
        aggregated = aggregated[['count', 'kWp', 'avg_surface', 'avg_kWp', 'lat', 'lon']] # reorder the columns.
        aggregated.to_csv(os.path.join(self.outputs_dir, 'aggregated_characteristics_{}.csv'.format(self.dpt)))

        # updated geojson with the characteristics
        # open the characteristics and arrays files
        characteristics = pd.read_csv(os.path.join(self.outputs_dir, "characteristics_{}.csv".format(self.dpt)))
        arrays = geojson.load(open(os.path.join(self.outputs_dir, 'arrays_{}.geojson'.format(self.dpt))))

        postprocessing_helpers.associate_characteristics_to_pv_polygons(characteristics, arrays, self.outputs_dir, self.dpt)
        
        return None

    def run(self):
        """
        chain all parts together.
        """

        # initialize the module
        arrays, lut, communes = self.initialize()

        # compute the characteristics of the arrays
        characteristics = self.characterize(arrays, lut, communes)

        # filter based on the installed capacities
        filtered_installations = self.filter_installations(characteristics, communes)

        # save the outputs
        self.export(filtered_installations)

