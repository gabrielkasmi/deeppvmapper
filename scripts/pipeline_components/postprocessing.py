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


