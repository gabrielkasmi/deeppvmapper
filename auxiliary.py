#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('scripts/pipeline_components/')
sys.path.append('scripts/src/')


import data_handlers
import helpers
import yaml
import sys
import os
import json
import argparse


"""
Script that generates the auxiliary outputs needed throughout the pipeline. 
This script needs to be run first. It initializes the aux/ directory 
that contains these outputs. 

If the user does not run this script first, a warning is sent in the main script

These auxiliary outputs are :

Extracted from the BDTOPO
- buildings : geographic coordinates of the buidlings in a given departement.
- plants : geographic coordinates of the plants in a given departement.


Extracted from their own folder : 
- IRIS : statistical clusters of 2,000 inhabitants each
- Communes : lowest aggregation level in the validation registries
"""


# Arguments
parser = argparse.ArgumentParser(description = 'Auxiliary files for the large scale detection pipeline')

parser.add_argument('--dpt', default = None, help = "Department to proceed", type=int)

args = parser.parse_args()


if args.dpt is not None:
    dpt = args.dpt
else:
    print('Please input a departement number to rune the pipeline.')
    raise ValueError

# Load the configuration file
config = 'config.yml'

with open(config, 'rb') as f:
    configuration = yaml.load(f, Loader=yaml.FullLoader)

# Get the folders from the configuration file
aux_dir = configuration.get('aux_dir')

# source directories for the auxiliary outputs
source_iris_dir = configuration.get('source_iris_dir') 
source_commune_dir = configuration.get('source_commune_dir') 
source_topo_dir = configuration.get('source_topo_dir') 

# main script 

# intialize the aux/ directory
if not os.path.isdir(aux_dir):
    os.mkdir(aux_dir)


def main():

    # buildings
    if not os.path.exists(os.path.join(aux_dir, 'buildings_locations_{}.json'.format(args.dpt))):

        print('Computing the location of the buildings...')
        buildings_locations = data_handlers.get_buildings_locations(source_topo_dir)

        # save the file
        print('Computation complete. Saving the file.')

        with open(os.path.join(aux_dir, 'buildings_locations_{}.json'.format(args.dpt)), 'w') as f:
            json.dump(buildings_locations, f, indent=2)

        print('Done.')

    # power plants
    if not os.path.exists(os.path.join(aux_dir, 'plants_locations_{}.json'.format(args.dpt))):

        # List of power plants
        print('Extracting the localization of the power plants...')
        plants_locations = data_handlers.get_power_plants(source_topo_dir)

        # Saving the file

        with open(os.path.join(aux_dir, 'plants_locations_{}.json'.format(args.dpt)), 'w') as f:
            json.dump(plants_locations, f, indent = 2)

    # Computation for the IRIS : 
    if not os.path.exists(os.path.join(aux_dir, 'iris_{}.json'.format(args.dpt))):

        print('Filtering the IRIS attached to departement {}...'.format(args.dpt))
        iris_location = data_handlers.get_iris(source_iris_dir, args.dpt)

        # save the file
        print('Computation complete. Saving the file.')

        with open(os.path.join(aux_dir, 'iris_{}.json'.format(args.dpt)), 'w') as f:
            json.dump(iris_location, f, indent=2)

        print('Done.')

    # communes
    if not os.path.exists(os.path.join(aux_dir, 'communes_{}.json'.format(args.dpt))):

        print('Filtering the communes attached to departement {}...'.format(args.dpt))
        communes_location = data_handlers.get_communes(source_commune_dir, args.dpt)

        # save the file
        print('Computation complete. Saving the file.')

        with open(os.path.join(aux_dir, 'communes_{}.json'.format(args.dpt)), 'w') as f:
            json.dump(communes_location, f, indent=2)

        print('Done.')

if __name__ == '__main__':

    # run the initialization of the auxiliary files.
    main()