#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('scripts/pipeline_components/')
sys.path.append('scripts/src/')

import argparse
import json
import pandas as pd
import numpy as np
import os
import yaml
import carbon
from datetime import datetime
import tqdm
from shapely.geometry import Point, Polygon

"""
Computes the accuracy by comparing the outputs of the detection model
with the RNI
"""


# Arguments
parser = argparse.ArgumentParser(description = 'Computation of the accuracy')

parser.add_argument('--dpt', default = None, help = "Department to proceed", type=int)
parser.add_argument('--filename', default = None, help = "name of the RNI to consider", type = str)

parser.add_argument('--source_dir', default = '../data/rni', help = 'location of the ground truth registry', type = str)
parser.add_argument('--evaluation_dir', default = 'evaluation',help = 'where the results are stored', type = str)
parser.add_argument('--outputs_dir', default = 'data',help = 'where the detection outputs are stored', type = str)
parser.add_argument('--aux_dir', default = 'aux',help = 'where the communes are located', type = str)
parser.add_argument('--recompute', default = False,help = 'state whether we reassign the cities', type = bool)



args = parser.parse_args()

# Load the configuration file
config = 'config.yml'

with open(config, 'rb') as f:
    configuration = yaml.load(f, Loader=yaml.FullLoader)

# Get the folders from the configuration file
outputs_dir = configuration.get('outputs_dir')
carbon_dir = configuration.get('carbon_dir')
aux_dir=configuration.get('aux_dir')

if not os.path.isdir(carbon_dir): # intialize the carbon directory
    os.mkdir(carbon_dir)


if args.dpt is not None:
    dpt = args.dpt
else:
    print('Please input a departement number to run the script.')
    raise ValueError

if args.filename is not None:
    filename = args.filename
else:
    print('Please input a file name to run the script.')
    raise ValueError
    
# create the evaluation dir if the latter does not exist
if not os.path.isdir(args.evaluation_dir):
    os.mkdir(args.evaluation_dir)


# Load the RNI

target_path = os.path.join(args.source_dir, filename)
year=int(filename.split('_')[1][:-5])

RNI = json.load(open(target_path))


"""
small helper to recompute the aggregations (circumvent the bug in DeepPVMapper)
"""
def recalculate_aggregation(outputs_dir, aux_dir, dpt):
    """
    outputs_dir: where the results of the model are located
    dpt: the number of the departement
    aux_dir: the location of the file with the communes

    """

    # load the file with the city info
    communes=json.load(open(
        os.path.join(aux_dir,"communes_{}.json".format(dpt))
    ))


    # registry
    results=pd.read_csv(os.path.join(outputs_dir,"characteristics_{}.csv".format(dpt)))

    # first reassign the systems to the cities
    for index in tqdm.tqdm(results.index): # iterate over the installations
        lat, lon=results[results.index==index]['lat'].item(), \
            results[results.index==index]['lon'].item()
        point=Point((lon,lat))

        # iterate over the cities
        for commune in communes.keys():
            arr=communes[commune]['coordinates']
            code_commune=communes[commune]['properties']['code_insee']
            if len(arr) == 1: # case where the commune in a single polygon
                arr=np.array(arr).squeeze(0)
                poly=Polygon(arr)
            else:#handle the multipolygons
                #poly=[]
                for item in arr:
                    
                    item=np.array(item)

                    if len(item.shape) < 2:
                        continue

                    if item.shape[0]==1:
                        item=item.squeeze(0)                    
                    poly=Polygon(item)





            # now that we defined the polygon, look for the intersection.
            if poly.contains(point):
                results.loc[results.index==index,'city']=float(code_commune)

    # now compute the aggregation
    aggregated_capacity = results[['kWp', 'city']].groupby(['city']).sum()

    # count the installations 
    installations_count = results[['city', 'kWp']].groupby(['city']).count()
    installations_count.columns = ['count']

    # average the localization, surface and installed capacity
    means = results[['surface', 'city', 'lat', 'lon', 'kWp']].groupby(['city']).mean()
    means.columns = ['avg_surface', 'lat', 'lon', 'avg_kWp']

    # aggregate in a single dataframe and save it in the outputs directory.
    aggregated = pd.concat([aggregated_capacity, installations_count, means], axis=1)
    aggregated = aggregated[['count', 'kWp', 'avg_surface', 'avg_kWp', 'lat', 'lon']] # reorder the columns.

    
    # save the files 
    aggregated.to_csv(
        os.path.join(outputs_dir, 'aggregated_characteristics_{}.csv'.format(dpt))
    )
    results.to_csv(
        os.path.join(outputs_dir, "characteristics_{}.csv".format(dpt))
    )
            
    return aggregated

# load the outputs
if args.recompute:
    aggregation=recalculate_aggregation(outputs_dir, aux_dir, dpt)
else:
    aggregation = pd.read_csv(os.path.join(outputs_dir, 'aggregated_characteristics_{}.csv'.format(dpt))).set_index('city')


"""
Cleans the RNI and returns a clean dataframe
"""

def refactor_rni(RNI, dpt, year):
    """
    refactors the RNI to keep the aggregated installations 
    registered in the departement of interest
    """
    
    if dpt < 10:
        if not isinstance(dpt, str):
            dpt = '0' + str(dpt)
    else:
        if not isinstance(dpt, str): # convert the departement number as a str
            dpt = str(dpt)
    
    # first filtering : retain only aggregated small installations
    if year==2023:
        targets = [rni for rni in RNI if rni['nominstallation'] == 'Agrégation des installations de moins de 36KW']

    else:
        targets = [rni['fields'] for rni in RNI if rni['fields']['nominstallation'] == 'Agrégation des installations de moins de 36KW']
    # keep installations that have a departement code
    
    filtered_targets, not_localized = [], []
    
    for target in targets:

        if 'codedepartement' not in target.keys():
            not_localized.append(target)
        elif target['codedepartement'] == dpt:

            filtered_targets.append(target)
            
    # compute the installed capacity and number of installations
    not_localized_cap = sum([item['puismaxrac'] for item in not_localized])
    not_localized_count = sum([item['nbinstallations'] for item in not_localized])
        
    
    # now focus on the installations that are localized on the departement of interest
    # and filter those that do not have a commune
    rni_baseline, no_commune, missing_keys = [], [], []

    for filtered_target in filtered_targets:
        if 'codeinseecommune' in filtered_target.keys():
            
            if 'puismaxrac' in filtered_target.keys():
                code_commune = filtered_target['codeinseecommune']
                aggregated_capacity = filtered_target['puismaxrac']
                installations_count = filtered_target['nbinstallations']

                values = [code_commune, aggregated_capacity, installations_count]
                rni_baseline.append(values)
            else: 
                missing_keys.append(filtered_target)

        else:
            no_commune.append(filtered_target)
            
    no_commune_cap = sum([item['puismaxrac'] for item in no_commune])
    no_commune_count = sum([item['nbinstallations'] for item in no_commune])
    
    # now that we have the list for the complete departement
    # and kept track of the unassigned installations, compute the reference
    # dataframe
    df = pd.DataFrame(rni_baseline, columns = ['city', 'kWc', 'count'])
    df = df.groupby(['city']).sum()
    
    df['count_missing_overall'] = not_localized_count
    df['capacity_missing_overall'] = not_localized_cap

    df['missing_dpt'] = no_commune_count
    df['capacity_missing_dpt'] = no_commune_cap

    df.index = df.index.astype(int)
    
    return df

def compute_metrics(table):
    """
    computes accuracy metrics and summarizes them in a pd.DataFrame. 

    metrics are taken from Mayer(2022) for the estimation of the installed capacity
    - MAPE
    - MedAE
    - MAE
    - detection ratio

    Means are taken over the whole dataframe.

    for the counts, we consider the deviation 
    D > 0 indicates an underreport
    D < 0 indicates an overreport
    D = 0 indicates a perfect match


    to assess representativeness, we compare the mean installed capacity 
    (estimated and real) AE for this quantity
    
    if mean_AE = 0 : correct estimation of the installation size
    if mean_AE < 0 : underestimation
    if mean_AE > 0 : overestimation
    
    """
    
    table['APE'] = np.abs((table['target_kWp'] - table['est_kWp']) / (table['target_kWp'])) * 100
    table['AE'] = np.abs(table['target_kWp'] - table['est_kWp'])
    table['ratio'] = table['est_kWp'] / table['target_kWp']
    
    mape = np.mean(table['APE'])
    mae = np.median(table['AE'])
    mean_ratio = np.mean(table['ratio'])
    
    table['MAPE'] = mape
    table["MAE"] = mae
    table['mean_ratio'] = mean_ratio
    
    # representativeness
    table['mean_target'] = table['target_kWp'] / table['target_count']
    table['mean_est'] = table['est_kWp'] / table['est_count']
    
    table['mean_AE'] = table['mean_target'] - table['mean_est']
    table['mean_APE'] = - ((table['mean_target'] - table['mean_est']) / table['mean_target']) * 100
    
    table['deviation'] = - ((table['target_count'] - table['est_count']) / table['target_count']) * 100
    
    return table

def compare(reference, outputs):
    """
    compares the reference table and the aggregated outputs.

    computes the accuracy metrics on the intersection, 
    indicates the missing indices in each dataframe
    """

    print(reference.shape)

    print(outputs.shape)
    # set the index to match the type of the index of the aggregated dataframe
    reference.index=reference.index.astype(float)
    stats = reference.merge(outputs, left_index=True, right_index=True)
        
    # record the indices for which either no detection is made or no installatinos are recorded
    no_detection = [index for index in reference.index.values if not index in outputs.index.values]
    no_reference = [index for index in outputs.index.values if not index in reference.index.values]

    # reshape and rename the columns
    stats = stats[['count_x', 'count_y', 'kWc', 'kWp']]
    stats.columns = ['target_count', 'est_count', 'target_kWp', 'est_kWp']

    # add the number of communes w/o detection and w/o reference in the dataframe
    stats['no_detection'] = len(no_detection)
    stats['no_reference'] = len(no_reference)
    
    stats['missing_count'] = sum(reference['count'][no_detection].values)
    stats['missing_kWp'] = sum(reference['kWc'][no_detection].values)
    
    stats['excess_count'] = sum(outputs['count'][no_reference].values)
    stats['excess_kWp'] = sum(outputs['kWp'][no_reference].values)

    # compute the metrics
    stats = compute_metrics(stats)

    return {
        'stats' : stats,
        'no_detection' : no_detection,
        'no_reference' : no_reference

    }


def filter_values(results, threshold = 3):
    """filter the impossible values"""
    
    results['valid'] = results.apply(lambda r : (r['target_kWp'] / r['target_count'] > threshold), axis = 1)
    
    return results[results['valid'] == True]

def main():
    """
    main function.
    """

    # initialize the energy consumption tracker
    # tracker, startDate = carbon.initialize()
    # tracker.start()

    # get the reference installations

    reference = refactor_rni(RNI, dpt, year)

    # compare 
    stats = compare(reference, aggregation)

    #print(stats['stats'].shape)
    
    # filter the impossible values
    stats['stats'] = filter_values(stats['stats'])

    # save the results
    stats['stats'].to_csv(os.path.join(args.evaluation_dir, 'results_{}.csv'.format(dpt)))
    #print('Location for which no detection is made :', stats['no_detection'])
    #print('Location for which no reference is recorded :', stats['no_reference'])

    # save the carbon instances
    # tracker.stop() # stop the tracker
    # endDate = datetime.now()
    # carbon.add_instance(startDate, endDate, tracker, carbon_dir, dpt, 'eval')

if __name__ == '__main__':
    # stuff to execute

    main()
