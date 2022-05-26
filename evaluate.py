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

"""
Computes the accuracy by comparing the outputs of the detection model
with the RNI
"""


# Arguments
parser = argparse.ArgumentParser(description = 'Computation of the accuracy')

parser.add_argument('--dpt', default = None, help = "Department to proceed", type=int)
parser.add_argument('--filename', default = None, help = "name of the RNI to consider", type = str)

parser.add_argument('--evaluation_dir', default = 'evaluation',help = 'location of the ground truth registry', type = str)
parser.add_argument('--outputs_dir', default = 'data',help = 'where the outputs are stored', type = str)


args = parser.parse_args()


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

target_path = os.path.join(args.evaluation_dir, filename)

RNI = json.load(open(target_path))


# load the outputs

aggregation = pd.read_csv(os.path.join(args.outputs_dir, 'aggregated_characteristics_{}.csv'.format(dpt))).set_index('city')

print(aggregation.index)



"""
Cleans the RNI and returns a clean dataframe
"""

def refactor_rni(RNI, dpt):
    """
    refactors the RNI to keep the aggregated installations 
    registered in the departement of interest
    """
    
    if not isinstance(dpt, str): # convert the departement number as a str
        dpt = str(dpt)
    
    # first filtering : retain only aggregated small installations
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
    rni_baseline, no_commune = [], []

    for filtered_target in filtered_targets:
        if 'codeinseecommune' in filtered_target.keys():
            code_commune = filtered_target['codeinseecommune']
            aggregated_capacity = filtered_target['puismaxrac']
            installations_count = filtered_target['nbinstallations']

            values = [code_commune, aggregated_capacity, installations_count]
            rni_baseline.append(values)

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
    
    table['APE'] = np.abs((table['target_kWp'] - table['est_kWp']) / (table['target_kWp']))
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
    
    table['mean_AE'] =  table['mean_est'] - table['mean_target']

    
    table['deviation'] = (table['target_count'] - table['est_count']) / table['target_count']
    
    return table

def compare(reference, outputs):
    """
    compares the reference table and the aggregated outputs.

    computes the accuracy metrics on the intersection, 
    indicates the missing indices in each dataframe
    """

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

    # compute the metrics
    stats = compute_metrics(stats)

    return {
        'stats' : stats,
        'no_detection' : no_detection,
        'no_reference' : no_reference

    }

def main():
    """
    main function.
    """

    # get the reference installations

    reference = refactor_rni(RNI, dpt)

    # compare 
    stats = compare(reference, aggregation)

    # save the results
    stats['stats'].to_csv(os.path.join(args.evaluation_dir, 'results_{}.csv'.format(dpt)))
    print('Location for which no detection is made :', stats['no_detection'])
    print('Location for which no reference is recorded :', stats['no_reference'])


if __name__ == '__main__':
    # stuff to execute

    main()