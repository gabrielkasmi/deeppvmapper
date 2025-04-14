# -*- coding: utf-8 -*-

"""
Set of helper functions to compute the carbon impact of processing the detection 
pipeline
"""

from datetime import datetime
from codecarbon import EmissionsTracker
import os
import json

"""
Manage the file containing the outputs
"""

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

def add_instance(startDate, endDate, tracker, carbon_dir, dpt, instance_name):
    """
    adds an instance that recorded the energy consumption for running 
    this current part of the pipeline.
    
    tracker is a EmissionsTracker object.
    """

    duration = endDate - startDate
    # report the impact in a dictionnary :

    cpu_power = 140 * (duration.seconds / 3600) # convert processor' TDP into a consumption in Wh

    values = {
        "cpu"          : cpu_power, # raw data is in kWh, convert in Wh
        "gpu"          : tracker._total_gpu_energy.kWh * 1000, # raw data is in kWh, convert in Wh
        "ram"          : tracker._total_ram_energy.kWh * 1000,
        "total"        : tracker._total_energy.kWh * 1000,
        "startDate"    : str(startDate),
        "endDate"      : str(endDate),
        "duration"     : str(duration),
        "departement"  : dpt
    }

    # record the values of the session
    save_instance(carbon_dir, instance_name, startDate, values)

    return None

def initialize():
    """
    initializes the energy tracking module
    """

    return EmissionsTracker(), datetime.now()

def save_instance(carbon_dir, instance_name, date, values):
    """
    record the session to the existing instance or create a new instance
    """

    key_date = date.strftime('%d%m%y-%H%M%S')

    if os.path.isfile(os.path.join(carbon_dir, "carbon_logs.json")):
        
        instance = json.load(open(os.path.join(carbon_dir, "carbon_logs.json")))

        if instance_name in instance.keys(): # if the instance (e.g. cls) already exists, add a new session in addition to the existing ones
            instance[instance_name][key_date] = values
            
        
        else: # otherwise create a new instance with the session records.

            instance[instance_name] = {}
            instance[instance_name] = {key_date : values}
    else: # if the file does ont exist, create a new instance.

        instance = {instance_name : {key_date : values}}
        
    # save
    with open(os.path.join(carbon_dir, "carbon_logs.json"), "w") as f:

        json.dump(instance, f)

    # remove the "emissions.csv" file
    os.remove('emissions.csv')

    return None