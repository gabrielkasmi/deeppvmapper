#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Takes as input a training dataset and a type of model. 
The model is trained and the best weights are saved in the desired location
Checkpoints weights are stored during training in a designated folder.

Models include :
DeepLabv3 (DeepSolar)

Training features : 
- Weighted sampling of the items in the test dataset to balance the training
- Augmentations for the training set only
"""

# Libraries
import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
import random
from src import dataset, segmentation
import json
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
import warnings
import shutil
from PIL import ImageFile
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101


ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

# Arguments of the script 
parser = argparse.ArgumentParser(description = 'Model training')

# model (str) : the name of the model to be considered. Not implemented at this stage.

# name (str) : the name of the model. Weights will be saved as model_`name`.pth in the "models_path" folder.
parser.add_argument('--name', default = 'segmentation', help = "name of the model being trained", type = str)
# models_dir (str) : the name of the directory where the final model is saved after training.
parser.add_argument('--models_dir', default = "models", help = "folder where the model weights are stored", type = str)
# source_weights (str) : the name of the directory where the source model weights are stored.
parser.add_argument('--source_weights', default = "../../models", help = "folder where the source model weights are stored", type = str)
# dataset_dir (str) : the name of the directory where the training data is located. Should have subfolders train/validation/test
parser.add_argument('--dataset_dir', default = "dataset", help = "path to the training data. Folder should contain subfolders train/validation/test", type = str)
# results_dir (str) : the name of the directory where the outputs of the trained model will be stored
parser.add_argument('--results_dir', default = "results", help = "where the results should be stored", type = str)

# parallel (bool) : whether training is parallelized.
parser.add_argument('--parallel', default = False, help = "whether the model should be parallelized.", type = bool)
# device (str) : the device on which the model should be trained
parser.add_argument('--device', default = "cuda", help = "name of the GPU device on which to train the model", type = str)
# num devices (str) : the devices on which the model should be sent
parser.add_argument('--num_devices', default = '0,1,2', help = "the index of the devices on which tensors will be sent.", type = str)

# batch size (int) : the batch size
parser.add_argument('--batch_size', default = 32, help = "batch size.", type = int)
# epochs (int) : the number of training epochs
parser.add_argument('--epochs', default = 25, help = "Number of training epochs.", type = int)
# seed (int) : the number of training epochs
parser.add_argument('--seed', default = 42, help = "Random seed.", type = int)
# threshold (float) : the default classification threshold
parser.add_argument('--threshold', default = 0.5, help = "Default classification threshold.", type = float)

args = parser.parse_args()

# Check that the dataset_dir contains training data
assert os.path.isdir(os.path.join(args.dataset_dir, 'train')), "dataset_dir folder does not contain training data."

# In the models directory : create the weights subfolder than will store the temporary weights 
# computed during training and removed at the end of the training.

if not os.path.isdir(args.models_dir): # Create the folder where the weights are stored
    os.mkdir(args.models_dir)

weights_dir = os.path.join(args.models_dir, "weights") # Create the folder where the intermediary weights will be stored
if not os.path.isdir(weights_dir):
    os.mkdir(weights_dir)


# Initialize the reproducibility seeds
seed = args.seed

torch.backends.cudnn.deterministic = True
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# Set up the dataloader and the associated transforms.
# Random transforms are only applied on the training set.

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
])

dataloader = {}

for case in ['train', 'validation', 'test']:
    
    images_dir = os.path.join(args.dataset_dir, case)
    masks_dir = os.path.join(args.dataset_dir, '{}_masks'.format(case))
    
    random_transform = (case == 'train')
    
    data = dataset.BDPVSegmentationDataset(masks_dir, images_dir, image_transform = transforms, random_transform = random_transform)
    database = DataLoader(data, batch_size = args.batch_size, drop_last = args.parallel)
    
    dataloader[case] = database


# Load the model
source_model_weights = os.path.join(args.source_weights, 'deeplabv3_weights.tar')

model = deeplabv3_resnet101(pretrained = True, progress = True)
model.classifier = DeepLabHead(2048, 1)
checkpoint = torch.load(source_model_weights, map_location = args.device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(args.device)

# Parallelize the model to do inference faster
if args.parallel:

    devices = [int(k) for k in args.num_devices.split(',')]
    
    model = nn.DataParallel(model, device_ids = devices)
    model.to(args.device)

     
"""
Main function of the script.
"""
def main():
    """
    main function. Trains a model on the input dataset
    then finetunes the model and saves it. 
    Intermiediary weights are removed at the end of the process.
    """

    ## Main function

    # items needed during the whole training

    writer = SummaryWriter()

    criterion = nn.BCELoss()
    parameters_to_update = [parameter for parameter in model.parameters()]     
    optimizer = optim.Adam(parameters_to_update, lr = 0.0001)

    # initialize the model
    segmentation_model = segmentation.BDPVSegmentationModel(model, args.device, dataloader, args.threshold, weights_dir, name = args.name, writer = writer)

    print('Training the model...')

    for epoch in range(args.epochs):
        
        print("Epoch {}/{} ..................".format(epoch + 1, args.epochs))

        # train the model
        # this steps trains and evaluates the model at regular time steps
        validation_losses = segmentation_model.train(criterion, optimizer)
        
        # at the end of each epoch, clean the models folder 
        # and keep only the best.
        best_model = validation_losses[np.nanmin(list(validation_losses.keys()))]
        
        for model_name in os.listdir(weights_dir):
            if not model_name == best_model:
                os.remove(os.path.join(weights_dir, model_name)) 

    print('Fine tuning the classification threshold...')
                
    # Now that training is complete, optimize the treshold of the best model
    segmentation_model.optimize_threshold(best_model, weights_dir)

    # save the IoU minimizer model
    final_model = torch.load(os.path.join(weights_dir, best_model), map_location = args.device)
    torch.save(final_model, os.path.join(args.models_dir, "model_{}.pth".format(args.name)))
    shutil.rmtree(weights_dir) #remove the directory

    # evaluate this model on the test set and print the performance
    outputs = segmentation_model.evaluate("model_{}.pth".format(args.name), args.models_dir)

    # Create the folder if the latter does not exist
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    with open(os.path.join(args.results_dir, 'results_{}.json'.format(args.name)), 'w') as f:
        json.dump(outputs, f, indent = 4)

    print("""Training complete. Best model has been saved in folder {}\n
        Corresponding performance on the test set : \n
        Jaccard : {:.2f}\n
        Model name : {} 
        """.format(args.models_dir, outputs['jaccard'], "model_{}.pth".format(args.name)))
    

    return None

if __name__=='__main__':
    main()


