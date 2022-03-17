#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Takes as input a training dataset and a type of model. 
The model is trained and the best weights are saved in the desired location
Checkpoints weights are stored during training in a designated folder.

Models include : 
ResNet (baseline from Pytorch, trained on ImageNet)

To be included
Inception v3 (DeepSolar Germany) : to be included with a dataset containing
thumnails of 299x299 pixels.
ResNet DINO (trained with self supervision on ImageNet)
ViT (pretrained on ImageNet)

Training features : 
- Weighted sampling of the items in the test dataset to balance the data
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
import tqdm
from src import helpers, dataset
import json
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
import warnings
import shutil
from PIL import ImageFile
from torch.utils.data import WeightedRandomSampler
import pandas as pd
from torchvision.models import Inception3




ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

# Arguments of the script 
parser = argparse.ArgumentParser(description = 'Model training')

# model (str) : the name of the model to be considered. Not implemented at this stage.

# name (str) : the name of the model. Weights will be saved as model_`name`.pth in the "models_path" folder.
parser.add_argument('--name', default = None, help = "name of the model being trained", type = str)
# models_dir (str) : the name of the directory where the final model is saved after training.
parser.add_argument('--models_dir', default = "models", help = "folder where the model weights are stored", type = str)
# source_weights (str) : the name of the directory where the source model weights are stored.
parser.add_argument('--source_weights', default = "../../models", help = "folder where the model weights are stored", type = str)
# dataset_dir (str) : the name of the directory where the training data is located. Should have subfolders train/validation/test
parser.add_argument('--dataset_dir', default = "dataset", help = "path to the training data. Folder should contain subfolders train/validation/test", type = str)
# results_dir (str) : the name of the directory where the outputs of the trained model will be stored
parser.add_argument('--results_dir', default = "results", help = "where the results should be stored", type = str)
# device (str) : the device on which the model should be trained
parser.add_argument('--device', default = "cuda", help = "name of the GPU device on which to train the model", type = str)

# batch size (int) : the batch size
parser.add_argument('--batch_size', default = 128, help = "batch size.", type = int)
# epochs (int) : the number of training epochs
parser.add_argument('--epochs', default = 25, help = "Number of training epochs.", type = int)
# seed (int) : the number of training epochs
parser.add_argument('--seed', default = 42, help = "Random seed.", type = int)

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


# Initialize the SummaryWriter for TensorBoard
writer = SummaryWriter()

# Initialize the reproducibility seeds
seed = args.seed

torch.backends.cudnn.deterministic = True
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# Setting up the training of the model

# Define the set of transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomRotation((90,90)),
    torchvision.transforms.RandomRotation((-90,-90)),
    torchvision.transforms.RandomRotation((180,180)),
    torchvision.transforms.RandomRotation((-180,-180)),
    torchvision.transforms.RandomRotation((-270,-270)),
    torchvision.transforms.RandomRotation((270,270)),

    #torchvision.transforms.ColorJitter(brightness=.5, hue=.3),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
])

eval_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
])



"""
"""
def assign_weights(annotations, seed = args.seed):
    """
    assigns a weight according to the share of the label
    in the dataset
    """
    
    # set up the generator
    generator = torch.Generator()
    generator.manual_seed(seed)

    
    # load the label files
    labels = pd.read_csv(annotations, header = None)
    labels.columns = ['img', 'label']
    label_values = labels["label"].values
    
    # count the number of instances for each label
    class_count = np.array([len(np.where(label_values == y)[0]) for y in np.unique(label_values)])

    # compute the normalized weights. 
    # weights of the Oth class are normalized to 1
    weight = [1 / (class_count[i]/sum(class_count)) for i in range(len(class_count))]
    normalized_weight = [weight[i]/weight[0] for i in range(len(class_count))]
    
    # assign to each sample a weight based on its label value
    sample_weights = torch.from_numpy(np.array([normalized_weight[i] for i in label_values]))
    
    return WeightedRandomSampler(sample_weights.type('torch.DoubleTensor'), len(sample_weights))


# Construct the dataloader
dataloader = {}

for case in os.listdir(args.dataset_dir): # Cases are train, validation, test

    # open the subfolder
    path = os.path.join(args.dataset_dir, case)
    annotations = os.path.join(path, 'labels.csv')

    # Load the data
    # if we are in the validation and testing case, we do not
    # add transforms 
    if case == 'train':
        # include a weighted random sampler for the training data
        sampler = None # assign_weights(annotations)
        data = dataset.BDPVClassificationDataset(annotations, path, transform = train_transforms)
    else:
        sampler = None
        data = dataset.BDPVClassificationDataset(annotations, path, transform = eval_transforms)

    # Construct the dataloader
    database = DataLoader(data, batch_size = args.batch_size, sampler = sampler)

    # Add it to the dictionnary
    dataloader[case] = database
    
# Load the model and send it to the GPU
# Load the model fine-tuned on NRW
# 
model = Inception3(num_classes = 2, aux_logits = True, transform_input = False, init_weights = True) # Load the architecture
checkpoint = torch.load(os.path.join(args.source_weights, 'inceptionv3_weights.tar'), map_location = args.device) # Load the weights
model.load_state_dict(checkpoint['model_state_dict']) # Upload the weights in the model
model = model.to(args.device) # move the model to the device

# Train the model 
# Train

"""
A function that trains a model and return the model achieving 
the lowest validation loss
"""
def train(model, dataloader, args, weights_dir):
    """
    trains a model

    returns the best model name
    """

    # Initialization of the model

    # Layers to update : all layers
    for parameter in model.parameters():
        parameter.requires_grad = True
    
    # Loss 
    criterion = nn.BCELoss()

    # Parameters to update and optimizer
    params_to_update = [parameter for parameter in model.parameters()]        
    optimizer = optim.Adam(params_to_update, lr = 0.0001)

    # Training
    steps = 0
    threshold = 0.5 # Threshold set by default. Will be fine tuned afterwards

    # losses
    # at the end of each validation, save the name of the model and the value of the loss
    # to recover at the end of training the best model
    validation_losses = {}
    best_loss = 1e3 # initialize a best loss that will be beaten by the model.

    for i in range(args.epochs):

        print("Epoch {}/{} ..................".format(i+1, args.epochs))

        running_loss = 0 # The running loss is reset to 0 at each epoch.
        model.train() # Set the model into training mode

        for inputs, labels, _ in tqdm.tqdm(dataloader["train"]):

            steps += 1
            labels = labels.to(torch.float32)
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            
            optimizer.zero_grad()


            outputs, aux_outputs = model(inputs)
            outputs, aux_outputs = F.softmax(outputs, dim=1), F.softmax(aux_outputs, dim=1)
            outputs, aux_outputs = outputs[:,1], aux_outputs[:,1]

            loss = criterion(outputs, labels) + 0.4 * criterion(aux_outputs, labels)

            writer.add_scalar("Loss/train", loss.item(), steps)
            running_loss += loss.item() # accumulate the losses over the whole epoch
            
            loss.backward()
            optimizer.step()

            # Evaluate the model regularly
            if steps % int(20 * (len(dataloader['train']) / args.batch_size)) == 0:               

                test_loss = 0
                model.eval()

                with torch.no_grad():

                    true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0

                    for inputs, labels, _ in dataloader["validation"]:

                        labels = labels.to(torch.float32)
                        inputs, labels = inputs.to(args.device), labels.to(args.device)
                        outputs = model.forward(inputs)
                        outputs = F.softmax(outputs, dim=1) # the model returns the unnormalized probs. Softmax it to get probs
                        outputs = outputs[:,1]
                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.item()
                        predicted = (outputs >= threshold).long()  # return 0 or 1 if an array has been detected

                        # compute the accuracy of the classification

                        tp, fp, tn, fn = helpers.confusion(predicted, labels)
                        true_positives += tp
                        false_positives += fp
                        true_negatives += tn
                        false_negatives += fn

                # Compute the loss over the validation dataset        
                test_total = test_loss / len(dataloader["validation"])                            
                # Add to the SummaryWriter            
                writer.add_scalar("Loss/val", test_total, steps)
                
                # Compute the F1 score
                precision = np.divide(true_positives, (true_positives + false_positives))
                recall = np.divide(true_positives, (true_positives + false_negatives))
                
                f1 = 2 * np.divide((precision * recall), (precision + recall))

                writer.add_scalar('F1/val', f1, steps)
                model.train()

                # keep track of the loss of the model
                name = "model_{}_{}.pth".format(args.name, str(steps))
                validation_losses[test_total] = name

                # if the model achieved a new "sota", save it.
                if test_total < best_loss:
                    best_loss = test_total # update the loss
                    clean_folder(weights_dir) # Clean the folder if necessary (the folder cannot contain more than 10 models)
                    torch.save(model, os.path.join(weights_dir, name))

    writer.flush()
    model.eval()
    return validation_losses[np.nanmin(list(validation_losses.keys()))]


"""
A function that cleans the folder of the temporary models 
"""
def clean_folder(weights_dir, cap = 10):
    """
    remove the items that are in the weights_dir folder.
    as soon as the number exceeds cap items
    """

    if len(os.listdir(weights_dir)) > cap:

        for item in os.listdir(weights_dir):
            os.remove(os.path.join(weights_dir, item))

    return None


"""
A function that fine tunes the threshold 
"""
def optimize_threshold(dataloader, args, best_model, weights_dir):
    """
    finds the best threshold and returns its value, along with the precision, recall curves
    and the pr_auc
    """

    def return_f1(precision, recall):
        """
        Given an array of precisions and of recalls
        computes the F1 score
        """
        return 2 * (np.array(precision) * np.array(recall)) / (np.array(precision) + np.array(recall))

    # Load the model

    model = torch.load(os.path.join(weights_dir, best_model), map_location = args.device)
    model = model.to(args.device)
    model.eval()

    # Compute the embeddings on the test set
    # Thresholds to be considered
    results_models = {}
    thresholds = np.linspace(0.01,.99, 99)

    # Forward pass on the validation dataset and accumulate the probabilities 
    # in a single vector.

    probabilities = []
    all_labels = []

    with torch.no_grad():

        for data in tqdm.tqdm(dataloader["validation"]):
            images, labels, _ = data

            # move the images to the device
            images = images.to(args.device)

            labels = labels.detach().cpu().numpy()
            all_labels.append(list(labels))

            # calculate outputs by running images through the network and computing the prediction using the threshold
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy() # the model returns the unnormalized probs. Softmax it to get probs
            probabilities.append(list(probs[:,1]))

    # Convert the probabilities and labels as an array
    probabilities = sum(probabilities, [])
    probabilities = np.array(probabilities)

    labels = sum(all_labels, [])
    labels = np.array(labels)

    results_models = {}

    for threshold in thresholds: # Use the probablity vector that has been computed 
                                 # To compute the precision and recall.

        # new key for the threshold under consideration
        results_models[threshold] = {}

        true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0

        # treshold the probabilities to get the predictions
        predicted = np.array(probabilities > threshold, dtype = int)

        # compute the true positives, true negatives, false positives and false negatives
        tp, fp, tn, fn = helpers.confusion(predicted, labels)
        true_positives += tp
        false_positives += fp
        true_negatives += tn
        false_negatives += fn

        # store the results
        results_models[threshold]['true_positives'] = true_positives
        results_models[threshold]['false_positives'] = false_positives
        results_models[threshold]['true_negatives'] = true_negatives
        results_models[threshold]['false_negatives'] = false_negatives

    # Compute the precision and recall
    precision, recall = helpers.compute_pr_curve(results_models)

    # Compute the F1 score
    f1 = return_f1(precision, recall)

    # Determine the best treshold (that maximizes F1) and the corresponding precision and recall
    best_threshold = thresholds[np.nanargmax(f1)]

    # save the results
    outputs = {}

    outputs['precision_curve'] = precision
    outputs['recall_curve'] = recall
    outputs['F1_scores'] = f1.tolist()
    outputs['best_threshold'] = best_threshold

    # return the results

    return outputs

"""
A function that computes the performance metric of the model that has been trained
"""
def compute_performance(dataloader, args, best_model, weights_dir, best_threshold):
    """Computes the performance of the best model
    on the test set given the best threshold fine tuned
    over the validation dataset"""

    # Load the model

    model = torch.load(os.path.join(weights_dir, best_model), map_location = args.device)
    model = model.to(args.device)
    model.eval()

    # Forward pass on the validation dataset and accumulate the probabilities 
    # in a single vector.

    probabilities = []
    all_labels = []

    with torch.no_grad():

        for data in tqdm.tqdm(dataloader["test"]):

            images, labels, _ = data


            # move the images to the device
            images = images.to(args.device)

            labels = labels.detach().cpu().numpy()
            all_labels.append(list(labels))

            # calculate outputs by running images through the network and computing the prediction using the threshold
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy() # the model returns the unnormalized probs. Softmax it to get probs
            probabilities.append(list(probs[:,1]))

    # Convert the probabilities and labels as arrays
    probabilities = sum(probabilities, [])
    probabilities = np.array(probabilities)

    labels = sum(all_labels, [])
    labels = np.array(labels)

    # Compute the precision and recall for the best threshold
    predicted = np.array(probabilities > best_threshold, dtype = int)

    # compute the confusion matrix
    tp, fp, _, fn = helpers.confusion(predicted, labels)

    # return the values
    precision = np.divide(tp, (tp + fp))
    recall = np.divide(tp, (tp + fn))
    f1 = np.divide(2 * precision * recall, precision + recall)


    return precision, recall, f1

"""
Main function of the script.
"""
def main():
    """
    main function. Trains a model on the input dataset
    then finetunes the model and saves it. 
    Intermiediary weights are removed at the end of the process.
    """

    # Train the model
    print('Training the model ...')
    best_model = train(model, dataloader, args, weights_dir)
    print('Model trained. Now finetuning the detection threshold...')
    # Fine tune the threshold
    outputs = optimize_threshold(dataloader, args, best_model, weights_dir)
    print("Detection threshold fine tuned. Now computing the model's accuracy.")
    # compute the performance metrics of the best model
    precision, recall, f1 = compute_performance(dataloader, args, best_model, weights_dir, outputs['best_threshold'])
    print(type(precision), type(recall))

    # save the results in the results directory
    outputs['precision'] = precision
    outputs["recall"] = recall
    outputs['f1'] = f1

    print(outputs)
    # Create the folder if the latter does not exist
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    with open(os.path.join(args.results_dir, 'results_{}.json'.format(args.name)), 'w') as f:
        json.dump(outputs, f, indent = 4)

    # Keep only the best model and clean the temporary
    # directory.
    final_model = torch.load(os.path.join(weights_dir, best_model), map_location = args.device)
    torch.save(final_model, os.path.join(args.models_dir, "model_{}.pth".format(args.name)))
    shutil.rmtree(weights_dir) #remove the directory

    # Print the name of the best model saved as... 
    # and its corresponding optimal threshold
    print("""Training complete. Best model has been saved in folder {}\n
        Corresponding performance : \n
        Precision : {:.2f}\n
        Recall : {:.2f}\n
        F1 : {:.2f}\n
        Model name : {} 
        """.format(args.models_dir, precision, recall, f1, best_model))

    return None

if __name__=='__main__':
    main()


