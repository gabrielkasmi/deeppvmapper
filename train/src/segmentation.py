# -*- coding: utf-8 -*-

"""
Class declaration for a segmentation model. 
"""

import os
import torch
import numpy as np
import tqdm
from torchmetrics.classification.jaccard import JaccardIndex

# Model class
class BDPVSegmentationModel():
    """
    wrapper that contains the utilities to train and do inference 
    with a model
    """
    def __init__(self, model, device, dataloader, threshold, weights_dir, name = None, writer = None):
        
        self.model = model
        self.device = device
        self.weights_dir = weights_dir
        self.train_data = dataloader['train']
        self.val_data = dataloader['validation']
        self.test_data = dataloader['test']
        self.writer = writer
        self.name = name
        self.threshold = threshold
        self.best_threshold = None
        self.steps = 0
        self.validation_losses = {}
        self.outputs = {}
        
        im, _, _ = next(iter(self.train_data))
        self.batch_size = im.shape[0]
        
    def inference(self, inference_model, criterion, model_dir = None):
        """
        does inference with the model passed as input
        returns the loss of the model
        """
        
        # if the model is a str, then look for the location of the model,
        # open it and load it
        if type(inference_model) == str:            
            model = torch.load(os.path.join(model_dir, inference_model), map_location = self.device)
            model.eval()
        else:
            model = inference_model  
            model.eval()
            
        test_loss = 0
        jaccards = {}
        index = 0
        
        iou = JaccardIndex(num_classes = 2) # Initialize the metric
        
        with torch.no_grad():

            for data in self.val_data:

                images, masks, _ = data

                # forward pass
                predicted = model.forward(images.to(self.device))
                predictions = predicted['out']
                # min-max scaling
                predictions = (predictions - torch.min(predictions)) / (torch.max(predictions) - torch.min(predictions) + 0.000000001)
                binary_outputs = (predictions >= self.threshold).long().cpu().detach()

                # compute the loss
                batch_loss = criterion(predictions, masks.to(self.device))
                test_loss += batch_loss.item() # accumulate the losses over the whole epoch

                # compute the jaccard index for the batch
                jaccard_temp = iou(binary_outputs, masks.type(torch.int64)).item()

                jaccards[index] = images.shape[0], jaccard_temp

            # Compute the loss over the validation dataset        
            test_total = test_loss / len(self.val_data)    
            model.train()

            # Add to the SummaryWriter     
            if self.writer is not None:
                self.writer.add_scalar("Loss/val", test_total, self.steps)
                # Compute the Jaccard index
                # each 0th item is the size of the batch
                # each 1st item is the value of Jaccard for this batch
                # we do the weighted average.

                jaccard = sum([jaccards[k][0] * jaccards[k][1] for k in jaccards.keys()]) / sum([jaccards[k][0] for k in jaccards.keys()])

                # add the value to the tensorboard
                self.writer.add_scalar('mIoU/val', jaccard, self.steps)

        return test_total           
        
    def train(self, criterion, optimizer):
        """
        Trains the model for one epoch
        Optionnally writes the outputs in a tensorboard writer 
        (if the latter is not none)
        
        performs SGD and evaluation of the model during training
        over the validation dataset at regular time steps.
        
        returns the name of the best model
        """
        
        running_loss = 0
                            
        for inputs, masks, _ in tqdm.tqdm(self.train_data):
            
            self.steps +=1
            masks = masks.to(torch.float32)
            
            # forward pass
            predicted = self.model.forward(inputs.to(self.device))
            predictions = predicted['out']
            # min-max scaling
            predictions = (predictions - torch.min(predictions)) / (torch.max(predictions) - torch.min(predictions) + 0.000000001)

            # compute the loss
            loss = criterion(predictions, masks.to(self.device))
            running_loss += loss.item() # accumulate the losses over the whole epoch
            
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", loss.item(), self.steps)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # evaluate the model regularly and update the 'sota'
            # model
            
            if self.steps % int(5 * (len(self.train_data) / self.batch_size)) == 0: 
                
                test_temp = self.inference(self.model, criterion) 
                
                # save the model
                name = "model_{}_{}.pth".format(self.name, str(self.steps))
                
                self.validation_losses[test_temp] = name
                
                # save the model
                torch.save(self.model, os.path.join(self.weights_dir, name))
                
        return self.validation_losses
                
    def optimize_threshold(self, inference_model, model_dir):
        """
        optimizes the classification threshold based
        necessarily takes as input a model.
        """
          
        model = torch.load(os.path.join(model_dir, inference_model), map_location = self.device)
        model.eval()
            
        jaccards = {}        
        iou = JaccardIndex(num_classes = 2) # Initialize the metric
        
        preds, truth = [], []
                        
        with torch.no_grad():
                                
            for data in self.val_data:
                

                images, masks, _ = data

                # forward pass
                predicted = model.forward(images.to(self.device))
                predictions = predicted['out']
                # min-max scaling
                predictions = (predictions - torch.min(predictions)) / (torch.max(predictions) - torch.min(predictions) + 0.000000001)

                # accumulates the prediction and the masks                
                preds.append(predictions.detach().cpu())
                truth.append(masks)
                
        # reshape as a tensor
        preds = torch.cat(preds)
        truth = torch.cat(truth).type(torch.int64)

        # now that we accumulated the predictions and grund truth,
        # compute the Jaccard associated to each threshold
        
        thresholds = np.linspace(0.01,.99, 99)
                        
        for threshold in thresholds:           

            # binarize the outputs
            binary_outputs = (preds >= threshold).long()
        
            # compute the jaccard index for the treshold
            jaccard_temp = iou(binary_outputs, truth).item()            
            jaccards[jaccard_temp] = threshold
            
            # save the jaccard maximizing threshold
            self.best_threshold = jaccards[np.nanmax(list(jaccards.keys()))]
            self.outputs['best_threshold'] = self.best_threshold
                
    def evaluate(self, best_model, model_dir):
        """
        evaluates the model on the test set
        model should be a str and is searched in the model_dir
        criterion is the loss.
        """
        
        model = torch.load(os.path.join(model_dir, best_model), map_location = self.device)
        model.eval()
            
        iou = JaccardIndex(num_classes = 2) # Initialize the metric
        
        preds, truth = [], []
                        
        with torch.no_grad():
                                
            for data in self.test_data:               

                images, masks, _ = data

                # forward pass
                predicted = model.forward(images.to(self.device))
                predictions = predicted['out']
                # min-max scaling
                predictions = (predictions - torch.min(predictions)) / (torch.max(predictions) - torch.min(predictions) + 0.000000001)

                # accumulates the prediction and the masks                
                preds.append(predictions.detach().cpu())
                truth.append(masks)
                
        # reshape as a tensor
        preds = torch.cat(preds)
        truth = torch.cat(truth).type(torch.int64)
        
        # compute the iou 
        binary_outputs = (preds >= self.best_threshold).long()        
        final_score = iou(binary_outputs, truth).item()    
        
        self.outputs["jaccard"] = final_score
        
        return self.outputs
        
        
# class that will be created to initialize a segmentation model
# for large scale inference.
# class BDPVSegmentationInference()