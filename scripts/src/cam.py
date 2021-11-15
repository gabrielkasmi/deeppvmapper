# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


"""
A helper that computes the class activation maps. 

Drawn from https://www.kaggle.com/bonhart/inceptionv3-tta-grad-cam-pytorch for inception v3

"""

class SaveFeatures():
    """
    A class that extracts the pretrained activations

    register_forward_hook returns the input and output of a given layer
    during the foward pass.
    """

    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy

    def remove(self):
        self.hook.remove()


def compute_class_activation_map(feature_conv, weight_fc, class_idx):
    """
    core computation of the class activation map

    args : 

    feature_conv : the last convolutional layer (before the GAP step).
    Corresponds to the matrices A1, ..., Ak in Zhang et al (2016)

    weights_fc : the weights of the fully connected layer.
    Corresponds to the weights w1, ..., wk in Zhang et al (2016)

    class_idx : the class id, i.e; the class_id th neuron to activate


    returns : 
    cam_image : the computed class activation map. It's a 2d map that has the size of the 
    last convolutional layer.

    """
    _, nc, h, w = feature_conv.shape # extract the shape of the convolutional layer

    # computation of the cam
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w) # reshape as a 2d matrix

    # numerical stability
    cam = cam - np.min(cam)
    cam_image = cam / np.max(cam)

    return cam_image


def class_activation_map_inception(img, model, device = 'cuda'):
    """
    returns the cam of the image which name is passed as an argument.

    arguments:

    this function is designed for inception v3, so the final layer 
    is the layer Mixed_7c
    
    img : the image for which the CAM is computed. Should be a tensor.
    model : the model used for inference

    device : default is cuda. The important thing is that both the model and the tensor 
    are on the same device.

    """

    # send the model and the image on the device.    
    model.to(device)
    image = Variable(img.unsqueeze(0).to(device), requires_grad=True)

    # get the layer from the model
    final_layer = model._modules.get('Mixed_7c')

    # register the layer in order to retrieve its value
    # after the forward pass.
    activated_features = SaveFeatures(final_layer)
        
    # prediction of the model for that image. Do a forward pass
    prediction = model(image)
    predicted_probabilities = F.softmax(prediction).data.squeeze()
    activated_features.remove()  

    # get the values of the weights
    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    
    # id of the class inpued by the network
    # topk return the kth larges element of an input
    class_idx = torch.topk(predicted_probabilities,1)[1].int()
    
    # get the class activation map
    cam_image = compute_class_activation_map(activated_features.features(), weight_softmax, class_idx)
    
    return cam_image


def inception_cam_from_image(img_name, data, model, device = 'cuda'):
    """computes the class activation map given an image and a dataset 
    passed as input

    dataset should have a labels() method that extracts
    the dataframe containing the image and the labels

    args :
    img_name : the name of the image to extract

    data : the dataset

    model : the model that is used to compute the cam

    device (opt) defaut : cuda

    return : the class activation map for the image inputed
    and the corresponding image (as a tensor)
    """

    # extract the image from the dataset
    labels = data.labels()
    id_ = labels.loc[labels['img_name'] == img_name].index[0]
    img, _, _ = data.__getitem__(id_)

    return class_activation_map_inception(img, model, device = device), img