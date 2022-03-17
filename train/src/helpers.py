# -*- coding: utf-8 -*-

# Libraries
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.cluster.vq import whiten, kmeans

def NormalizeData(data):
    """helper to normalize in [0,1] for the plots"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    if isinstance(confusion_vector, np.ndarray):
        false_positives = np.sum(confusion_vector == np.inf).item()
        true_negatives = np.sum(np.isnan(confusion_vector)).item()
        true_positives = np.sum(confusion_vector == 1.).item()
        false_negatives = np.sum(confusion_vector == 0.).item()

    else:

        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()
    
    return true_positives, false_positives, true_negatives, false_negatives

def compute_pr_curve(results):
    """computes the PR curve as a function of the keys
    passed as input, typically the detection thresholds
    """

    thresholds = list(results.keys())
    precision, recall = [], []

    for threshold in thresholds:
        tp, fp, fn = results[threshold]['true_positives'], results[threshold]['false_positives'], results[threshold]['false_negatives']
        precision.append(np.divide(tp,(tp + fp)))
        recall.append(np.divide(tp,(tp + fn)))

    return precision, recall

def return_f1(precision, recall):
    f1 = 2 * (np.array(precision) * np.array(recall)) / (np.array(precision) + np.array(recall)) 

    return 2 * (np.array(precision) * np.array(recall)) / (np.array(precision) + np.array(recall))

def confusion_samples(prediction, truth, names):
    """ Computes the confusion matrix and returns a list with
    the TP/FP/TN/FN names
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)
    
    # convert as an aray
    confusion_vector= confusion_vector.cpu().detach().numpy()
    
    true_positives = np.where(confusion_vector == 1)[0]
    false_positives = np.where(confusion_vector == float('inf'))[0]
    true_negatives = np.where(np.isnan(confusion_vector))[0]
    false_negatives = np.where(confusion_vector == 0)[0]
        
    def check_if_empty(index_items, items):
        """returns an empty list if index_items is empty"""
        return [items[i] for i in index_items]   
    
    return check_if_empty(true_positives, names), check_if_empty(false_positives, names), check_if_empty(true_negatives, names), check_if_empty(false_negatives, names)   

def confusion_samples_with_probs(prediction, truth, names, probabilities):
    """ Computes the confusion matrix and returns a list with
    the TP/FP/TN/FN names
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    # convert as an aray
    confusion_vector = confusion_vector.cpu().detach().numpy()
    probabilities = probabilities.cpu().detach().numpy()

    true_positives = np.where(confusion_vector == 1)[0]
    false_positives = np.where(confusion_vector == float('inf'))[0]
    true_negatives = np.where(np.isnan(confusion_vector))[0]
    false_negatives = np.where(confusion_vector == 0)[0]
    
    def check_if_empty(index_items, items_1, items_2):
        """returns an empty list if index_items is empty"""
        return [items_1[i] for i in index_items], [items_2[i] for i in index_items]   

    return check_if_empty(true_positives, names, probabilities), check_if_empty(false_positives, names, probabilities), check_if_empty(true_negatives, names, probabilities), check_if_empty(false_negatives, names, probabilities)   

def get_activation_map(image_name, model, data, layer = 'Mixed_7c'):
    """
    returns the cam of the image which name is passed as an argument.
    """

    labels = data.labels()
    model.to(device)

    # get the id of the image in the dataset
    id_ = labels.loc[labels['img_name'] == image_name].index[0]
    img, label, name = data.__getitem__(id_)

    # convert the image as a variable
    prediction_var = Variable(img.unsqueeze(0).to(device), requires_grad=True)
        
    # class to extract the desired feature layer
    class SaveFeatures():
        features=None
        def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
        def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
        def remove(self): self.hook.remove()
            
    # get the final pooling layer of the model
    final_layer = model._modules.get(layer)
    activated_features = SaveFeatures(final_layer)

    # prediction of the model for that image
    prediction = model(prediction_var)
    pred_probabilities = F.softmax(prediction).data.squeeze()
    activated_features.remove()    

    def get_cam(feature_conv, weight_fc, class_idx):
        """computes the cam"""
        _, nc, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return [cam_img]

    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

    # id of the class inpued by the network
    class_idx = torch.topk(pred_probabilities,1)[1].int()

    # get the class activation map
    overlay = get_cam(activated_features.features, weight_softmax, class_idx)

    return img, label, overlay

def compute_width_and_length(XY):
    """computes the width and the length of the mnimum bounding rectangle
    associated with the polygon passed as input
    
    TODO. modifier pour avoir un NAN quand le polygone est un point.
    
    """

    # check if the array is empty. If this is the case, then NaNs will be returned.
    if np.sum(XY) == 0:
        width, length = np.nan, np.nan
    
    # Compute the mnimum bounding rectangle
    try : 
        pol = Polygon(XY).minimum_rotated_rectangle
    except : 
        return np.nan, np.nan 

    if type(pol) == Point: # degenerate case
        return np.nan, np.nan

    # Get coordinates of polygon vertices
    try :
        x, y = pol.exterior.coords.xy

        # get length of bounding box edges
        edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))

        # get length of polygon as the longest edge of the bounding box
        length = max(edge_length)

        # get width of polygon as the shortest edge of the bounding box
        width = min(edge_length)

        return width, length
    except : # degenerate case otherwise
        return np.nan, np.nan

def compute_image_mask(img, coordinates):
    """
    given an image and the coordinates of the array
    returns the mask associated with the image, i.e.
    the pixels where the array is located.  
    """
    
    pixels = np.empty(img.size, dtype = int) # initialize the mask
    
    polygon = Polygon(coordinates) # Convert the coordinates as a polygon

    # activate only the pixels that are in the polygon
    for x,y in np.ndindex(pixels.shape):
        point = Point(y,x)
        pixels[x,y] = polygon.contains(point) 

    return pixels



def return_rgb(img, mask):
    """
    computes the distribution within the 
    r, g and b channels 
    returns the filtered image as well.
    """
    arr_im = np.array(img) # convert the image as an array

    r = []
    g = []
    b = []
    for x in range(arr_im.shape[0]):
        for y in range(arr_im.shape[1]):
            if mask[x,y] == 0:
                arr_im[x,y,:] = np.zeros(arr_im.shape[2]) # set pixels that are outside of the array to 0
            else:
                r.append(arr_im[x,y,0])  
                g.append(arr_im[x,y,1])
                b.append(arr_im[x,y,2])
    return [r, g, b], arr_im



def compute_buffer_mask(img, coordinates, buffer_size = 3.0):
    """computes the buffer mask of the inputed image
    this buffer mask excludes the region where the array
    is located
    """
    # compute the base mask and the larger mask
    mask = compute_image_mask(img, coordinates)
    
    # compute the larger polygon
    buffer_pixels = np.empty(img.size, dtype = int) # initialize the mask


    try :     
        polygon = Polygon(Polygon(coordinates).buffer(buffer_size)) # Convert the coordinates as a polygon
    except : 
        # in case of a ill posed polygon :
        # take the convex hull of the buffer
        x, y = Polygon(coordinates).buffer(buffer_size).convex_hull.exterior.coords.xy
        xy_ = np.array([x.tolist(), y.tolist()]).transpose()
        polygon = Polygon(xy_)

    # activate only the pixels that are in the polygon
    for x,y in np.ndindex(buffer_pixels.shape):
        point = Point(y,x)
        buffer_pixels[x,y] = polygon.contains(point) 

    # returns the difference of the two, so that the array is excluded.
    return buffer_pixels - mask



def return_dominant_color(rbg_img):
    """computes the dominant color of the image
        pixels with intensity 0 are filtered out
    
    """

    # Store RGB values of all pixels in lists r, g and b
    r = []
    g = []
    b = []

    for row in rbg_img:
        for temp_r, temp_g, temp_b in row:
            if temp_r > 0:
                r.append(temp_r)
            if temp_g > 0:
                g.append(temp_g)
            if temp_b > 0:
                b.append(temp_b)  
    # Saving as DatFrame
    batman_df = pd.DataFrame({'red' : r,
                          'green' : g,
                          'blue' : b})

    # Scaling the values
    batman_df['scaled_color_red'] = whiten(batman_df['red'])
    batman_df['scaled_color_blue'] = whiten(batman_df['blue'])
    batman_df['scaled_color_green'] = whiten(batman_df['green'])

    cluster_centers, _ = kmeans(batman_df[['scaled_color_red',
                                       'scaled_color_blue',
                                       'scaled_color_green']], 1) # 1 cluster as we are looking for the dominant color

    dominant_colors = []

    # Get standard deviations of each color
    red_std, green_std, blue_std = batman_df[['red',
                                          'green',
                                          'blue']].std()

    for cluster_center in cluster_centers:
        red_scaled, green_scaled, blue_scaled = cluster_center

        # Convert each standardized value to scaled value
        dominant_colors.append((
            red_scaled * red_std / 255,
            green_scaled * green_std / 255,
            blue_scaled * blue_std / 255
        ))  

    return dominant_colors


def return_gs(img, mask):
    """
    computes the distribution within the 
    r, g and b channels 
    returns the filtered image as well.
    """
    arr_im = np.array(img) # convert the image as an array

    r = []
    
    for x in range(arr_im.shape[0]):
        for y in range(arr_im.shape[1]):
            if mask[x,y] == 0:
                arr_im[x,y] = 0
            else:
                r.append(arr_im[x,y])  

    return r, arr_im


def confusion_samples_embeddings(prediction, truth, names, embeddings):
    """ Computes the confusion matrix and returns a list with
    the TP/FP/TN/FN names
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)
    
    # convert as an aray
    confusion_vector= confusion_vector.cpu().detach().numpy()
    
    true_positives = np.where(confusion_vector == 1)[0]
    false_positives = np.where(confusion_vector == float('inf'))[0]
    true_negatives = np.where(np.isnan(confusion_vector))[0]
    false_negatives = np.where(confusion_vector == 0)[0]
    
    tp_embeddings = embeddings[true_positives,:]
    fp_embeddings = embeddings[false_positives,:]
    tn_embeddings = embeddings[true_negatives,:]
    fn_embeddings = embeddings[false_negatives,:]

    
    def check_if_empty(index_items, items):
        """returns an empty list if index_items is empty"""
        return [items[i] for i in index_items]   
    
    return check_if_empty(true_positives, names), check_if_empty(false_positives, names), check_if_empty(true_negatives, names), check_if_empty(false_negatives, names), tp_embeddings, fp_embeddings, tn_embeddings, fn_embeddings   

def reshape_as_array(results, img_list):
    """
    reshapes the results from the dictionnary `results`
    as an array. The first column of the array encodes the status
    of the image :
        0 : true positive
        1 : false positive
        2 : true negative
        3 : false negative
    
    returns embeddings : a matrix where each row encodes the embedding of the corersponding
    sample.
    """
    
    # split the keys in two parts : one matches the name of the image and the 
    # index in the whole dataset
    # the other matches the index of the image in the batch and its embedding 
    keys_name_batch_index = list(results.keys())[::2]
    keys_batch_index_embedding =list(results.keys())[1::2]
        
    # instantiate the empty array
    embeddings = np.empty((len(img_list), 2048 + 1))
    
    # dictionnary that will code each case as a number
    category = {keys_name_batch_index[i]: i for i in range(len(keys_name_batch_index))}
    
    
    # initialize a list of unsorted embeddings.
    # this list contains tuples where the first element is the image name
    # and the second element is the embedding, so a (2048,) np.array.
    # the third item is the encoding (0, 1, 2, 3) of the image; 
    results_unsorted = []
    
    # loop per case (true positive, false positive, etc)
    for key_name, key_embedding in zip(keys_name_batch_index, keys_batch_index_embedding):
        # for each case, extract in each batch the name of the image and its embedding
        for name_index, index_embedding in zip(results[key_name], results[key_embedding]):
            for i in range(len(index_embedding)): # both lists have the same length (by construction)
                results_unsorted.append((name_index[i], index_embedding[i], category[key_name]))
    
    # now we sort each image.
    for i, img in enumerate(img_list):
        for result in results_unsorted:
            if img == result[0]:
                embeddings[i,0] = result[2] # add the encoding in the first column
                embeddings[i,1:] = result[1] # add the 2048-dimensional encoding.
    
    return embeddings
def rescaler(XY, size = 299):

    def coord_rescaler(x, size = 299):
        x = min(size, x)
        x = max(0, x)
        return x

    for i, row in enumerate(XY):
        x, y = row
        print(x,y)
        x, y = coord_rescaler(x), coord_rescaler(y)
        print(x,y)
        XY[i,:] = np.array((x,y))
        
    return XY
    

def reshape_mask(mask):

    """
    reshpes the mask as a polygon. Returns the input
    if the polygon is a line.
    """

    coords = np.column_stack(np.where(mask > 0))
    try : 
        x,y = Polygon(coords).convex_hull.exterior.coords.xy
        return np.array([y.tolist(), x.tolist()]).transpose()
    except : 
        return coords



def confusion_matrix(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    cm = prediction / truth
    
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)
    
    true_positives = torch.tensor([torch.sum(cm[i] == 1).item() for i in range(cm.shape[0])])
    false_positives = torch.tensor([torch.sum(cm[i] == float('inf')).item() for i in range(cm.shape[0])])
    true_negatives = torch.tensor([torch.sum(torch.isnan(cm[i])).item() for i in range(cm.shape[0])])
    false_negatives = torch.tensor([torch.sum(cm[i] == 0).item() for i in range(cm.shape[0])])
    
    return true_positives, false_positives, true_negatives, false_negatives

def jaccard(pred, truth):
    """
    computes the jaccard index based on the
    confusion matrix
    """
    
    tp, fp, _, fn = confusion_matrix(pred, truth)
    
    jaccards = tp / (tp + fp + fn)
        
    return np.mean(jaccards.cpu().detach().numpy())