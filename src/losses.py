import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

        
def Loglike_loss(y_true, y_pred, n_intervals=10):
    '''
    y_true: Tensor.
        First half: 1 if individual survived that interval, 0 if not.
        Second half: 1 for time interval before which failure has occured, 0 for other intervals.
    y_pred: Tensor.
        Predicted survival probability (1-hazard probability) for each time interval.
    '''
    
    cens_uncens = torch.clamp(1.0 + y_true[:,0:n_intervals]*(y_pred-1.0), min=1e-5)
    uncens = torch.clamp(1.0 - y_true[:,n_intervals:2*n_intervals]*y_pred, min=1e-5)
    loss = -torch.mean(torch.log(cens_uncens) + torch.log(uncens))
        
    return loss


def L2_Regu_loss(_, weights, alpha=0.1):
    '''
    Loss for L2 Regularization on weights
    '''
    
    loss = 0
    for weight in weights:
        loss += torch.square(weight).sum()

    return alpha * loss


def Dice_loss(y_true, y_pred):
    
    ndims = len(list(y_pred.size())) - 2
    vol_axes = list(range(2, ndims + 2))
    
    top = 2 * (y_true * y_pred).sum(dim=vol_axes)
    bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
    dice = torch.mean(top / bottom)
    
    return -dice


def Focal_loss(y_true, y_pred, alpha=0.25, gamma=2, epsilon = 1e-5):
    
    y_pred = torch.clamp(y_pred, min=epsilon, max=1-epsilon)
    logits = torch.log(y_pred / (1 - y_pred))
    weight_a = alpha * torch.pow((1 - y_pred), gamma) * y_true
    weight_b = (1 - alpha) * torch.pow(y_pred, gamma) * (1 - y_true)
    loss = torch.log1p(torch.exp(-logits)) * (weight_a + weight_b) + logits * weight_b
    
    return torch.mean(loss)


def Seg_loss(y_true, y_pred):
    return Dice_loss(y_true, y_pred) + Focal_loss(y_true, y_pred)
