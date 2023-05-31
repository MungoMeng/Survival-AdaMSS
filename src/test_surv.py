# py imports
import os
import sys
import glob
import time
import cv2
import pandas as pd
import numpy as np
import torch
from argparse import ArgumentParser
from lifelines.utils import concordance_index

# project imports
import networks
import datagenerators


def Get_survival_time(Survival_pred):

    breaks = np.array([0,300,600,900,1100,1300,1500,2100,2700,3500,6000])
    
    intervals = breaks[1:] - breaks[:-1]
    n_intervals = len(intervals)
    
    Survival_time = 0
    for i in range(n_intervals):
        cumulative_prob = np.prod(Survival_pred[0:i+1])
        Survival_time = Survival_time + cumulative_prob * intervals[i]
    
    return Survival_time


def test(data_dir,
         train_samples,
         test_samples,
         device, 
         load_model):
    
    # prepare data files
    train_samples = np.load(train_samples, allow_pickle=True)
    test_samples = np.load(test_samples, allow_pickle=True)
    
    # device handling
    if 'gpu' in device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'
    
    # prepare the model
    model = networks.DeepMSS_Surv()
    print('loading', load_model)
    state_dict = torch.load(load_model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # evaluate on training set
    Survival_time_train = []
    Survival_label_train = []
    for train_image in train_samples:
        
        # load subject
        PET, CT, _, _, Label = datagenerators.load_by_name(data_dir, train_image)
        PET = torch.from_numpy(PET).to(device).float()
        CT = torch.from_numpy(CT).to(device).float()
        Survival_label_train.append(Label)
        
        with torch.no_grad():
            pred = model(PET, CT)
        
        Survival_pred = pred[0].detach().cpu().numpy().squeeze()
        Survival_time = Get_survival_time(Survival_pred)
        Survival_time_train.append(Survival_time)
    
    # evaluate on testing set
    Survival_time_test = []
    Survival_label_test = []
    for test_image in test_samples:
        
        # load subject
        PET, CT, _, _, Label = datagenerators.load_by_name(data_dir, test_image)
        PET = torch.from_numpy(PET).to(device).float()
        CT = torch.from_numpy(CT).to(device).float()
        Survival_label_test.append(Label)
        
        with torch.no_grad():
            pred = model(PET, CT)
        
        Survival_pred = pred[0].detach().cpu().numpy().squeeze()
        Survival_time = Get_survival_time(Survival_pred)
        Survival_time_test.append(Survival_time)
     
    # calculat the mean results
    Survival_label_train = np.array(Survival_label_train)
    Survival_label_test = np.array(Survival_label_test)
    cindex_train = concordance_index(Survival_label_train[:,0], Survival_time_train, Survival_label_train[:,1])
    cindex_test = concordance_index(Survival_label_test[:,0], Survival_time_test, Survival_label_test[:,1])
    print('C-index: {:.3f}/{:.3f}'.format(cindex_train, cindex_test))
    

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        dest="data_dir", default='./',
                        help="data folder")
    parser.add_argument("--train_samples", type=str,
                        dest="train_samples", default='./',
                        help="training samples")
    parser.add_argument("--test_samples", type=str,
                        dest="test_samples", default='./',
                        help="testing samples")
    parser.add_argument("--device", type=str, default='gpu0',
                        dest="device", help="cpu or gpuN")
    parser.add_argument("--load_model", type=str,
                        dest="load_model", default='./',
                        help="load best model")

    args = parser.parse_args()
    test(**vars(args))
