# py imports
import os
import sys
import glob
import time
import cv2
import nibabel as nib
import numpy as np
import torch
from argparse import ArgumentParser

# project imports
import networks
import datagenerators


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
    model = networks.AdaMSS_Seg()
    print('loading', load_model)
    state_dict = torch.load(load_model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # evaluate on training set
    Tumor_inter_train = Tumor_union_train = 0
    Node_inter_train = Node_union_train = 0
    for train_image in train_samples:
        
        # load subject
        PET, CT, Seg_Tumor, Seg_Node, _ = datagenerators.load_by_name(data_dir, train_image)
        PET = torch.from_numpy(PET).to(device).float()
        CT = torch.from_numpy(CT).to(device).float()
        
        with torch.no_grad():
            pred = model(PET, CT)
        
        Seg_Tumor_pred = pred[0].detach().cpu().numpy().squeeze()
        Seg_Node_pred = pred[1].detach().cpu().numpy().squeeze()
        
        _, Seg_Tumor_pred = cv2.threshold(Seg_Tumor_pred,0.5,1,cv2.THRESH_BINARY)
        Tumor_inter_train = Tumor_inter_train + np.sum(Seg_Tumor_pred * Seg_Tumor)
        Tumor_union_train = Tumor_union_train + np.sum(Seg_Tumor_pred + Seg_Tumor)
        
        _, Seg_Node_pred = cv2.threshold(Seg_Node_pred,0.5,1,cv2.THRESH_BINARY)
        Node_inter_train = Node_inter_train + np.sum(Seg_Node_pred * Seg_Node)
        Node_union_train = Node_union_train + np.sum(Seg_Node_pred + Seg_Node)
    
    # evaluate on testing set
    Tumor_inter_test = Tumor_union_test = 0
    Node_inter_test = Node_union_test = 0
    for test_image in test_samples:
        
        # load subject
        PET, CT, Seg_Tumor, Seg_Node, _ = datagenerators.load_by_name(data_dir, test_image)
        PET = torch.from_numpy(PET).to(device).float()
        CT = torch.from_numpy(CT).to(device).float()
        
        with torch.no_grad():
            pred = model(PET, CT)
        
        Seg_Tumor_pred = pred[0].detach().cpu().numpy().squeeze()
        Seg_Node_pred = pred[1].detach().cpu().numpy().squeeze()
        
        _, Seg_Tumor_pred = cv2.threshold(Seg_Tumor_pred,0.5,1,cv2.THRESH_BINARY)
        Tumor_inter_test = Tumor_inter_test + np.sum(Seg_Tumor_pred * Seg_Tumor)
        Tumor_union_test = Tumor_union_test + np.sum(Seg_Tumor_pred + Seg_Tumor)
        
        _, Seg_Node_pred = cv2.threshold(Seg_Node_pred,0.5,1,cv2.THRESH_BINARY)
        Node_inter_test = Node_inter_test + np.sum(Seg_Node_pred * Seg_Node)
        Node_union_test = Node_union_test + np.sum(Seg_Node_pred + Seg_Node)
     
    # calculat the mean results
    Dice_Tumor_train = 2*Tumor_inter_train/Tumor_union_train
    Dice_Tumor_test = 2*Tumor_inter_test/Tumor_union_test
    Dice_Node_train = 2*Node_inter_train/Node_union_train
    Dice_Node_test = 2*Node_inter_test/Node_union_test
    Dice_train = np.mean([Dice_Tumor_train,Dice_Node_train])
    Dice_test = np.mean([Dice_Tumor_test,Dice_Node_test])
    print('Tumor Dice: {:.3f}/{:.3f}'.format(Dice_Tumor_train, Dice_Tumor_test))
    print('Node Dice: {:.3f}/{:.3f}'.format(Dice_Node_train, Dice_Node_test))
    print('Average Dice: {:.3f}/{:.3f}'.format(Dice_train, Dice_test))
    

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
