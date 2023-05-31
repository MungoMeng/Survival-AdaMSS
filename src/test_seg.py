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
         load_model,
         save_dir):
    
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
    model = networks.DeepMSS_Seg()
    print('loading', load_model)
    state_dict = torch.load(load_model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # evaluate on training set
    T_inter_train = T_union_train = 0
    N_inter_train = N_union_train = 0
    for train_image in train_samples:
        
        # load subject
        PT, CT, Seg_T, Seg_N, _ = datagenerators.load_by_name(data_dir, train_image)
        PT = torch.from_numpy(PT).to(device).float()
        CT = torch.from_numpy(CT).to(device).float()
        
        with torch.no_grad():
            pred = model(PT, CT)
        
        Seg_T_pred = pred[0].detach().cpu().numpy().squeeze()
        Seg_N_pred = pred[1].detach().cpu().numpy().squeeze()
        
        _, Seg_T_pred = cv2.threshold(Seg_T_pred,0.5,1,cv2.THRESH_BINARY)
        T_inter_train = T_inter_train + np.sum(Seg_T_pred * Seg_T)
        T_union_train = T_union_train + np.sum(Seg_T_pred + Seg_T)
        
        _, Seg_N_pred = cv2.threshold(Seg_N_pred,0.5,1,cv2.THRESH_BINARY)
        N_inter_train = N_inter_train + np.sum(Seg_N_pred * Seg_N)
        N_union_train = N_union_train + np.sum(Seg_N_pred + Seg_N)
        
        # save to nii file
        if save_dir != './':
            save_path = save_dir+bytes.decode(train_image[:-4])
            print('saving to', save_path)
            
            nii_img = nib.Nifti1Image(Seg_T_pred, np.eye(4))
            nib.save(nii_img, save_path+'_Seg_T.nii.gz')
    
            nii_img = nib.Nifti1Image(Seg_N_pred, np.eye(4))
            nib.save(nii_img, save_path+'_Seg_N.nii.gz')
    
            Seg_pred = Seg_T_pred + Seg_N_pred
            Seg_pred[Seg_pred>1.0] = 1.0
            nii_img = nib.Nifti1Image(Seg_pred, np.eye(4))
            nib.save(nii_img, save_path+'_Seg_TN.nii.gz')
    
    # evaluate on testing set
    T_inter_test = T_union_test = 0
    N_inter_test = N_union_test = 0
    for test_image in test_samples:
        
        # load subject
        PT, CT, Seg_T, Seg_N, _ = datagenerators.load_by_name(data_dir, test_image)
        PT = torch.from_numpy(PT).to(device).float()
        CT = torch.from_numpy(CT).to(device).float()
        
        with torch.no_grad():
            pred = model(PT, CT)
        
        Seg_T_pred = pred[0].detach().cpu().numpy().squeeze()
        Seg_N_pred = pred[1].detach().cpu().numpy().squeeze()
        
        _, Seg_T_pred = cv2.threshold(Seg_T_pred,0.5,1,cv2.THRESH_BINARY)
        T_inter_test = T_inter_test + np.sum(Seg_T_pred * Seg_T)
        T_union_test = T_union_test + np.sum(Seg_T_pred + Seg_T)
        
        _, Seg_N_pred = cv2.threshold(Seg_N_pred,0.5,1,cv2.THRESH_BINARY)
        N_inter_test = N_inter_test + np.sum(Seg_N_pred * Seg_N)
        N_union_test = N_union_test + np.sum(Seg_N_pred + Seg_N)
        
        # save to nii file
        if save_dir != './':
            save_path = save_dir+bytes.decode(test_image[:-4])
            print('saving to', save_path)
            
            nii_img = nib.Nifti1Image(Seg_T_pred, np.eye(4))
            nib.save(nii_img, save_path+'_Seg_T.nii.gz')
    
            nii_img = nib.Nifti1Image(Seg_N_pred, np.eye(4))
            nib.save(nii_img, save_path+'_Seg_N.nii.gz')
    
            Seg_pred = Seg_T_pred + Seg_N_pred
            Seg_pred[Seg_pred>1.0] = 1.0
            nii_img = nib.Nifti1Image(Seg_pred, np.eye(4))
            nib.save(nii_img, save_path+'_Seg_TN.nii.gz')
     
    # calculat the mean results
    Dice_T_train = 2*T_inter_train/T_union_train
    Dice_T_test = 2*T_inter_test/T_union_test
    Dice_N_train = 2*N_inter_train/N_union_train
    Dice_N_test = 2*N_inter_test/N_union_test
    Dice_train = np.mean([Dice_T_train,Dice_N_train])
    Dice_test = np.mean([Dice_T_test,Dice_N_test])
    print('Tumor Dice: {:.3f}/{:.3f}'.format(Dice_T_train, Dice_T_test))
    print('Node Dice: {:.3f}/{:.3f}'.format(Dice_N_train, Dice_N_test))
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
    parser.add_argument("--save_dir", type=str,
                        dest="save_dir", default='./',
                        help="save outputs to folder")

    args = parser.parse_args()
    test(**vars(args))
