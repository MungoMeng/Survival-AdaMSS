import os, sys
import random
import numpy as np
import cv2
import random
import nibabel as nib  
import pandas as pd

import imgaug as ia
from imgaug import augmenters as iaa


def gen_seg(gen):
    
    while True:
        X = next(gen)
        PT = X[0]
        CT = X[1]
        Seg_T = X[2]
        Seg_N = X[3]
        
        #data augmentation
        PT, CT, Seg_T, Seg_N = Data_augmentation_seg(PT, CT, Seg_T, Seg_N)
        
        yield [PT, CT], [Seg_T, Seg_N]
        
        
def gen_surv(gen):
    
    while True:
        X = next(gen)
        PT = X[0]
        CT = X[1]
        Label = X[4]
        
        # data augmentation
        PT, CT = Data_augmentation_surv(PT, CT)
        
        # convert label to survival array
        Label = make_surv_array(Label[:,0], Label[:,1])
        
        # generate a zero tensor
        Zero = np.zeros((1))
        
        yield [PT, CT], [Label, Zero]
        
        
def gen_multitask(gen):
    
    while True:
        X = next(gen)
        PT = X[0]
        CT = X[1]
        Seg_T = X[2]
        Seg_N = X[3]
        Label = X[4]
        
        #data augmentation
        PT, CT, Seg_T, Seg_N = Data_augmentation_seg(PT, CT, Seg_T, Seg_N)
        
        # convert label to survival array
        Label = make_surv_array(Label[:,0], Label[:,1])
        
        # generate a zero tensor
        Zero = np.zeros((1))
        
        yield [PT, CT], [Seg_T, Seg_N, Label, Zero]
    
    
def gen_rtload(data_path, sample_names, batch_size=1, balance_class=False):

    while True:
        
        if balance_class == True:
            # manually balance class
            idxes = []
            num_pos = num_neg = 0
            df = pd.read_csv(data_path+'Clinical_info.csv')
            while num_pos<batch_size/2 or num_neg<batch_size/2:
                idx = np.random.randint(len(sample_names))
                idx_df = df.loc[df['Patient'] == bytes.decode(sample_names[idx][:-4])]
                idx_Event = idx_df['Event'].values[0]
                if idx_Event==0 and num_neg<batch_size/2:
                    idxes.append(idx)
                    num_neg = num_neg+1
                if idx_Event==1 and num_pos<batch_size/2:
                    idxes.append(idx)
                    num_pos = num_pos+1
        else:
            idxes = np.random.randint(len(sample_names), size=batch_size)
        
        # load the selected data
        npz_data = []
        for idx in idxes:
            X = load_volfile(data_path+bytes.decode(sample_names[idx]), np_var='all')
            npz_data.append(X)
            
        X_data = []
        for i in range(batch_size):
            X = npz_data[i]['PT']
            X = X[np.newaxis, np.newaxis, ...]
            X_data.append(X)
        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]
                           
        X_data = []
        for i in range(batch_size):
            X = npz_data[i]['CT']
            X = X[np.newaxis, np.newaxis, ...]
            X_data.append(X)
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data, 0))
        else:
            return_vals.append(X_data[0])
        
        X_data = []
        for i in range(batch_size):
            X = npz_data[i]['Seg_T']
            X = X[np.newaxis, np.newaxis, ...]
            X_data.append(X)
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data, 0))
        else:
            return_vals.append(X_data[0])
            
        X_data = []
        for i in range(batch_size):
            X = npz_data[i]['Seg_N']
            X = X[np.newaxis, np.newaxis, ...]
            X_data.append(X)
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data, 0))
        else:
            return_vals.append(X_data[0])
            
        X_data = []
        for i in range(batch_size):
            Time = npz_data[i]['Time']
            Event = npz_data[i]['Event']
            X = np.array([Time,Event])
            X = X[np.newaxis, ...]
            X_data.append(X)
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data, 0))
        else:
            return_vals.append(X_data[0])
            
        yield tuple(return_vals)

    
def load_by_name(data_path, sample_name):
    
    npz_data = load_volfile(data_path+bytes.decode(sample_name), np_var='all')
    
    X = npz_data['PT']
    X = X[np.newaxis, np.newaxis, ...]
    return_vals = [X]
    
    X = npz_data['CT']
    X = X[np.newaxis, np.newaxis, ...]
    return_vals.append(X)
    
    X = npz_data['Seg_T']
    return_vals.append(X)
    
    X = npz_data['Seg_N']
    return_vals.append(X)
    
    Time = npz_data['Time']
    Event = npz_data['Event']
    X = np.array([Time, Event])
    return_vals.append(X)
    
    return tuple(return_vals)


#--------------------------------------------------------------------------------------
# Util Functions

def load_volfile(datafile, np_var):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nibabel' not in sys.modules:
            try :
                import nibabel as nib  
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()
        
    else: # npz
        if np_var == 'all':
            X = X = np.load(datafile)
        else:
            X = np.load(datafile)[np_var]

    return X


def make_surv_array(time, event):
    '''
    Transforms censored survival data into vector format that can be used in Keras.
    Arguments
        time: Array of failure/censoring times.
        event: Array of censoring indicator. 1 if failed, 0 if censored.
        breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
    Returns
        surv_array: Dimensions with (number of samples, number of time intervals*2)
    '''
    
    breaks = np.array([0,300,600,900,1100,1300,1500,2100,2700,3500,6000])
    
    n_samples=time.shape[0]
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5*timegap
    
    surv_array = np.zeros((n_samples, n_intervals*2))
    for i in range(n_samples):
        if event[i] == 1:
            surv_array[i,0:n_intervals] = 1.0*(time[i]>=breaks[1:]) 
            if time[i]<breaks[-1]:
                surv_array[i,n_intervals+np.where(time[i]<breaks[1:])[0][0]]=1
        else: # event[i] == 0
            surv_array[i,0:n_intervals] = 1.0*(time[i]>=breaks_midpoint)
    
    return surv_array

#--------------------------------------------------------------------------------------
# Function for data argumentation
    
def Data_augmentation_seg(PT, CT, Seg_T, Seg_N):
    
    # define augmentation sequence
    aug_seq = iaa.Sequential([
        # translate/move them and rotate them.
        iaa.Affine(translate_percent={"x": [-0.1, 0.1], "y": [0, 0]},
                   scale={"x": (0.9, 1.1), "y": (1.0, 1.0)},
                   shear=(-10, 10),
                   rotate=(-10, 10)),
        iaa.CropToFixedSize(width=112, height=None)
        ],random_order=False)
    
    # pre-process data shape
    PT = PT[:,0,:,:,:]
    CT = CT[:,0,:,:,:]
    Seg_T = Seg_T[:,0,:,:,:]
    Seg_N = Seg_N[:,0,:,:,:]
    
    # flip/translate in x axls, rotate along z axls
    images = np.concatenate((PT,CT,Seg_T,Seg_N), -1)
    
    images_aug = np.array(aug_seq(images=images))
    
    PT = images_aug[..., 0:int(images_aug.shape[3]/4)]    
    CT = images_aug[..., int(images_aug.shape[3]/4):int(images_aug.shape[3]/4*2)]
    Seg_T = images_aug[..., int(images_aug.shape[3]/4*2):int(images_aug.shape[3]/4*3)]
    Seg_N = images_aug[..., int(images_aug.shape[3]/4*3):int(images_aug.shape[3])]
    
    # translate in z axls, rotate along y axls
    PT = np.transpose(PT,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    Seg_T = np.transpose(Seg_T,(0,3,1,2))
    Seg_N = np.transpose(Seg_N,(0,3,1,2))
    images = np.concatenate((PT,CT,Seg_T,Seg_N), -1)
    
    images_aug = np.array(aug_seq(images=images))
    
    PT = images_aug[..., 0:int(images_aug.shape[3]/4)]    
    CT = images_aug[..., int(images_aug.shape[3]/4):int(images_aug.shape[3]/4*2)]
    Seg_T = images_aug[..., int(images_aug.shape[3]/4*2):int(images_aug.shape[3]/4*3)]
    Seg_N = images_aug[..., int(images_aug.shape[3]/4*3):int(images_aug.shape[3])]
    
    # translate in y axls, rotate along x axls
    PT = np.transpose(PT,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    Seg_T = np.transpose(Seg_T,(0,3,1,2))
    Seg_N = np.transpose(Seg_N,(0,3,1,2))
    images = np.concatenate((PT,CT,Seg_T,Seg_N), -1)
    
    images_aug = np.array(aug_seq(images=images))
    
    PT = images_aug[..., 0:int(images_aug.shape[3]/4)]    
    CT = images_aug[..., int(images_aug.shape[3]/4):int(images_aug.shape[3]/4*2)]
    Seg_T = images_aug[..., int(images_aug.shape[3]/4*2):int(images_aug.shape[3]/4*3)]
    Seg_N = images_aug[..., int(images_aug.shape[3]/4*3):int(images_aug.shape[3])]
    
    # recover axls
    PT = np.transpose(PT,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    Seg_T = np.transpose(Seg_T,(0,3,1,2))
    Seg_N = np.transpose(Seg_N,(0,3,1,2))
    
    # reset Seg mask to 1/0
    for i in range(Seg_T.shape[0]):
        _, Seg_T[i] = cv2.threshold(Seg_T[i],0.2,1,cv2.THRESH_BINARY)
        _, Seg_N[i] = cv2.threshold(Seg_N[i],0.2,1,cv2.THRESH_BINARY)
    
    # post-process data shape
    PT = PT[..., np.newaxis].transpose((0,4,1,2,3))
    CT = CT[..., np.newaxis].transpose((0,4,1,2,3))
    Seg_T = Seg_T[..., np.newaxis].transpose((0,4,1,2,3))
    Seg_N = Seg_N[..., np.newaxis].transpose((0,4,1,2,3))
    
    return PT, CT, Seg_T, Seg_N
    

def Data_augmentation_surv(PT, CT):
    
    # define augmentation sequence
    aug_seq = iaa.Sequential([
        # translate/move them and rotate them.
        iaa.Affine(translate_percent={"x": [-0.1, 0.1], "y": [0, 0]},
                   scale={"x": (0.9, 1.1), "y": (1.0, 1.0)},
                   shear=(-10, 10),
                   rotate=(-10, 10)),
        iaa.CropToFixedSize(width=112, height=None)
        ],random_order=False)
    
    
    # pre-process data shape
    PT = PT[:,0,:,:,:]
    CT = CT[:,0,:,:,:]
    
    # flip/translate in x axls, rotate along z axls
    images = np.concatenate((PT,CT), -1)
    images_aug = np.array(aug_seq(images=images))
    PT = images_aug[..., 0:int(images_aug.shape[3]/2)]    
    CT = images_aug[..., int(images_aug.shape[3]/2):int(images_aug.shape[3])]
    
    # flip/translate in z axls, rotate along y axls
    PT = np.transpose(PT,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    
    images = np.concatenate((PT,CT), -1)
    images_aug = np.array(aug_seq(images=images))
    PT = images_aug[..., 0:int(images_aug.shape[3]/2)]    
    CT = images_aug[..., int(images_aug.shape[3]/2):int(images_aug.shape[3])]
    
    # flip/translate in y axls, rotate along x axls
    PT = np.transpose(PT,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    
    images = np.concatenate((PT,CT), -1)
    images_aug = np.array(aug_seq(images=images))
    PT = images_aug[..., 0:int(images_aug.shape[3]/2)]    
    CT = images_aug[..., int(images_aug.shape[3]/2):int(images_aug.shape[3])]
    
    # recover axls
    PT = np.transpose(PT,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    
    #post-process data shape
    PT = PT[..., np.newaxis].transpose((0,4,1,2,3))
    CT = CT[..., np.newaxis].transpose((0,4,1,2,3))
    
    return PT, CT
