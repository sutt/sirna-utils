import os, sys, copy
import numpy as np 

import cv2
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from .imgplots import plot_two_imgs_by_channel
from .dataloader import DataLoader

# PROBLEM - dataloader needs to be instatiated, does this work?
dc = DataLoader()

def negcontrol_dif_sites(experiment=None, plate=None, b_ret=False):
    ''' return - list len-2, each inner list contains 6
        input  - none required; will do a random choice
                 [optional], plate (int), experiment (str)

        Same siRNA at different: sites 
                   at the same:  experiment, plate, well

        -> should be the closest matches of any imgs

        TODO - [ ] plate, experiment inputs (non-randomized)
    '''
    negc_df = dc.get_all_sirna_df(sirna=sirna)
    
    dc.get_neg_controls_df()
    
    rand_sirna = np.random.choice(sirna_df['id_code'].tolist(), 1)[0]

    l_idcs = dc.ida_to_idcs(dc.train_id_to_ida(rand_sirna))
    
    l_l_imgs = dc.load_img_from_l_idc(l_idcs)

    if b_ret:
        return l_l_imgs
    
    plot_two_imgs_by_channel(l_l_imgs, b_crop=False)

def same_sirna_dif_sites(sirna, plate=1, b_ret=False):
    ''' return - list len-2, each inner list contains 6
        input  - (int) sirna code-num
                 [optional], plate (int), experiment (str)

        Same siRNA at different: sites 
                   at the same:  experiment, plate, well

        -> should be the closest matches of any imgs

        TODO - [ ] plate, experiment inputs (non-randomized)
    '''
    sirna_df = dc.get_all_sirna_df(sirna=sirna)
    
    rand_sirna = np.random.choice(sirna_df['id_code'].tolist(), 1)[0]

    l_idcs = dc.ida_to_idcs(dc.train_id_to_ida(rand_sirna))
    
    l_l_imgs = dc.load_img_from_l_idc(l_idcs)

    if b_ret:
        return l_l_imgs
    
    plot_two_imgs_by_channel(l_l_imgs, b_crop=False)

def same_sirna_dif_plates(sirna, plates=(1,2), b_ret=False):
    ''' return - list len-2, each inner list contains 6
        input  - 

        Same siRNA at different: sites 
                   at the same:  experiment, plate, well

        -> should be the closest matches of any imgs
    '''
    sirna_df = dc.get_all_sirna_df(sirna=sirna)
    sirna = np.random.choice(sirna_df['id_code'].tolist(), 1)[0]

    l_idcs = dc.ida_to_idcs(dc.train_id_to_ida(sirna))
    l_l_imgs = dc.load_img_from_l_idc(l_idcs)

    if b_ret:
        return l_l_imgs
    
    plot_two_imgs_by_channel(l_l_imgs, b_crop=False)