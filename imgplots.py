import os, sys, copy
import numpy as np 

import cv2
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def plot_two_imgs_by_channel(l_l_imgs,
                             b_crop=True,
                             crop_dims=(40,100,40,100),
                             figsize=(10,30),
                            ):
    ''' 6 by 2 plot of two images with all channels in descending order'''
    
    fig, ax = plt.subplots(nrows=6, ncols=2, figsize=figsize)

    for _ibatch in range(2):
        for _i in range(6):

            _img = l_l_imgs[_ibatch][_i]
            
            if b_crop:
                _img = _img[crop_dims[0]:crop_dims[1], crop_dims[2]:crop_dims[3]]

            ax[_i, _ibatch].imshow(_img)
            ax[_i, _ibatch].axis('off')
            ax[_i, _ibatch].set_title(str(_i))

    print('columns are different batches: ' + 'TODO - implement col names')
    if b_crop: print('cropping is on; cropped at ' + str(crop_dims))
    plt.tight_layout()