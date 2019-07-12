import os, sys, copy
import numpy as np 
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import cv2
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

sys.path.append('../')
from libs.rxrx1_utils.rxrx.io import (
                                 load_images_as_tensor
                                ,convert_tensor_to_rgb
                                )

'''
    Wrapper on rxrx-utils functions
'''

# load as list of fn
def img_fns_to_rgb(list_fns):
    ''' return - rgb img (512,512,3) ndarray
        input  - input list of path_fns to the img's channels
    '''
    t_imgs = load_images_as_tensor(list_fns)
    rgb_t_img = convert_tensor_to_rgb(t_imgs, )
    return rgb_t_img

