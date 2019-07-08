import os, sys, copy
import numpy as np 
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


class DataLoader:
    
    '''
        Load Dataset utilities
        
        NOT IMPLEMENTED----
        returns (with prepend_msg=True)
            tuple:
                elem-0: msg on img creation [no]
                elem-1: list of imgs channel
        -------------------
                    
        Terminology for id<x>-dicts:
        
            ida - exp, plate, well
            idb - exp, plate, well, site(2)
            idc - exp, plate, well, site(2), channel(6)
            
            (indexes for site and channel start at 1)
            
    '''
    
    def __init__(self, 
                 csv_data_dir = 'data/csvdata/',
                 img_data_dir = 'data/',
                 prepend_msg=False,
                 ):
        
        self.IMG_DATA_DIR = img_data_dir
        self.CSV_DATA_DIR = csv_data_dir
        
        self.prepend_msg = prepend_msg
        
        self.train_control = pd.read_csv(self.CSV_DATA_DIR + 'train_controls.csv')
        self.test_control = pd.read_csv(self.CSV_DATA_DIR + 'test_controls.csv')
        self.train = pd.read_csv(self.CSV_DATA_DIR + 'train.csv')
        self.test = pd.read_csv(self.CSV_DATA_DIR + 'test.csv')
        self.pixel_stats = pd.read_csv(self.CSV_DATA_DIR + 'pixel_stats.csv')
        self.sample_submission = pd.read_csv(self.CSV_DATA_DIR + 'sample_submission.csv')
        
        
        self.exp_sirna = self.train['sirna']
        
    def msg(self, on=False):
        '''alter the msg on/off'''
        self.prepend_msg = on
        
    @staticmethod
    def random(x, k=1):
        return np.random.choice(x,k=k)
    
    def row_to_ida(self,row):
        '''return dict with exp, plate, wellwith 
           input (int) row in train 
           [ ] TODO - add train_controls here
       '''
        s_experiment = self.train.loc[row, 'experiment']
        s_plate = self.train.loc[row, 'plate']
        s_well = self.train.loc[row, 'well']
        return {
            'experiment': s_experiment,
            'plate': s_plate,
            'well': s_well,
        }
    
    @staticmethod
    def train_id_to_ida(train_id):
        ''' using train table id_code, pase into 3 components and 
            return as ida
        '''
        ida = {}
        for k,v in zip(
                 ('experiment', 'plate', 'well'),
                 train_id.split('_')
                      ):
            ida[k] = v
        return ida
    
    @staticmethod
    def ida_to_idb(ida, site):
        '''add site to id-dict and return'''
        ida['site'] = site
        return ida
    
    @staticmethod
    def ida_to_idbs(ida):
        ''' return: list of dict - idb (exp, plate, well, channel)
            input:  dict - ida (exp, plate, well)
        '''
        idbs = []
        for site in range(1,3):
            _idb = ida.copy()
            _idb['site'] = site
            idbs.append(_idb)
        return idbs
    
    @staticmethod
    def idb_to_idc(idb, channel):
        '''add channel to id-dict and return'''
        idb['channel'] = channel
        return idb
    
    @staticmethod
    def idb_to_idcs(idb):
        ''' return: list of list of dict - idc
            input:  dict - idb (exp, plate, well, site)
        '''
        idcs = []
        for channel in range(1,7):
            _idc = idb.copy()
            _idc['channel'] = channel
            idcs.append(_idc)
        return idcs
    
    @staticmethod
    def ida_to_idc(ida, site, channel):
        '''add specific channel and site to id-dict and return'''
        ida['site'] = channel
        ida['channel'] = channel
        return ida
    
    @classmethod
    def ida_to_idcs(cls, ida):
        ''' return list of list of dict of idc
            input  dict of ida
        '''
        idbs = cls.ida_to_idbs(ida)
        return [cls.idb_to_idcs(e) for e in idbs]
    
    
    def idc_to_fn(self, experiment, plate, well, site, channel):
        '''return: str - the img-fn
            input: idc as unpacked-dict
        '''
        return os.path.join( 
                self.IMG_DATA_DIR,
                'train',
                 experiment,
                 ('Plate' + str(plate)),
                 (  well + 
                   '_s' + str(site) + 
                   '_w' + str(channel) 
                    + '.png'
                 )
        )
        
    @staticmethod
    def load_img(fn, npix=512):
        '''return img-obj from fn (via cv2)'''
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        if npix != 512:
            img = cv2.resize(img, (npix, npix))
        return img
    
    @classmethod
    def load_imgs(cls, fns, npix=512):
        '''return list of img-objs from list of fns (via cv2)'''
        return [cls.load_img(fn, npix=npix) for fn in fns]
        
    
    def get_two_sites(self, sirna=None, experiment=None):
        '''return the two sites with the same:
            experiment, well, (sirna / well)
        '''
        
        try:
            row = list(
                        (self.train['sirna'] == sirna).mul(
                            self.train['experiment'] == experiment)
                        ).index(True)
            
            assert isinstance(row, int)
            assert row >= 0
        
        except Exception as e:
            print(f'failed to find row for exp {experiment}, sirna: {sirna}')
            raise e
        
        ida = self.row_to_ida(row)
        
        args = self.ida_to_idcs(ida)
        
        list_list_imgs = [self.load_imgs(
                              [self.idc_to_fn(**e) for e in arg]
                                        ) 
                          for arg in args]
        
        if self.prepend_msg:
            return (args, list_list_imgs)
        
        return list_list_imgs

    def load_img_from_idc(self, idc):
        print('testing2 ftp')

    def hello_world(self):
        print('the real test')
    
    def get_all_sirna_df(self, sirna=None):
        '''return the train rows with a certain sirna'''
        if sirna is None: 
            pass
        return self.train[self.train['sirna'] == sirna]
    
    def get_negative_control(self,x):
        pass
        
