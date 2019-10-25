#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:13:20 2019

@author: yafei
"""

import pickle
from torch.utils.data import Dataset
import torch

class ChoiceDataset(Dataset):
    def __init__(self, data_path, data_file, if_z01 = False, if_zall = False):
        """
        Parameters:
            data_file (string): name of data pickle file
        """
        data = pickle.load(open(data_path + "/" + data_file, "rb"))
        self.x = torch.Tensor(data["x"][:,1:,:]) # N,K,J alter attributes, excl intercept
        self.z = torch.Tensor(data['z']) # N,D socio-demo
        if if_z01: # only add z0*z1
            self.z = torch.cat([self.z, (self.z[:,0]*self.z[:,1]).view(-1,1)], dim=1)
        if if_zall: #add all interactions between z (true model)
            self.z = torch.cat([self.z, (self.z[:,0]*self.z[:,1]).view(-1,1), (self.z[:,0]*self.z[:,2]).view(-1,1), (self.z[:,1]*self.z[:,2]).view(-1,1)], dim=1)
            
        self.y = torch.LongTensor(data["y"]) # N
        self.vots = torch.Tensor(data["vots"]) #vots
        self.acc = data["acc"]
        self.nll = data['nll']
        self.params = data['params']
        self.N, self.K, self.J = self.x.size()
        _, self.D = self.z.size()
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        '''
        Get the sample given its idx in the list 
        '''
        return {"x": self.x[idx], "y": self.y[idx], "z":self.z[idx]}


####Testing
#data_path = '../toy_data'
#data_file = 'train_10k.pkl'
#ds_train = ChoiceDataset(data_path, data_file, True)
#ds_train[0]
