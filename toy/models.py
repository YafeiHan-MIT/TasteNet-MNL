#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:04:31 2019

@author: yafei
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class MNL(nn.Module):
    '''
    MNL with same parameters for all alternatives 
    (neural network: CNN (generic parameters across alternatives) without hidden layer)
    ''' 
    def __init__(self, K, args):
        super(MNL, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=K, out_channels=1, kernel_size=1, bias=False)
        self.args = args
    
    def forward(self, x):
        '''
        x.size = (N,K,J)   batch size, in channel, sequence length (J)
        '''
        x = self.conv1(x)  # input: N,K,J   output: N,1,J
        return x.flatten(start_dim=1,end_dim=2)  # (N,J)


class ChoiceFlexVOT(nn.Module):
    def __init__(self, args):
        super(ChoiceFlexVOT, self).__init__()
        self.vot_module = VOTMLP(args.layer_sizes, args) # vot is in negative value 
        self.util_module = Utility(args)
        self.args = args
    
    def forward(self, z, x):
        vot = self.vot_module(z)
        v = self.util_module(x,vot)
        return v
    
    def getCoefBias(self):
        '''
        get coef and bias of the TasteParams of the model 
        '''
        count = 0
        bias = []
        coef = []
        for params in self.parameters():
            if count % 2==0:
                coef.append(params)
            else:
                bias.append(params)
            count += 1
        return coef, bias
    
    def getParams(self):
        params_list = []
        for params in self.parameters():
            params_list.append(params)
        return copy.deepcopy(params_list)
        
class Utility(nn.Module):
    def __init__(self, args):
        super(Utility, self).__init__()
        self.linear = nn.Linear(1,1,bias=False) # asc for 1 alternative
        self.args = args
        
    def forward(self, x, vot):
        '''
        x: (N,K,J): not including intercept.  K=2: (cost, time), J=2
        vot: (N,1) <0
        '''
        N,_,_ = x.size()
        
        asc0 = torch.zeros(N,1) #asc0 fixed to 0
        asc1 = self.linear(torch.ones(N,1)) # learn asc1 by linear (1-1) here 
        
        b_vot = torch.cat([vot,vot], dim=1)
        b_cost = -torch.ones(N,2)
        
        v = x[:,0,:]*b_cost + x[:,1,:]*b_vot + torch.cat((asc0,asc1),dim=1)
        return v # (N,J)
    
def get_act(nl_func):
    if nl_func=="tanh":
        return nn.Tanh()
    elif nl_func == "relu":
        return nn.ReLU()
    elif nl_func == "sigmoid":
        return nn.Sigmoid()
    else:
        return None
    
class VOTMLP(nn.Module):
    '''
    Network for value of time. 
    Structure: MLP
    '''
    def __init__(self, layer_sizes, args):
        super(VOTMLP, self).__init__()
        self.seq = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]) ):
            self.seq.add_module(name="L%i"%(i+1), module=nn.Linear(in_size, out_size, bias=True))
            if i<len(layer_sizes)-2:
                self.seq.add_module(name="A%i"%(i+1), module=get_act(args.act_func))
        self.args = args
        
    def forward(self,z):
        '''
        Parameters:
            z: (N,D) # batch size, input dimension
        Returns:
            V: (N,1) # value of time 
        '''
        N,D = z.size()
        if self.args.transform == "exp":
            return -torch.exp(-1*self.seq(z))# (N,1)
        elif self.args.transform == "relu":
            return -F.relu(-self.seq(z))
        else: # no transform
            return self.seq(z)
