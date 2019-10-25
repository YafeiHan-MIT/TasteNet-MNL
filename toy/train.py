#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 11:24:46 2019

@author: yafei
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
import pandas as pd
import numpy as np
import copy 


def train(model, data_train, data_dev, args):
    '''
    Run num_epochs training epochs, and evaluate on data_dev at the end of each epoch
    '''
   
    optimizer =  optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    for param in model.parameters():
        print param
    best_model = None
    best_dev_loss = 10000 # BIG number 
    count_no_chg = 0  # track dev_loss if there is change  
    loss_train_list, loss_dev_list = [],[]
    
    for epoch in range(1,args.num_epochs+1):
        train_epoch(model, data_train, optimizer)
        
        loss_train = evaluate_epoch(model, data_train)
        loss_dev = evaluate_epoch(model, data_dev)
        loss_train_list.append(loss_train)
        loss_dev_list.append(loss_dev)
        
        print('====> Epoch: {} Train loss: {:.4f}'.format(epoch, loss_train))
        print('====> Epoch: {} Dev loss: {:.4f}'.format(epoch, loss_dev))
        
#        if epoch%5==0:
#            for param in model.parameters():
#                print param
        if loss_dev < best_dev_loss:
            best_dev_loss = loss_dev
            best_model = copy.deepcopy(model)
            count_no_chg = 0
        else:
            count_no_chg += 1
        
        if count_no_chg >=10:
            break
        
    pickle.dump(best_model, open(args.result_path + "/best_model.pkl", "wb"))
    pd.DataFrame(np.array([loss_train_list,loss_dev_list]).T,\
                 columns = ["train_loss", "dev_loss"]).to_csv(args.result_path + "/train_dev_loss.csv", index=True)
    
    return loss_train_list, loss_dev_list, best_model

def train_epoch(model, data_train, optimizer):
    '''
    Run 1 forward pass through the data 
    Parameters:
        data_train: training data DataLoader
        optimizer: 
    '''
    model.train()
    total_loss = 0.0
    batches= 0

    # iterate over batches
    for batch_idx, data in enumerate(data_train):
        # forward pass
        loss = F.cross_entropy(model.forward(data["z"],data["x"]), data["y"]) #input of cross_entropy is before taking log_softmax! batch average loss
        # back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record loss, batch size 
        total_loss += loss.item()
        batches += 1 
    return total_loss / batches 

def evaluate_epoch(model, data_loader):
    '''
    Loss over data 
    
    Parameters:
        model: 
        data_loader: data to evaluate loss (DataLoader)
    '''
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            loss = F.cross_entropy(model.forward(data["z"],data["x"]), data["y"]) #
            total_loss += loss.item()
            batches += 1
    return total_loss / batches