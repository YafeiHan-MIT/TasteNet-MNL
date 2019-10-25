#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 11:35:11 2019

@author: yafei
"""
import torch.nn.functional as F
import torch
from train import evaluate_epoch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def predictChoice(model, dataset):
    '''
    Predict final choice probability and label
    '''
    with torch.no_grad():
        prob_choice = F.softmax(model.forward(dataset.z, dataset.x), dim=1)
             
    return prob_choice, prob_choice.argmax(dim=1)

def summarizeDataset(model, ds):
    dl = DataLoader(ds, batch_size=model.args.batch_size, shuffle=False, num_workers=5)
    
    # nll
    nll = evaluate_epoch(model, dl)
    # predict choice 
    prob_choice, pred_choice = predictChoice(model, ds)
    
    # choice accuracy
    acc_choice = accuracy_score(ds.y, pred_choice)
    
    sm = {"nll": nll, "acc": acc_choice, "acc_true": ds.acc, "nll_true": ds.nll}
    return sm


def summarize(model, ds_train, ds_dev, ds_test):
    sm_train = summarizeDataset(model, ds_train)
    sm_dev = summarizeDataset(model, ds_dev)
    sm_test = summarizeDataset(model, ds_test)
    summary = {"train":sm_train, "dev": sm_dev, "test": sm_test}
    return summary

from prettytable import PrettyTable

def printSummary(summary, precision=3):
    x = PrettyTable()
    data_list = ["train", "dev", "test"]
    x.field_names = [""]+["acc_"+data for data in data_list] + ["nll_"+data for data in data_list]
    x.add_row(["model"] + [round(summary[data]["acc"],precision) for data in data_list] + [round(summary[data]["nll"],precision) for data in data_list])
    x.add_row(["true"] + [round(summary[data]["acc_true"], precision) for data in data_list] + [round(summary[data]["nll_true"], precision) for data in data_list])
    return x

def predictVOT(model,z):
    '''
    vot is negative
    '''
    return model.vot_module(z)

def predictVOTds(model, ds):
    vot_pred = predictVOT(model,ds.z)
    return {"vot_pred": vot_pred, "vot_true": ds.vots}

def computeA(params):
    A1 = torch.Tensor([params['a1'],params['a2'],params['a3']]).reshape(-1,1) # D by 1
    A2 = torch.zeros(3,3)
    A2[0,1] = params['a12']
    A2[0,2] = params['a13']
    A2[1,2] = params['a23']
    return A1,A2

def valueOfTime(params, z):
    '''
    Compute value of time for N persons given person attributes z (N,D)
    Input:
        params: dict of parameters 
        z: person input (N,D)
    Return:
        vot: (N,1)   note: vot is the - coef
    '''
    N,D = z.size()
    A1, A2 = computeA(params)
    vots = params['a0'] + torch.matmul(z,A1) + torch.diag(torch.matmul(torch.matmul(z,A2), z.transpose(0,1))).reshape(N,1)
    return vots #negative 
    
## Error measures
def RMSE(pred_vots, true_vots):
    '''
    pred_vots: dictionary {key:tensor value of time}
    '''
    sum_error = 0.0
    count=0
    for i in range(len(pred_vots)):
        sum_error +=((pred_vots[i]-true_vots[i])**2).sum()
        count += len(pred_vots[i])
    rmse = (sum_error/count)**0.5
    return rmse

def ABSE(pred_vots, true_vots):
    sum_error = 0.0
    count=0
    for i in range(len(pred_vots)):
        sum_error+=(torch.abs(pred_vots[i]-true_vots[i])).sum()
        count += len(pred_vots[i])
    return sum_error/count

def RE(pred_vots, true_vots):
    '''
    Relative error (pred-true)/true
    '''
    sum_error = 0.0
    count=0
    for i in range(len(pred_vots)):
        sum_error += (torch.abs(pred_vots[i]-true_vots[i])/torch.abs(true_vots[i])).sum()
        count += len(pred_vots[i])
    return sum_error/count

def printError(rmse, mabse, re):
    s = ""
    s += "rmse:"+ str(rmse) +"\n"
    s += "mean absolute error:" + str(mabse) +"\n"
    s += "percentage error:" + str(re) +"\n"
    return s


def plotVOT(pred_vots, true_vots, x_values, legend, result_path):
    fg, (ax1, ax2) = plt.subplots(1,2,sharey=True, figsize=(14,6))
    for vot in pred_vots:
        ax1.plot(x_values,vot.flatten().numpy())
    for vot in true_vots:
        ax2.plot(x_values,vot.flatten().numpy())
    ax1.set_title("Predicted")
    ax2.set_title("True")
    ax1.legend(legend)
    ax2.legend(legend)
    ax1.set_xlabel("income ($ per hour)", fontsize=12)
    ax1.set_ylabel("value of time ($ per hour)", fontsize=12)
    ax2.set_xlabel("income ($ per hour)", fontsize=12)
    ax2.set_ylabel("value of time ($ per hour)", fontsize=12)
    
    fg.savefig(result_path + "/" + "VOT_vs_inc.png", dpi=250)
    plt.close()

def plotLoss(loss_train_list, loss_dev_list, fig_path):
    plt.plot(loss_train_list)
    plt.plot(loss_dev_list)
    plt.legend(["loss_train", "loss_dev"])
    plt.xlabel("number of epochs")
    plt.ylabel("negative loglikelihood")
    plt.savefig(fig_path + "/loss_train_dev.png", dpi=300)