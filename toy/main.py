#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Interpretable Neural Networks for Discrete Choice Modeling: Flexible Utility Specification & Fast Estimation

@author: yafei
"""

import argparse
import torch
from torch.utils.data import DataLoader
from data_utils import ChoiceDataset
from models import ChoiceFlexVOT
from train import train
from evaluation import summarize, printSummary, predictVOT, printError, RMSE, ABSE, RE, plotVOT, plotLoss
from simulate import *
import copy
import pickle

# Specify input parameters 
parser = argparse.ArgumentParser(description='TasteNet (Toy Example)')

# Training-related 
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_epochs', type=int, default=100, 
                    help='number of epochs to train (default: 100)')
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--weight_decay", type=float, required=True, default=0.000)
parser.add_argument("--nll_tol", type=float, default=0.0001, help="tolerance for nll convergence")

## NN structure
parser.add_argument("--layer_sizes", nargs='+', required=True, default=[3,1])
parser.add_argument("--act_func", type=str, required=True, default=None)
parser.add_argument("--transform", type=str, default=None, help="what kind of transform on the taste: exp, relu or empty string")

## For MNL specification only
parser.add_argument("--if_z01", type=bool, default=False, help="whether to add interaction z0*z1")
parser.add_argument("--if_zall", type=bool, default=False, help="whether to add all interactions among z")

parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=None, 
                    help='random seed (default: None)')

# input data directory
parser.add_argument("--data_path", type=str, default="../toy_data")
parser.add_argument("--N_train", type=str, default='10k')
parser.add_argument("--result_root", type=str, required=True)

# model run number
parser.add_argument("--model_no", type=int, required=True, default=0)

#=======Parse arguments=============
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

args.data_train = "train_"+args.N_train + ".pkl"
args.data_dev = "dev_"+args.N_train + ".pkl"
args.data_test = "test_"+args.N_train + ".pkl"

def str_of_arg(arg):
    if arg!="":
        return "_" + arg
    else:
        return ""
    
# input data name 
str_act = str_of_arg(args.act_func)
str_transform = str_of_arg(args.transform)

str_H = ""
if len(args.layer_sizes)>2:
    str_H = "_H"+"_".join(str(e) for e in args.layer_sizes[1:-1])

str_wd = ""
if args.weight_decay > 0:
    str_wd = "_" + str(args.weight_decay)
    
str_suffix = ""
if args.if_z01:
    str_suffix = "_z01"
if args.if_zall:
    str_suffix = "_zall"
args.scenario = "model" + str_H + str_act + str_transform + str_wd + str_suffix + "_no" + str(args.model_no)

args.result_path = args.result_root + "/" + args.scenario
args.fig_path = args.result_path + "/figures"

#====== make output directory =====
import os
if not os.path.exists(args.result_root):
    os.mkdir(args.result_root)
if not os.path.exists(args.result_path):
    os.mkdir(args.result_path)
if not os.path.exists(args.fig_path):
    os.mkdir(args.fig_path)
    
if args.seed != None: 
    torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

#======Prepare input data ========
ds_train = ChoiceDataset(args.data_path, args.data_train, args.if_z01, args.if_zall)
ds_dev = ChoiceDataset(args.data_path, args.data_dev, args.if_z01, args.if_zall)
ds_test = ChoiceDataset(args.data_path, args.data_test, args.if_z01, args.if_zall)

data_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=5)
data_dev = DataLoader(ds_dev, batch_size=args.batch_size, shuffle=False, num_workers=5)
data_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=5)

args.K, args.J = ds_train[0]["x"].size()
args.layer_sizes = [int(e) for e in args.layer_sizes]

#======Get model==================
print args
model = ChoiceFlexVOT(args)
print model

model.args.init_params = model.getParams()
pickle.dump(model.args.init_params, open(args.result_path + "/params_init.pkl", "wb"))


loss_train_list, loss_dev_list, best_model = train(model, data_train, data_dev, args)


#======Plot loss=====================
plotLoss(loss_train_list, loss_dev_list, args.fig_path)

#======Summary=====================
summary = summarize(best_model, ds_train, ds_dev, ds_test)
x = printSummary(summary, precision=3)
print x

## save sumamary
f = open(args.result_path + "/" + "summary_result.txt", "wb")
f.write(str(x))  
f.close()
#
# save model args 
f = open(args.result_path + "/" + "summary_args.txt", "wb")
f.write(str(args))  
f.close()
#
## save model string
f = open(args.result_path + "/" + "summary_modelstr.txt", "wb")
f.write(str(best_model))  
f.close()

#======Toy data VOT=====================
allds = {"train":ds_train, "dev":ds_dev, "test":ds_test}
pred_vots = [predictVOT(best_model,allds[name].z) for name in ["train", "dev", "test"]]
true_vots = [allds[name].vots for name in ["train", "dev", "test"]]

rmse = RMSE(pred_vots, true_vots).item()
mabse = ABSE(pred_vots, true_vots).item()
re = RE(pred_vots, true_vots).item()

s = ""
s += "rmse:"+ str(rmse) +"\n"
s += "mean absolute error:" + str(mabse) +"\n"
s += "percentage error:" + str(re) +"\n"

f = open(args.result_path + "/" + "vot_error.txt", "wb")
f.write(str(s))  
f.close()

pickle.dump(pred_vots, open(args.result_path + "/" + "pred_vots.pkl", "wb"))

#======= VOT on simulated z ==========
if args.if_z01:
    input_z = copy.deepcopy(dic_z_z01)
elif  args.if_zall:
    input_z = copy.deepcopy(dic_z_zall)
else:
    input_z = copy.deepcopy(dic_z)

#======= Error of VOT on simulated z ==========
sim_pred_vots, sim_true_vots, rmse, mabse, re = error_of_vot(best_model, dic_z, input_z, params)
f = open(args.result_path + "/" + "sim_vot_error.txt", "wb")
f.write(str(printError(rmse, mabse, re)))  
f.close()

#======= Plot VOT on simulated z ==========
plotVOT(sim_pred_vots, sim_true_vots, (inc*60).numpy(), dic_z.keys(), args.result_path)

#======= Model parameters ==========
params = best_model.getParams()
pickle.dump(params, open(args.result_path + "/params.pkl", "wb"))

