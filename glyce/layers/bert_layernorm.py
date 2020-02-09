#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Contact: xiaoya_li@shannonai.com 
# Last update: 2019.04.04 
# First create: 2019.03.29 
# Description:
# bert_layernorm.py 



import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch 
from torch import nn 
from torch.nn import CrossEntropyLoss 


import copy 
import json 
import math 
import tarfile 
import shutil 
import logging 
import tempfile 




class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        # construct a layernorm module in the TF style
        # epsilon inside the square are not 
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps 


    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias 