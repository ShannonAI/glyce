import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-4])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch
from torch import nn
from torch.autograd import Variable


import numpy as np


from glyce.models.srl.data_utils import _PAD_,_UNK_,_ROOT_,_NUM_


USE_CUDA = torch.cuda.is_available() and True

def get_torch_variable_from_np(v):
    if USE_CUDA:
        return Variable(torch.from_numpy(v)).cuda()
    else:
        return Variable(torch.from_numpy(v))

def get_torch_variable_from_tensor(t):
    if USE_CUDA:
        return Variable(t).cuda()
    else:
        return Variable(t)

def get_data(v):
    if USE_CUDA:
        return v.data.cpu().numpy()
    else:
        return v.data.numpy()

def create_trees(sentence, deprel2idx):
    ids = [int(item[4]) for item in sentence]
    parents = [int(item[10]) for item in sentence]

    trees = dict()
    roots = dict()

    for i in range(len(ids)):
        tree = Tree(i, deprel2idx.get(sentence[i][11],deprel2idx[_UNK_]))
        trees[ids[i]] = tree
    
    for i in range(len(parents)):
        index = ids[i]
        parent = parents[i]
        if parent == 0:
            roots[i] = trees[index]
            continue
        trees[parent].add_child(trees[index])
    
    return trees,roots
