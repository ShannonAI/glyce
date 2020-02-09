# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: mask_cross_entropy
@time: 2019/1/23 17:43

    这一行开始写关于本文件的说明与解释
"""

import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch
from torch import nn



class MaskCrossEntropy(nn.Module):
    def __init__(self, mask_ids=None):
        super(MaskCrossEntropy, self).__init__()
        self.mask_ids = mask_ids or []

    def forward(self, logits, golden):
        mask = torch.zeros_like(golden).byte().view(-1)
        for mask_id in self.mask_ids:
            mask = mask | (golden == mask_id).byte().view(-1)
        n = mask.size(0)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), golden.view(-1), reduce=False)
        loss = loss.masked_fill_(mask, 0.0)   # ignore padding char loss
        loss = loss.sum() / (n - mask.float().sum())
        return loss

    def __repr__(self):
        return F'MaskCrossEntropy(mask_ids={self.mask_ids})'