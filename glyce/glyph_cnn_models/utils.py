# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: utils
@time: 2019/1/7 19:25

这一行开始写关于本文件的说明与解释
"""


import os 
import sys 

root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch


def count_params(conv):
    """计算模型参数量
    conv(torch.nn.Module)
    """
    return sum(p.numel() for p in conv.parameters())


def channel_shuffle(x, groups):
    """channel shuffle，为了缓解group conv造成的同源问题
    groups(int): 组数
    """
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % groups == 0)
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x