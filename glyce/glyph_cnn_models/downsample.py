# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: shufflenet
@time: 2019/1/7 19:36

这一行开始写关于本文件的说明与解释
"""


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


from collections import OrderedDict


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def channel_shuffle(x, groups):
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


class DownsampleUnit(nn.Module):
    def __init__(self, inplanes, c_tag=0.5, activation=nn.ReLU, groups=2):
        super(DownsampleUnit, self).__init__()

        self.conv1r = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn1r = nn.BatchNorm2d(inplanes)
        self.conv2r = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False, groups=inplanes)
        self.bn2r = nn.BatchNorm2d(inplanes)
        self.conv3r = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn3r = nn.BatchNorm2d(inplanes)

        self.conv1l = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False, groups=inplanes)
        self.bn1l = nn.BatchNorm2d(inplanes)
        self.conv2l = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn2l = nn.BatchNorm2d(inplanes)
        self.activation = activation(inplace=False)

        self.groups = groups
        self.inplanes = inplanes
        # self.init_params()

    def forward(self, x):
        out_r = self.conv1r(x)
        # out_r = self.bn1r(out_r)
        out_r = self.activation(out_r)

        out_r = self.conv2r(out_r)
        # out_r = self.bn2r(out_r)

        out_r = self.conv3r(out_r)
        # out_r = self.bn3r(out_r)
        out_r = self.activation(out_r)

        out_l = self.conv1l(x)
        # out_l = self.bn1l(out_l)

        out_l = self.conv2l(out_l)
        # out_l = self.bn2l(out_l)
        out_l = self.activation(out_l)

        return channel_shuffle(torch.cat((out_r, out_l), 1), self.groups)

