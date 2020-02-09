# encoding: utf-8
"""
@author: wuwei 
@contact: wu.wei@pku.edu.cn

@version: 1.0
@license: Apache Licence
@file: random_erasing.py
@time: 19-1-11 下午5:51

"""

import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import math
import random


class RandomErasing(object):
    """
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area erase的最大面积
    sh: max erasing area erase的最小面积
    r1: min aspect ratio 长宽比
    mean: erasing value erase所填的值
    -------------------------------------------------------------------------------------
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, random_num=False):
        self.probability = probability
        self.random = random_num
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        # (bs, num_channels, hight, width) => (bs * num_channels, hight, width) => (bs, num_channels, hight, width)
        reshaped_img = img.reshape((-1, img.size()[2], img.size()[3]))
        if random.uniform(0, 1) > self.probability:
            return img

        area = reshaped_img.size()[1] * reshaped_img.size()[2]
        for attempt in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < reshaped_img.size()[2] and h < reshaped_img.size()[1]:
                x1 = random.randint(0, reshaped_img.size()[1] - h)
                y1 = random.randint(0, reshaped_img.size()[2] - w)
                # for i in range(reshaped_img.size()[0]):
                mean = random.randint(0, 255) if self.random else 128
                reshaped_img[:, x1:x1 + h, y1:y1 + w] = mean
                return reshaped_img.reshape(img.size())
        return img
