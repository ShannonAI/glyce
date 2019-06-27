# encoding: utf-8
"""
@author: wuwei
@contact: wu.wei@pku.edu.cn

@version: 1.0
@file: cnn_for_fonts.py
@time: 19-1-2 上午11:07

用CNN将字体的灰度图卷积成特征向量
"""


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn import init 


import math 


from glyce.glyph_cnn_models.utils import channel_shuffle
from glyce.glyph_cnn_models.downsample import DownsampleUnit 
from glyce.glyph_cnn_models.self_attention import MultiHeadSelfAttention




def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))



class GlyphGroupCNN(nn.Module):     #55000
    def __init__(self, cnn_type='simple', kernel_size=5, font_channels=1, shuffle=False, ntokens=4000,
                 num_features=8*12*4, final_width=2, cnn_drop=0.5, groups=16):
        super(GlyphGroupCNN, self).__init__()
        self.aux_logits=False
        self.cnn_type = cnn_type
        output_channels = num_features
        self.conv1 = nn.Conv2d(font_channels, output_channels, kernel_size)
        midchannels = output_channels//4
        self.mid_groups = max(groups//2, 1)
        self.downsample = nn.Conv2d(output_channels, midchannels, kernel_size=1, groups=self.mid_groups)   #//2是因为参数量主要集中在下一层
        self.max_pool = nn.AdaptiveMaxPool2d((final_width, final_width))
        self.num_features = num_features
        self.reweight_conv = nn.Conv2d(midchannels, output_channels, kernel_size=final_width, groups=groups)
        self.output_channels=output_channels
        self.shuffle = shuffle
        self.dropout = nn.Dropout(cnn_drop)
        self.init_weights()

    def forward(self, x):
        # x = self.base_conv(x)
        x = F.relu(self.conv1(x) ) # [(seq_len*batchsize, Co, h, w), ...]*len(Ks)
        x = self.max_pool(x)  # n, c, 2, 2
        x = self.downsample(x)
        if self.shuffle:
            x = channel_shuffle(x, groups=2)
        x = F.relu(self.reweight_conv(x))
        if self.shuffle:
            x = channel_shuffle(x, groups=2)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x  # (seq_len*batchsize, nfeats)

    def init_weights(self):
        initrange = 0.1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.uniform_(-initrange, initrange)
                # init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(mean=1, std=0.001)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(std=0.001)
                if m.bias is not None:
                    m.bias.data.zero_()




if __name__ == '__main__':
    from glyph_cnn_models.utils import count_params, make_dot
    from tensorboardX import SummaryWriter
    conv = Yuxian8(num_features=1024, groups=16, font_channels=8)
    print('No. Parameters', count_params(conv))
    # print(conv)
    x = torch.rand([233, 8, 16, 16])
    y,_, _ = conv(x)
    # print(y.shape)
    # inputs = torch.randn(1, 3, 224, 224)
    print(y.shape)

    writer = SummaryWriter()
    writer.add_graph(conv, y)

    writer.close()
