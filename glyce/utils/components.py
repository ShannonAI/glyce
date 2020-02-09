# encoding: utf-8
"""
@author: wuwei 
@contact: wu.wei@pku.edu.cn

@version: 1.0
@license: Apache Licence
@file: components.py
@time: 19-1-5 上午11:37

使用text_cnn和text_rnn将每个token的拼音字符串或者五笔编码字符串压缩成一个向量
"""


import os
import sys 

root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch
import torch.nn as nn

import json 
from pypinyin import pinyin
from pywubi import wubi


data_path = os.path.join(root_path, "glyce", "fonts")
with open(os.path.join(data_path, 'dictionary.json')) as fo:
    dict_word = json.load(fo)
with open(os.path.join(data_path, 'pinyin_vocab.json')) as fo:
    dict_pinyin = json.load(fo)
with open(os.path.join(data_path, 'wubi_vocab.json')) as fo:
    dict_wubi = json.load(fo)


batch_size = 128



class SubCharComponent(nn.Module):

    def __init__(self, encoding_type, composing_func, embedding_size, hidden_size, num_layers=1):
        super(SubCharComponent, self).__init__()
        self.encoding_type = encoding_type  # 拼音，五笔
        self.composing_func = composing_func  # 构造函数：lstm, cnn, avg, max
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if self.composing_func == 'LSTM':
            self.composing = nn.LSTM(input_size=embedding_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     bidirectional=True)
        elif self.composing_func == 'GRU':
            self.composing = nn.GRU(input_size=embedding_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    bidirectional=True)
        if self.encoding_type == 'wubi':
            self.embedding = nn.Embedding(len(dict_wubi['idx2char']), embedding_size)
        elif self.encoding_type == 'pinyin':
            self.embedding = nn.Embedding(len(dict_pinyin['idx2char']), embedding_size)

    def forward(self, t_input):
        """
        根据单词id，找到五笔拼音的id，过LSTM得到的向量作为单词的表示
        :param t_input: (seq_len, batch_size)
        :return:
        """
        token_data = torch.Tensor([token_indexing(i, self.encoding_type, 'tokens') for i in t_input])
        token_len = torch.Tensor([token_indexing(i, self.encoding_type, 'lens') for i in t_input])
        tokens = t_input.new().long().new(*token_data.shape).copy_(token_data)
        token_lens = t_input.new().float().new(*token_len.shape).copy_(token_len)
        te = self.embedding(tokens)  # (batch_size, num_char, emb_size)
        reshaped_embeddings = te.permute(1, 0, 2)

        h0 = t_input.new().float().new(2, t_input.size()[0], self.hidden_size).zero_()
        to, _ = self.composing(reshaped_embeddings, h0)  # (seq_len, batch, num_directions * hidden_size)
        reshaped_outputs = to.permute(1, 0, 2)
        max_out, _ = torch.max(reshaped_outputs * token_lens.unsqueeze(-1), 1)  # (seq_len, batch, 2 * emb_size)
        return max_out

    def init_weight(self):
        weight = next(self.parameters())  # weight只是想获得别的参数的数据类型和存储位置
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                    weight.new_zeros(self.num_layers, batch_size, self.hidden_size))
        else:
            return weight.new_zeros(self.num_layers, batch_size, self.hidden_size)


def token_indexing(idx, encoding_type, return_type):
    """
    将输入的单词id映射为每个字五笔、拼音的字符的id
    :param idx: (seq_len, batch_size)
    :return: chars: (seq_len, batch_size, num_char)  token_lens: (seq_len, batch_size, num_char)
    """
    c = dict_word['idx2word'][idx]
    if c == '<eos>':
        c = '。'
    if encoding_type == 'wubi':
        encoding = wubi(c)[0] if wubi(c) else c
        full_encoding = encoding if len(encoding) == 8 else encoding + '。' * (8 - len(encoding))
        assert len(full_encoding) == 8, full_encoding
        tokens = [dict_wubi['char2idx'][c] for c in full_encoding]
        length = [i < len(encoding) for i in range(len(tokens))]
    elif encoding_type == 'pinyin':
        encoding = pinyin(c)[0][0] if pinyin(c) else c
        full_encoding = encoding if len(encoding) == 8 else encoding + '。' * (8 - len(encoding))
        assert len(full_encoding) == 8, full_encoding
        tokens = [dict_pinyin['char2idx'][c] for c in full_encoding]
        length = [i < len(encoding) for i in range(len(tokens))]
    else:
        raise NotImplementedError
    # print(idx, c, encoding, tokens, length)
    return tokens if return_type == 'tokens' else length



class Highway(nn.Module):
    def __init__(self, input_size, output_size, num_layers, f):
        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(output_size if i else input_size, output_size) for i in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(output_size if i else input_size, output_size) for i in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(output_size if i else input_size, output_size) for i in range(num_layers)])
        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x

if __name__ == '__main__':
    for i in range(4000):
        token_indexing(i, 'pinyin', 'tokens')
