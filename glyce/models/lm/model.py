# encoding: utf-8
"""
@author: wuwei 
@contact: wu.wei@pku.edu.cn

@version: 1.0
@license: Apache Licence
@file: glyph_embedding_for_lm.py
@time: 19-1-17 下午8:54

用于language modeling的模型
"""

import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-4])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch.nn as nn


from glyce import CharGlyceEmbedding 
from glyce import WordGlyceEmbedding 




class GlyphEmbeddingForLM(nn.Module):
    def __init__(self, model_config):
        super(GlyphEmbeddingForLM, self).__init__()
        self.config = model_config
        self.glyph_embedding = CharGlyceEmbedding(model_config) if model_config.level == 'char' else \
            WordGlyceEmbedding(model_config)
        self.drop = nn.Dropout(model_config.dropout)
        if self.config.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.config.rnn_type)(self.config.output_size, self.config.nhid, self.config.nlayers,
                                                         dropout=self.config.dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.config.rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.config.output_size, self.config.nhid, self.config.nlayers, nonlinearity=nonlinearity,
                              dropout=self.config.dropout)


        self.decoder = nn.Linear(model_config.nhid, len(model_config.idx2char) if model_config.level == 'char' else len(model_config.idx2word))
        self.init_weights()

    def forward(self, data, hidden):
        emb, glyph_classification_loss = self.glyph_embedding(data)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)  # 过线性层之前reshape一下
        reshaped = output.view(output.size(0) * output.size(1), output.size(2))
        decoded = self.decoder(reshaped)
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return decoded, hidden, glyph_classification_loss

    def init_weights(self):
        initrange = 0.1  # (-0.1, 0.1)的均匀分布，只对embedding和最后的线性层做初始化
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):  # 用全零向量初始化隐层并返回
        weight = next(self.parameters())  # weight只是想获得别的参数的数据类型和存储位置
        if self.config.rnn_type == 'LSTM':
            return (weight.new_zeros(self.config.nlayers, batch_size, self.config.nhid),
                    weight.new_zeros(self.config.nlayers, batch_size, self.config.nhid))
        else:
            return weight.new_zeros(self.config.nlayers, batch_size, self.config.nhid)
