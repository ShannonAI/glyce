# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-01-05 23:15:17


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-5])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch.nn as nn

from glyce.models.latticeLSTM.model.bilstm import BiLSTM
from glyce.models.latticeLSTM.model.crf import CRF


class BiLSTMCRF(nn.Module):
    def __init__(self, data):
        super(BiLSTMCRF, self).__init__()
        print("build batched lstmcrf...")
        self.gpu = data.HP_gpu
        self.glyph_ratio = data.HP_glyph_ratio
        #  add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2
        self.lstm = BiLSTM(data)
        self.crf = CRF(label_size, self.gpu)
        self.iteration = 0

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs,
                                char_seq_lengths, char_seq_recover, batch_label, mask):
        outs, glyph_loss = self.lstm.get_output_score(gaz_list, word_inputs, biword_inputs, word_seq_lengths,
                                                      char_inputs, char_seq_lengths, char_seq_recover)
        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        scores, tag_seq = self.crf.viterbi_decode(outs, mask)
        self.iteration += 1
        if self.iteration % 500 == 0:
            self.glyph_ratio *= 0.8
        return total_loss + self.glyph_ratio * glyph_loss, tag_seq

    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                char_seq_recover, mask):
        outs, glyph_loss = self.lstm.get_output_score(gaz_list, word_inputs, biword_inputs, word_seq_lengths,
                                                      char_inputs, char_seq_lengths, char_seq_recover)
        scores, tag_seq = self.crf.viterbi_decode(outs, mask)
        return tag_seq

    def get_lstm_features(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                          char_seq_recover):
        return self.lstm.get_lstm_features(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs,
                                           char_seq_lengths, char_seq_recover)
