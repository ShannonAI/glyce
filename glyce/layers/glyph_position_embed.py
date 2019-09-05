#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.03.23 
# First create: 2019.03.23 
# Description:
# glyph_position_embedder 



import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import json 
import math 
import shutil 
import tarfile 
import logging 
import tempfile


import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss 


from glyce.layers.bert_basic_model import * 
from glyce.utils.tokenization import BertTokenizer  
from glyce.layers.char_glyph_embedding import CharGlyphEmbedding




class GlyphPositionEmbedder(nn.Module):
    def __init__(self, config):
        super(GlyphPositionEmbedder, self).__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.output_size)

        token_tool = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=False)
        idx2tokens = token_tool.ids_to_tokens 
        self.glyph_encoder = CharGlyphEmbedding(config, idx2tokens)

        self.layer_norm = BertLayerNorm(config.output_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1) 
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        glyph_embeddings, glyph_cls_loss = self.glyph_encoder(input_ids)

        embeddings = position_embeddings  + glyph_embeddings 

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, glyph_cls_loss 


if __name__ == "__main__":
    pass 
