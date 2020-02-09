#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Contact: xiaoya_li@shannonai.com 
# Last update: 2019.03.25 
# First create: 2019.03.23 
# Description:
# glyce_transformer.py 


import os 
import sys 

root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import json
import math
import copy
import logging
import tarfile
import tempfile
import shutil
import numpy as np


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


from glyce.layers.bert_basic_model import *
from glyce.layers.glyph_position_embed import GlyphPositionEmbedder



class GlyceTransformer(nn.Module):
    def __init__(self, config, num_labels=4):
        super(GlyceTransformer, self).__init__()
        self.num_labels = num_labels
        self.glyph_embedder = GlyphPositionEmbedder(config.glyph_config)
        bert_config = BertConfig.from_dict(config.bert_config.to_dict())
        self.bert_model = BertModel(bert_config)
        self.transformer_layer = BertEncoder(config.transformer_config)
        self.pooler = BertPooler(config)
        self.bert_model = self.bert_model.from_pretrained(config.glyph_config.bert_model)
        if config.bert_frozen == "true":
            print("!=!"*20)
            print("Please notice that the bert model if frozen")
            print("the loaded weights of models is ")
            print(config.glyph_config.bert_model)
            print("!-!"*20)
            for param in self.bert_model.parameters():
                param.requires_grad=False

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,):
        glyph_embed, glyph_cls_loss = self.glyph_embedder(input_ids, token_type_ids=token_type_ids)
        sequence_output, pooled_output = self.bert_model(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        context_bert_output = sequence_output[-1]
        input_features = torch.cat([glyph_embed, context_bert_output], -1)


        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_tyep_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * - 10000.0
        encoded_layers = self.transformer_layer(input_features, extended_attention_mask,
            output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        return encoded_layers, pooled_output, glyph_cls_loss
